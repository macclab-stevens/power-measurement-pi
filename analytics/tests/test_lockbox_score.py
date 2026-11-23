"""Lockbox scoring on synthetic lockbox cells: truth, Mode A, calculator, Mode L, probe, and the integrity rules."""
import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

import cli
import deploy
import lockbox as lb
import lockbox_score as ls
import mode_a
import mode_a_models as mm
import mode_l
import mode_l_models as mlm
import report_inputs as ri
from synth_lockbox import CONFIGS, MASTER
from test_lockbox import idle
from test_mode_l import make_windows

NAN = float("nan")
MODELS = ("yolo12n", "yolo11n-seg", "resnet18", "mbv3")
IMGSZ = {"yolo12n": 640, "yolo11n-seg": 640, "resnet18": 224, "mbv3": 224}
NON_PRIMARY = "yolo12n_r2"


def inputs():
    """(cat, der, qc_train, qc_l, w): 3 cells per model (the last yolo12n one is not QC-primary) and 2 training cells."""
    w, cat_l = make_windows(models=MODELS, cells_per_model=3, seed=5)
    cat_l = cat_l.assign(source="lockbox", path="", task="image", has_counters=True, serve_fps=NAN, soak_minutes=NAN,
                         concurrent_mode=None, input_source="repeat", n_quantized_modules=0, quant="fp32",
                         mode_a_eligible=False, imgsz=cat_l["base_model"].map(IMGSZ))
    cat_t = pd.DataFrame({"cell_id": ["train_r0", "train_r1"], "source": "v2", "base_model": "yolov8n", "variant": "yolov8n",
                          "precision": "fp32", "imgsz": 640, "batch": 1, "threads": 4, "cpu_mhz": 2400, "rig": "nov2026",
                          "governor": "pinned", "fan_mode": "full", "task": "image", "has_counters": False,
                          "mode_a_eligible": True})
    cat = pd.concat([cat_t, cat_l], ignore_index=True)
    rows = []
    for i, r in enumerate(cat.itertuples()):
        lock = r.source == "lockbox"
        rows.append({"cell_id": r.cell_id, "mode": "forward", "batch": 1, "energy_mJ_per_image": 1000.0 + 10 * i,
                     "latency_ms_per_image": 100.0 + i, "mean_P_W": 7.0 + 0.1 * i, "baseline_W": 5.0 if r.cell_id.endswith("r0") else 5.2,
                     "serve_n_requests": 10 if lock else NAN, "serve_dur_ms": 10 * 300.0 if lock else NAN,
                     "serve_E_mJ": 20000.0 + 100 * i if lock else NAN})
    der = pd.DataFrame(rows)
    qc_l = pd.DataFrame({"cell_id": cat_l["cell_id"], "primary": cat_l["cell_id"] != NON_PRIMARY})
    qc_t = pd.DataFrame({"cell_id": ["train_r0", "train_r1"], "primary": True})
    return cat, der, qc_t, qc_l, w


@pytest.fixture
def env(lockbox_env, tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(lockbox_env.data))
    log = tmp_path / "runs.jsonl"
    shutil.copy(lockbox_env.paths.runs_log, log)
    paths = dataclasses.replace(lockbox_env.paths, runs_log=log, out_dir=tmp_path / "out")
    lb.run_predict(paths, CONFIGS, lockbox_env.static_dir, MASTER, t=lockbox_env.t, desc_train=lockbox_env.desc, idle=idle())
    cat, der, qc_t, qc_l, w = inputs()
    return paths, dict(cat=cat, der=der, qc_train=qc_t, qc_l=qc_l, w=w), lockbox_env


def score(env):
    paths, kw, _ = env
    return ls.run_score(paths, MASTER, **kw)


def forbid_fits(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a fit was called while scoring the lockbox")
    for cls in [*mm.CANDIDATES.values(), mlm.L0Lookup, mlm.L1Pmic, mlm.L1CellPmic, mlm.L2Busy, mlm.L3Counters,
                mlm.L4CountersPmic, HistGradientBoostingRegressor, Ridge, LinearRegression]:
        monkeypatch.setattr(cls, "fit", boom)
    for mod, name in ((mode_a, "fit_final"), (mode_l, "fit_final"), (mode_l, "_fit"), (mode_l, "_select"), (mode_l, "run_lomo")):
        monkeypatch.setattr(mod, name, boom)


def test_score_never_fits_anything(env, monkeypatch):
    forbid_fits(monkeypatch)
    out = score(env)
    assert len(out["mode_a"]) and len(out["mode_l"]) and out["report"].is_file()


def test_score_fails_clearly_when_the_lockbox_is_absent(env):
    paths, kw, _ = env
    cat = kw["cat"][kw["cat"]["source"] != "lockbox"]
    with pytest.raises(lb.LockboxError, match="no lockbox cells"):
        ls.run_score(paths, MASTER, **{**kw, "cat": cat})


def test_cli_lockbox_score_reports_an_absent_lockbox_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    assert cli.main(["lockbox-score"]) == 1
    err = capsys.readouterr().err
    assert "no lockbox cells" in err and str(tmp_path / "lockbox" / "cells") in err


def test_truth_uses_only_primary_cells_without_the_mode_a_eligible_filter(env):
    _, kw, _ = env
    cat_l = kw["cat"][kw["cat"]["source"] == "lockbox"]
    t = ls.truth_table(cat_l, kw["der"], kw["qc_l"]).set_index("base_model")
    assert t.loc["yolo12n", "n_cells"] == 2 and NON_PRIMARY not in t.loc["yolo12n", "cell_ids"]
    d = kw["der"].set_index("cell_id")
    want = d.loc[["yolo12n_r0", "yolo12n_r1"], "energy_mJ_per_image"].mean()
    assert t.loc["yolo12n", "energy_mJ"] == pytest.approx(want) and t.loc["mbv3", "n_cells"] == 3
    assert t.loc["resnet18", "imgsz"] == 224


def test_mode_a_ape_interval_and_pooled_mdape(env):
    out = score(env)
    s = out["mode_a"]
    pred = pd.read_csv(env[0].out_dir / lb.PREDICTIONS_CSV)
    r = s.query("model == 'yolo12n' and imgsz == 640 and candidate == 'M1Ridge' and target == 'energy_mJ'").iloc[0]
    truth = out["truth"].set_index("base_model").loc["yolo12n", "energy_mJ"]
    p = pred.query("model == 'yolo12n' and imgsz == 640 and candidate == 'M1Ridge' and target == 'energy_mJ'").iloc[0]
    assert r["pred"] == p["pred"] and r["ape"] == pytest.approx(abs(p["pred"] - truth) / truth)
    assert r["inside"] == (p["lo90"] <= truth <= p["hi90"]) and r["flagged"] == p["flagged"]
    nomatch = s[(s["model"] == "yolo12n") & (s["imgsz"] == 320)]       # no lockbox cell: counted, not scored
    assert nomatch["ape"].isna().all() and (nomatch["n_cells"] == 0).all()
    sm = out["summary_a"].query("candidate == 'M1Ridge' and target == 'energy_mJ'").set_index("group")
    ok = s[(s["candidate"] == "M1Ridge") & (s["target"] == "energy_mJ") & s["ape"].notna()]
    assert sm.loc["all", "n"] == len(ok) == 4 and sm.loc["all", "mdape"] == pytest.approx(ok["ape"].median())
    new = ok[ok["new_model"]]
    assert sm.loc["new models", "n"] == len(new) == 2 and set(new["model"]) == {"yolo12n", "yolo11n-seg"}


def test_calculator_uses_frozen_predictions_measured_duration_and_training_idle(env):
    out = score(env)
    c = out["calculator"].set_index("cell_id")
    assert NON_PRIMARY not in c.index and len(c) == 11
    pred = pd.read_csv(env[0].out_dir / lb.PREDICTIONS_CSV).query("role == 'primary' and model == 'yolo12n' and imgsz == 640")
    e, t = (pred[pred["target"] == k]["pred"].iloc[0] for k in ("energy_mJ", "latency_ms"))
    der = env[1]["der"].set_index("cell_id")
    scen = {"cpu_key": "2400", "fan_mode": "full", "rig": "nov2026", "n_images": 10.0, "duration_s": 3.0}
    want = deploy.predict(scen, e, t, pd.DataFrame([("nov2026", "2400", "full", 5.1)], columns=["rig", "cpu_key", "fan_mode", "P_idle_W"]),
                          None, None)["E_J"]                       # idle = median of the two training baselines
    row = c.loc["yolo12n_r0"]
    assert row["predicted_J"] == pytest.approx(want) and row["measured_J"] == pytest.approx(der.loc["yolo12n_r0", "serve_E_mJ"] / 1000)
    assert row["ape"] == pytest.approx(abs(want - row["measured_J"]) / row["measured_J"])


def test_mode_l_scores_frozen_rungs_per_cell_and_reports_non_yolo_separately(env):
    out = score(env)
    c = out["mode_l"]
    assert set(c["rung"]) == set(mlm.default_rungs()) and NON_PRIMARY not in set(c["cell_id"])
    rec = ri.latest_record(ri.read_records(env[0].runs_log), command="final-mode-l", part="fit_final", rung="L2")
    model = lb.load_frozen(rec)
    w = env[1]["w"]
    cell = w[w["cell_id"] == "mbv3_r1"].sort_values("w_idx")
    p = model.predict(cell)
    e_true = cell["E_mJ"].sum() / 1000
    got = c.query("cell_id == 'mbv3_r1' and rung == 'L2'").iloc[0]
    assert got["ape"] == pytest.approx(abs((p * cell["dur_s"]).sum() - e_true) / e_true)
    assert {"ape_meas", "ape_1s", "ape_10s"} <= set(c.columns)
    sm = out["mode_l_summary"]
    assert set(sm["family"]) == {"YOLO family", "non-YOLO"}
    nony = c[c["model"].isin(["resnet18", "mbv3"]) & (c["rung"] == "L2")]
    assert sm.query("family == 'non-YOLO' and rung == 'L2'")["median_ape"].iloc[0] == pytest.approx(nony["ape"].median())


def test_probe_pairs_per_configuration_against_frozen_mode_a(env):
    out = score(env)
    pr = out["probe"]
    assert len(pr) == len(mlm.default_rungs()) * (2 * 1 + 3 * 2 + 3 * 2 + 3 * 2)   # yolo12n 2 primary cells: 2 pairs
    assert NON_PRIMARY not in set(pr["j"]) | set(pr["k"]) and (pr["j"] != pr["k"]).all()
    pred = pd.read_csv(env[0].out_dir / lb.PREDICTIONS_CSV).query("role == 'primary' and target == 'energy_mJ'")
    a = pred.query("model == 'resnet18'")["pred"].iloc[0]
    assert (pr[pr["config_key"].str.startswith("resnet18")]["mode_a_mJ"] == a).all()
    one = pr.query("rung == 'L0' and j == 'resnet18_r0' and k == 'resnet18_r1'").iloc[0]
    d = env[1]["der"].set_index("cell_id")
    assert one["true_mJ"] == d.loc["resnet18_r1", "energy_mJ_per_image"]
    cells = out["mode_l"].query("cell_id == 'resnet18_r0' and rung == 'L0'").iloc[0]
    assert one["probe_mJ"] == pytest.approx(cells["mean_P_pred_meas_W"] * d.loc["resnet18_r0", "latency_ms_per_image"])


def test_report_and_outputs_record_hashes_and_the_scored_once_statement(env):
    paths = env[0]
    out = score(env)
    text = out["report"].read_text()
    sha = lb.runlog.file_hash(paths.out_dir / lb.PREDICTIONS_CSV)
    assert sha in text and "Scored once; any later change is exploratory" in text
    for r in ri.read_records(paths.runs_log):
        if r.get("command") in ("final-mode-l", "mode-a-extras") and r.get("part") == "fit_final":
            assert r["sha256"] in text
    rec = [r for r in ri.read_records(paths.runs_log) if r.get("command") == "lockbox-score"][-1]
    assert rec["predictions_sha256"] == sha and rec["run_id"] == out["run_id"] and rec["predictions_found_in_runlog"]
    import settings
    for part in ("truth", "mode_a_scores", "calculator", "mode_l_cells", "probe"):
        assert (settings.derived_dir() / "lockbox" / f"{out['run_id']}_{part}.parquet").is_file()
    assert lb.runlog.file_hash(paths.out_dir / lb.PREDICTIONS_CSV) == sha


def test_a_second_scoring_is_marked_as_a_repeat(env):
    score(env)
    out = score(env)
    assert "earlier scoring runs: 1" in out["report"].read_text()


def _rewrite(paths, edit):
    recs = [json.loads(x) for x in paths.runs_log.read_text().splitlines()]
    for r in recs:
        edit(r)
    paths.runs_log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_score_refuses_a_changed_mode_l_pickle(env):
    _rewrite(env[0], lambda r: r.update(sha256="a" * 64) if r.get("command") == "final-mode-l" and r["rung"] == "L3-trees" else None)
    with pytest.raises(lb.LockboxError, match="changed after it was frozen"):
        score(env)


def test_score_refuses_predictions_written_with_other_models(env):
    paths = env[0]
    pred = pd.read_csv(paths.out_dir / lb.PREDICTIONS_CSV)
    pred.loc[pred["candidate"] == "B3LogLinear", "model_sha256"] = "c" * 64
    pred.to_csv(paths.out_dir / lb.PREDICTIONS_CSV, index=False)
    with pytest.raises(lb.LockboxError, match="was written with"):
        score(env)


def test_score_requires_the_committed_predictions(env):
    (env[0].out_dir / lb.PREDICTIONS_CSV).unlink()
    with pytest.raises(lb.LockboxError, match="lockbox-predict"):
        score(env)


def test_cells_outside_the_lockbox_design_are_rejected(env):
    paths, kw, _ = env
    cat = kw["cat"].copy()
    cat.loc[cat["cell_id"] == "mbv3_r0", "cpu_mhz"] = 1500
    with pytest.raises(ValueError, match="mbv3_r0.*cpu_mhz"):
        ls.run_score(paths, MASTER, **{**kw, "cat": cat})
    cat = kw["cat"].copy()
    cat.loc[cat["cell_id"] == "mbv3_r1", "serve_fps"] = 5.0
    with pytest.raises(ValueError, match="serve_fps"):
        ls.run_score(paths, MASTER, **{**kw, "cat": cat})


def test_the_four_commands_are_registered_with_their_arguments():
    p = cli.build_parser()
    a = p.parse_args(["lockbox-predict", "--config", "yolo12n:640", "--config", "mbv3:224", "--static-dir", "d", "--seed", "5"])
    assert a.config == ["yolo12n:640", "mbv3:224"] and a.static_dir == "d" and a.seed == 5 and not a.force
    assert p.parse_args(["final-mode-l", "--rung", "L0", "--rung", "L2", "--n-jobs", "1"]).rung == ["L0", "L2"]
    assert p.parse_args(["final-mode-a-secondary", "--seed", "7"]).seed == 7 and p.parse_args(["lockbox-score"]).seed is None
    with pytest.raises(SystemExit):
        p.parse_args(["lockbox-predict"])                      # --static-dir is required


def test_bad_config_specs_are_rejected():
    for bad in ("yolo12n", "yolo12n:", ":640", "yolo12n:big"):
        with pytest.raises(lb.LockboxError, match="bad --config"):
            lb.parse_config(bad)
    assert lb.parse_config("yolo11n-seg:640") == ("yolo11n-seg", 640)
