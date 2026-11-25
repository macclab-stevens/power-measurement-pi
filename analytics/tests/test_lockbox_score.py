"""Lockbox scoring on synthetic lockbox cells: truth, Mode A, calculator, Mode L, probe, and the integrity rules."""
import dataclasses
import json
import shutil
from datetime import datetime, timedelta, timezone

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
MODELS = ("yolo12n", "yolov8n", "resnet18", "mbv3")
IMGSZ = {"yolo12n": 640, "yolov8n": 480, "resnet18": 224, "mbv3": 224}
NON_PRIMARY = "yolo12n_r2"


def inputs():
    """(cat, der, qc_train, qc_l, w): 3 cells per model (the last yolo12n one is not QC-primary) and 2 training cells."""
    w, cat_l = make_windows(models=MODELS, cells_per_model=3, seed=5)
    cat_l = cat_l.assign(source="lockbox", path="", task="image", has_counters=True, serve_fps=NAN, soak_minutes=NAN,
                         concurrent_mode=None, input_source="repeat", n_quantized_modules=0, quant="fp32",
                         mode_a_eligible=False, imgsz=cat_l["base_model"].map(IMGSZ))
    tr = [("train_r0", "yolov8n", 640), ("train_r1", "yolov8n", 640), ("train_resnet18_r0", "resnet18", 224),
          ("train_mbv3_r1", "mbv3", 224)]
    cat_t = pd.DataFrame({"cell_id": [c for c, _, _ in tr], "source": "v2", "base_model": [m for _, m, _ in tr],
                          "variant": [m for _, m, _ in tr], "precision": "fp32", "imgsz": [z for _, _, z in tr], "batch": 1,
                          "threads": 4, "cpu_mhz": 2400, "rig": "nov2026", "governor": "pinned", "fan_mode": "full",
                          "task": "image", "has_counters": False, "mode_a_eligible": True})
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
    qc_t = pd.DataFrame({"cell_id": cat_t["cell_id"], "primary": True})
    return cat, der, qc_t, qc_l, w


@pytest.fixture
def env(lockbox_env, tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(lockbox_env.data))
    log = tmp_path / "runs.jsonl"
    shutil.copy(lockbox_env.paths.runs_log, log)
    paths = dataclasses.replace(lockbox_env.paths, runs_log=log, out_dir=tmp_path / "out")
    lb.run_predict(paths, CONFIGS, lockbox_env.static_dir, MASTER, t=lockbox_env.t, desc_train=lockbox_env.desc, idle=idle())
    cat, der, qc_t, qc_l, w = inputs()
    cells = tmp_path / "cells"                                      # every lockbox cell has the layer table of its configuration
    for i, r in cat[cat["source"] == "lockbox"].iterrows():
        d = cells / r["cell_id"]
        d.mkdir(parents=True)
        shutil.copy(lockbox_env.static_dir / f"{r['base_model']}_{r['imgsz']}.csv", d / "layer_table.csv")
        cat.loc[i, "path"] = str(d)
    later = datetime.now(timezone.utc) + timedelta(days=1)          # the lockbox is measured after the predictions
    return paths, dict(cat=cat, der=der, qc_train=qc_t, qc_l=qc_l, w=w, starts={c: later for c in qc_l["cell_id"]},
                       static_dir=lockbox_env.static_dir), lockbox_env


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
    assert sm.loc["new models", "n"] == len(new) == 1 and set(new["model"]) == {"yolo12n"}
    size = ok[ok["new_size"]]
    assert set(size["model"]) == {"yolov8n"} and sm.loc["new size of a training model", "n"] == 1   # 480 px, never trained


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
    assert set(c["rung"]) == set(mlm.R1_RUNGS) and NON_PRIMARY not in set(c["cell_id"])
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
    assert len(pr) == len(mlm.R1_RUNGS) * (2 * 1 + 3 * 2 + 3 * 2 + 3 * 2)   # yolo12n 2 primary cells: 2 pairs
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
    assert rec["predictions_sha256"] == sha and rec["run_id"] == out["run_id"] and rec["predict_run_id"] == out["rec"]["predict_run_id"]
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
    with pytest.raises(lb.LockboxError, match="refitted after the predictions"):
        score(env)


def test_score_refuses_predictions_written_with_other_models(env):
    paths = env[0]
    pred = pd.read_csv(paths.out_dir / lb.PREDICTIONS_CSV)
    pred.loc[pred["candidate"] == "B3LogLinear", "model_sha256"] = "c" * 64
    pred.to_csv(paths.out_dir / lb.PREDICTIONS_CSV, index=False)
    sha = lb.runlog.file_hash(paths.out_dir / lb.PREDICTIONS_CSV)
    _rewrite(paths, lambda r: r.update(predictions_sha256=sha) if r.get("command") == "lockbox-predict" else None)
    man = paths.out_dir / lb.MANIFEST
    man.write_text(json.dumps({**json.loads(man.read_text()), "predictions_sha256": sha}))     # a consistent forgery of the file
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
    assert lb.parse_config("yolov8n:480") == ("yolov8n", 480)


# ------------------------------------------------------------------ freeze verification

def test_score_refuses_when_a_rung_is_refitted_after_predicting(env, tmp_path, monkeypatch):
    paths, _, e = env
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "other_data"))        # the frozen pickles stay untouched
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"), seed=99)
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    _, info = mode_l.fit_final(w, cat.assign(source="v2"), qc, "L2", MASTER, mlm.first_point({k: v for k, v in mlm.default_rungs().items() if k in mlm.R1_RUNGS}))
    frozen = {r["rung"]: r for r in ri.read_records(paths.runs_log) if r.get("command") == "final-mode-l"}["L2"]
    assert info["sha256"] != frozen["sha256"]
    lb.runlog.append({"command": "final-mode-l", "part": "fit_final", "seed": MASTER, "rung": "L2", **info}, paths.runs_log)
    monkeypatch.setenv("ANALYTICS_DATA", str(e.data))
    with pytest.raises(lb.LockboxError, match="rung L2 was refitted after the predictions"):
        score(env)


def test_score_refuses_a_mode_l_pickle_that_no_longer_matches_its_line(env, tmp_path):
    paths, _, _ = env
    victim = [r for r in ri.read_records(paths.runs_log) if r.get("command") == "final-mode-l" and r["rung"] == "L3-trees"][-1]
    bad = tmp_path / "mode_l_L3-trees.pkl"
    bad.write_bytes(lb.frozen_path(victim).read_bytes() + b"\0")
    _rewrite(paths, lambda r: r.update(path=str(bad)) if r.get("run_id") == victim["run_id"] else None)
    with pytest.raises(lb.LockboxError, match="changed after it was frozen"):
        score(env)


def test_score_refuses_predictions_that_are_not_in_the_runlog(env):
    paths = env[0]
    recs = [x for x in paths.runs_log.read_text().splitlines() if '"lockbox-predict"' not in x]
    paths.runs_log.write_text("\n".join(recs) + "\n")
    with pytest.raises(lb.LockboxError, match="no `lockbox-predict` line"):
        score(env)


def test_score_refuses_predictions_written_after_the_cells_started(env):
    paths, kw, _ = env
    early = datetime.now(timezone.utc) - timedelta(days=1)
    with pytest.raises(lb.LockboxError, match="after the earliest lockbox cell started"):
        ls.run_score(paths, MASTER, **{**kw, "starts": {**kw["starts"], "mbv3_r1": early}})


def test_score_refuses_a_missing_or_disagreeing_manifest(env):
    paths = env[0]
    man = paths.out_dir / lb.MANIFEST
    good = man.read_text()
    man.write_text(json.dumps({**json.loads(good), "mode_l": {}}))
    with pytest.raises(lb.LockboxError, match="disagrees with the `lockbox-predict` runlog line"):
        score(env)
    man.unlink()
    with pytest.raises(lb.LockboxError, match=lb.MANIFEST):
        score(env)


def test_probe_carries_the_frozen_primary_energy_run_id(env):
    out = score(env)
    rec = [r for r in ri.read_records(env[0].runs_log) if r.get("command") == "mode-a-extras" and r["target"] == "energy_mJ"][-1]
    assert set(out["probe"]["mode_a_run_id"]) == {rec["run_id"]}


def test_design_check_also_covers_input_quant_and_variant(env):
    paths, kw, _ = env
    for col, val in (("input_source", "camera"), ("quant", "int8"), ("variant", "yolo12n_pruned")):
        cat = kw["cat"].copy()
        cat.loc[cat["cell_id"] == "mbv3_r0", col] = val
        with pytest.raises(ValueError, match=f"mbv3_r0.*{col}"):
            ls.run_score(paths, MASTER, **{**kw, "cat": cat})


def test_novelty_note_is_in_the_report(env):
    assert "320 px configuration uses its 640 px architecture descriptor" in score(env)["report"].read_text()


def test_cell_starts_take_the_earlier_of_run_meta_and_the_index(tmp_path, monkeypatch):
    from conftest import make_cell
    import data
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t0 = datetime(2026, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
    make_cell(tmp_path, "lockbox", "a_r00", meta={"t0_real_ns": int(t0.timestamp() * 1e9)}, index={"started": "2026-12-01T05:00:00"})
    make_cell(tmp_path, "lockbox", "b_r00", meta={"t0_real_ns": int(t0.timestamp() * 1e9)})
    cat = data.catalogue()
    st = ls.cell_starts(cat[cat["source"] == "lockbox"])
    assert st["b_r00"] == t0
    assert st["a_r00"] == datetime.fromisoformat("2026-12-01T05:00:00").astimezone(timezone.utc) < t0


@pytest.mark.parametrize("argv", [["final-mode-l"], ["final-mode-a-secondary"], ["mode-a-extras"]])
def test_final_commands_refuse_once_lockbox_cells_exist(argv, tmp_path, monkeypatch, capsys):
    (tmp_path / "lockbox" / "cells" / "x_r00").mkdir(parents=True)
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "lockbox cells already exist" in err and argv[0] in err
    assert not (tmp_path / "derived" / "models").exists()


# ------------------------------------------------------------------ post hoc (R2) gap calculator

def _gap_frames():
    cat_l = pd.DataFrame({"cell_id": ["a_r0", "b_r0", "np_r0"], "source": "lockbox", "base_model": ["yolov8n", "resnet18", "yolov8n"],
                          "imgsz": [480, 224, 480], "batch": 1, "cpu_mhz": 2400, "governor": "pinned", "fan_mode": "full",
                          "rig": "nov2026", "path": ""})                                     # path "": default 200 ms gap
    der = pd.DataFrame({"cell_id": cat_l["cell_id"], "mode": "forward", "serve_n_requests": 10, "serve_dur_ms": [5900.0, 5400.0, 5900.0],
                        "serve_E_mJ": [44000.0, 41000.0, 44000.0], "latency_ms_per_image": 300.0, "energy_mJ_per_image": 3000.0})
    qc_l = pd.DataFrame({"cell_id": cat_l["cell_id"], "primary": [True, True, False]})
    rows = []
    for m, z in (("yolov8n", 480), ("resnet18", 224)):
        for tgt, v in (("energy_mJ", 3000.0), ("latency_ms", 300.0)):
            rows.append({"model": m, "imgsz": z, "target": tgt, "role": "primary", "pred": v, "lo90": 0.9 * v, "hi90": 1.1 * v,
                         "flagged": False, "novelty": 0.0 if m == "resnet18" else 1.0})
    idle_t = pd.DataFrame([("nov2026", "2400", "full", 5.0)], columns=["rig", "cpu_key", "fan_mode", "P_idle_W"])
    train = pd.DataFrame([{"cell_id": f"t{i}", "base_model": m, "serve_n_requests": 10, "batch": 1, "serve_dur_ms": d,
                           "latency_ms_per_image": 300.0, "serve_fps": NAN, "serve_gap_ms": 200.0}
                          for i, (m, d) in enumerate([("yolo11n", 5500.0), ("yolo11s", 5500.0), ("effb0", 5200.0)])])
    return cat_l, der, qc_l, pd.DataFrame(rows), idle_t, train


def test_calculator_posthoc_gap_two_variants_beside_the_frozen_known_duration():
    cat_l, der, qc_l, pred, idle_t, train = _gap_frames()
    g = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle_t, train_cells=train).set_index("cell_id")
    assert list(g.index) == ["a_r0", "b_r0"] and list(g.columns) == ls.POSTHOC_GAP_COLS[1:]      # non-primary cell dropped
    a = g.loc["a_r0"]
    # overhead 0: 10 x (0.3 + 0.2) = 5 s, busy 3 s, 2 s idle at 5 W
    assert a["predicted_J_oh0"] == pytest.approx(30.0 + 2.0 * 5.0) and a["gap_s"] == pytest.approx(0.2)
    # yolo overhead = median of (5.5 - 3 - 2) / 10 = 0.05 s -> 5.5 s, 2.5 s idle; resnet18 (other): 0.02 s
    assert a["overhead_est_s"] == pytest.approx(0.05) and a["predicted_J_est"] == pytest.approx(30.0 + 2.5 * 5.0)
    assert g.loc["b_r0", "overhead_est_s"] == pytest.approx(0.02) and g.loc["b_r0", "predicted_J_est"] == pytest.approx(30.0 + 2.2 * 5.0)
    # the frozen calculator's number: the measured 5.9 s duration, shown only as the reference
    assert a["measured_duration_s"] == pytest.approx(5.9) and a["predicted_J_known"] == pytest.approx(30.0 + 2.9 * 5.0)
    assert a["ape_oh0"] == pytest.approx(abs(40.0 - 44.0) / 44.0) and a["ape_est"] == pytest.approx(abs(42.5 - 44.0) / 44.0)
    assert a["ape_known"] == pytest.approx(abs(44.5 - 44.0) / 44.0)
    s = ls.summarise_posthoc_gap(g.reset_index())
    assert list(s["n"]) == [2, 2, 2] and s["median_ape"].iloc[1] == pytest.approx(g["ape_oh0"].median())


def test_calculator_posthoc_gap_never_reads_the_measured_duration():
    cat_l, der, qc_l, pred, idle_t, train = _gap_frames()
    base = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle_t, train_cells=train)
    other = ls.calculator_posthoc_gap(cat_l, der.assign(serve_dur_ms=99000.0), qc_l, pred, idle_t, train_cells=train)
    for c in ("predicted_J_oh0", "predicted_J_est"):
        assert list(other[c]) == pytest.approx(list(base[c]))
    assert list(other["predicted_J_known"]) != pytest.approx(list(base["predicted_J_known"]))


def test_calculator_posthoc_gap_without_training_cells_has_no_estimated_variant():
    cat_l, der, qc_l, pred, idle_t, _ = _gap_frames()
    g = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle_t)
    assert g["predicted_J_est"].isna().all() and g["ape_oh0"].notna().all() and g.attrs["overhead"] is None


def test_run_posthoc_gap_writes_r2_files_and_a_runlog_line(tmp_path):
    cat_l, der, qc_l, pred, idle_t, train = _gap_frames()
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    pred.assign(candidate="AUTO", config_key="k", model_sha256="x").to_csv(out_dir / lb.PREDICTIONS_CSV, index=False)
    paths = ri.Paths.default().under(out_dir)
    paths = dataclasses.replace(paths, runs_log=tmp_path / "runs.jsonl")
    cat_t = pd.DataFrame({"cell_id": train["cell_id"], "source": "v2", "base_model": train["base_model"], "variant": train["base_model"],
                          "precision": "fp32", "imgsz": 640, "batch": 1, "threads": 4, "cpu_mhz": 2400, "rig": "nov2026",
                          "governor": "pinned", "fan_mode": "full", "task": "image", "serve_fps": NAN, "soak_minutes": NAN,
                          "concurrent_mode": None, "input_source": "repeat", "mode_a_eligible": True})
    cat = pd.concat([cat_t, cat_l.assign(task="image", serve_fps=NAN, soak_minutes=NAN, concurrent_mode=None, input_source="repeat")],
                    ignore_index=True)
    der_t = pd.DataFrame({"cell_id": train["cell_id"], "mode": "forward", "batch": 1, "baseline_W": 5.0, "serve_n_requests": 10,
                          "serve_dur_ms": train["serve_dur_ms"], "serve_E_mJ": 40000.0, "latency_ms_per_image": 300.0,
                          "energy_mJ_per_image": 3000.0})
    out = ls.run_posthoc_gap(paths, cat=cat, der=pd.concat([der_t, der], ignore_index=True),
                             qc_train=pd.DataFrame({"cell_id": train["cell_id"], "primary": True}), qc_l=qc_l)
    assert out["csv"] == out_dir / "r2" / "lockbox_calculator_gap.csv" and out["md"].is_file()
    assert list(pd.read_csv(out["csv"]).columns) == ls.POSTHOC_GAP_COLS
    assert "post hoc (R2)" in out["md"].read_text()
    rec = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[-1])
    assert rec["command"] == "lockbox-calculator-gap" and set(rec["median_ape"]) == {"known", "oh0", "est"}
    assert not (out_dir / "lockbox.md").exists()


def test_lockbox_calculator_gap_command_is_registered():
    assert cli.build_parser().parse_args(["lockbox-calculator-gap"]).command == "lockbox-calculator-gap"
