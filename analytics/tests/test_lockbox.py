"""Lockbox predictions: frozen-model integrity, intervals, novelty and the refusals. Synthetic models only."""
import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import lockbox as lb
import mode_a
import mode_a_models as mm
import mode_l
import mode_l_models as mlm
import report_inputs as ri
import tables
from synth_lockbox import CONFIGS, MASTER, TRAIN


def idle():
    return pd.DataFrame([("nov2026", "2400", "full", 5.0)], columns=["rig", "cpu_key", "fan_mode", "P_idle_W"])


@pytest.fixture
def env(lockbox_env, tmp_path, monkeypatch):
    """A private copy of the output folder and runlog so tests never see each other's files."""
    monkeypatch.setenv("ANALYTICS_DATA", str(lockbox_env.data))
    log = tmp_path / "runs.jsonl"
    shutil.copy(lockbox_env.paths.runs_log, log)
    return dataclasses.replace(lockbox_env.paths, runs_log=log, out_dir=tmp_path / "out"), lockbox_env


def predict(env, configs=CONFIGS, **kw):
    paths, e = env
    return lb.run_predict(paths, configs, e.static_dir, MASTER, t=e.t, desc_train=e.desc, idle=idle(), **kw)


def test_predict_refuses_a_missing_static_csv_naming_file_and_tool_command(env, tmp_path):
    paths, e = env
    (tmp_path / "s").mkdir()
    for f in e.static_dir.glob("*.csv"):
        if f.name != "yolo12n_320.csv":
            shutil.copy(f, tmp_path / "s" / f.name)
    with pytest.raises(lb.LockboxError) as exc:
        lb.run_predict(paths, CONFIGS, tmp_path / "s", MASTER, t=e.t, desc_train=e.desc, idle=idle())
    msg = str(exc.value)
    assert "yolo12n_320.csv" in msg and "static_features.py" in msg and "--imgsz 320" in msg
    assert not (paths.out_dir / lb.PREDICTIONS_CSV).exists()


def test_predictions_have_the_agreed_columns_roles_and_row_count(env):
    out = predict(env)
    p = pd.read_csv(out["path"])
    assert list(p.columns) == lb.PRED_COLS and out["path"].name == "lockbox_predictions.csv"
    assert len(p) == len(CONFIGS) * 7        # 3 primary + B3 x 3 + M3 (energy)
    roles = p.drop_duplicates(["candidate", "target"]).set_index(["candidate", "target"])["role"]
    assert {t: roles[("M1Ridge", t)] for t in ri.TARGETS} == dict.fromkeys(ri.TARGETS, "primary")
    assert roles[("M3Factored", "energy_mJ")] == "secondary" and roles[("B3LogLinear", "mean_P_W")] == "secondary"
    assert ("M3Factored", "latency_ms") not in roles.index
    assert (p["pred"] > 0).all() and (p["lo90"] < p["pred"]).all() and (p["hi90"] > p["pred"]).all()
    assert out["rec"]["command"] == "lockbox-predict" and len(out["rec"]["predictions_sha256"]) == 64


def test_predictions_are_reproducible(env):
    a = predict(env)["path"].read_bytes()
    b = predict(env, force=True)["path"].read_bytes()
    assert a == b


def test_existing_predictions_are_not_overwritten_without_force(env):
    predict(env)
    with pytest.raises(lb.LockboxError, match="already exists"):
        predict(env)


def test_predict_refuses_once_lockbox_cells_exist(env, lockbox_env):
    (lockbox_env.data / "lockbox" / "cells" / "x_r00").mkdir(parents=True)
    try:
        with pytest.raises(lb.LockboxError, match="before"):
            predict(env)
    finally:
        shutil.rmtree(lockbox_env.data / "lockbox")


def test_static_csv_sha256_is_recorded_per_configuration(env):
    out = predict(env)
    p = out["pred"]
    one = p[(p["model"] == "yolo12n") & (p["imgsz"] == 320)]["static_csv_sha256"].unique()
    assert list(one) == [lb.runlog.file_hash(env[1].static_dir / "yolo12n_320.csv")]


def test_interval_is_the_slomo_residual_quantiles_of_that_candidate(env):
    paths, e = env
    p = predict(env)["pred"]
    for target, cand in (("energy_mJ", "M1Ridge"), ("energy_mJ", "M3Factored"), ("latency_ms", "B3LogLinear")):
        s = pd.read_parquet(paths.mode_a_dir / f"{target}.parquet")
        r = s[s["candidate"] == cand]
        resid = np.log(r["y_true"]) - np.log(r["y_pred"])
        q5, q95 = np.quantile(resid, [0.05, 0.95])
        g = p[(p["candidate"] == cand) & (p["target"] == target)]
        assert len(r) == len(e.t)                       # all configurations, every fold
        np.testing.assert_allclose(g["lo90"], g["pred"] * np.exp(q5))
        np.testing.assert_allclose(g["hi90"], g["pred"] * np.exp(q95))


def test_primary_prediction_equals_the_frozen_model_on_the_tool_features(env):
    paths, e = env
    out = predict(env)
    feats, _ = lb.feature_rows([lb.parse_config(c) for c in CONFIGS], e.static_dir)
    rec = ri.latest_record(ri.read_records(paths.runs_log), command="mode-a-extras", part="fit_final", target="energy_mJ")
    want = lb.load_frozen(rec).predict(feats)
    got = out["pred"].query("role == 'primary' and target == 'energy_mJ'")["pred"].to_numpy()
    np.testing.assert_allclose(got, want)


def test_feature_rows_use_the_shared_derived_columns(env):
    paths, e = env
    f, _ = lb.feature_rows([("yolo12n", 640)], e.static_dir)
    r = f.iloc[0]
    assert r["precision"] == "fp32" and r["rig"] == "nov2026" and r["n_quantized_modules"] == 0 and r["is_int8"] == 0 \
        and r["is_fp16"] == 0 and r["cpu_ghz"] == pytest.approx(2.4) and r["log_macs"] == pytest.approx(np.log(r["macs"]))
    again = tables.add_derived_columns(f.drop(columns=["log_macs", "cpu_ghz", "is_int8", "is_fp16", "log_params",
                                                       "log_weight_bytes", "log_act_bytes", "vcore_V", "v2f",
                                                       "log_n_calls"]))
    pd.testing.assert_frame_equal(f, again[f.columns])
    assert r["config_key"] == tables.config_key(r) == "yolo12n|yolo12n|fp32|640|1|4|2400|nov2026"


def test_novelty_is_zero_for_a_training_model_and_positive_for_a_new_one(env):
    p = predict(env)["pred"]
    nov = p.drop_duplicates("model").set_index("model")["novelty"]
    assert nov["resnet18"] == 0 and nov["mbv3"] == 0
    assert nov["yolo12n"] > 0 and nov["yolov8n"] == 0          # yolov8n is a training model (at a new size here)
    assert (p["flagged"] == (p["novelty"] > p["threshold"])).all()
    assert not p[p["model"].isin(["resnet18", "mbv3"])]["flagged"].any()


def test_threshold_is_the_largest_fold_novelty_with_auto_mdape_below_15_percent():
    def fold(name, nov, err):
        return pd.DataFrame({"candidate": "AUTO", "fold": name, "novelty": nov, "y_true": 100.0, "y_pred": [100.0 * (1 + err)] * 3})
    s = pd.concat([fold("a", 1.0, 0.05), fold("b", 2.5, 0.10), fold("c", 4.0, 0.30), fold("d", 3.0, 0.20)])
    assert lb.threshold_from_slomo(s) == 2.5
    assert lb.threshold_from_slomo(pd.concat([fold("c", 4.0, 0.30)])) == 0.0


def test_a_pickle_whose_sha256_differs_from_its_runlog_line_is_refused(env):
    paths, _ = env
    rec = ri.latest_record(ri.read_records(paths.runs_log), command="mode-a-extras", part="fit_final", target="energy_mJ")
    assert lb.load_frozen(rec) is not None
    with pytest.raises(lb.LockboxError, match="changed after it was frozen"):
        lb.load_frozen({**rec, "sha256": "0" * 64})


def test_predict_refuses_a_tampered_frozen_model_before_writing(env, tmp_path):
    paths, e = env
    recs = [json.loads(x) for x in paths.runs_log.read_text().splitlines()]
    for r in recs:
        if r.get("command") == "final-mode-a-secondary" and r["target"] == "mean_P_W":
            r["sha256"] = "f" * 64
    paths.runs_log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    with pytest.raises(lb.LockboxError, match="changed after it was frozen"):
        predict(env)
    assert not (paths.out_dir / lb.PREDICTIONS_CSV).exists()


def test_missing_secondary_fit_names_the_command(env):
    paths, _ = env
    recs = [r for r in paths.runs_log.read_text().splitlines() if "final-mode-a-secondary" not in r]
    paths.runs_log.write_text("\n".join(recs) + "\n")
    with pytest.raises(lb.LockboxError, match="final-mode-a-secondary"):
        predict(env)


def test_serve_prediction_uses_the_200_ms_gap(env):
    out = predict(env)
    s = out["serve"].set_index("model")
    p = out["pred"].query("role == 'primary'")
    t = p[(p["target"] == "latency_ms") & (p["model"] == "yolo12n") & (p["imgsz"] == 640)]["pred"].iloc[0]
    e_ = p[(p["target"] == "energy_mJ") & (p["model"] == "yolo12n") & (p["imgsz"] == 640)]["pred"].iloc[0]
    row = out["serve"].query("model == 'yolo12n' and imgsz == 640").iloc[0]
    dur = 10 * (t / 1000 + 0.2)
    assert row["duration_s"] == pytest.approx(dur) and row["E_J"] == pytest.approx(10 * e_ / 1000 + (dur - 10 * t / 1000) * 5.0)
    assert len(s) == len(CONFIGS) and (out["path"].with_name(lb.SERVE_CSV)).is_file()


# ------------------------------------------------------------------ final fits

def test_secondary_fits_skip_the_candidate_that_is_already_the_auto_final(env):
    recs = ri.read_records(env[0].runs_log)
    sec = {(r["target"], r["candidate"]): r for r in recs if r.get("command") == "final-mode-a-secondary"}
    assert set(sec) == {("energy_mJ", "B3LogLinear"), ("energy_mJ", "M3Factored"), ("latency_ms", "B3LogLinear"),
                        ("mean_P_W", "B3LogLinear")}
    assert all(len(r["sha256"]) == 64 for r in sec.values())


def test_mode_l_fit_final_pickles_every_rung_and_reports_sha(lockbox_env):
    recs = ri.read_records(lockbox_env.paths.runs_log)
    got = {r["rung"]: r for r in recs if r.get("command") == "final-mode-l"}
    assert list(got) == list(mlm.default_rungs())
    for name, r in got.items():
        assert r["path"].endswith(f"mode_l_{name}.pkl") and len(r["sha256"]) == 64 and r["n_cells"] == 6
        assert r["base_models"] == ["yolov8m", "yolov8n", "yolov8s"]


def test_mode_l_fit_final_ignores_lockbox_cells_and_picks_the_grid_by_inner_lomo(lockbox_env, monkeypatch):
    from test_mode_l import make_windows
    monkeypatch.setenv("ANALYTICS_DATA", str(lockbox_env.data))
    w, cat = make_windows()
    cat["source"] = "v2"
    lock = w[w["cell_id"] == "yolov8n_r0"].assign(cell_id="lock_r0")
    cat = pd.concat([cat, cat[cat["cell_id"] == "yolov8n_r0"].assign(cell_id="lock_r0", source="lockbox")])
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    seen = []
    orig = mode_l._fit
    monkeypatch.setattr(mode_l, "_fit", lambda rung, seed, params, train: (seen.append(set(train["cell_id"])), orig(rung, seed, params, train))[1])
    _, info = mode_l.fit_final(pd.concat([w, lock]), cat, qc, "L2", 1)
    assert info["n_cells"] == 6 and all("lock_r0" not in s for s in seen)
    assert info["params"] in mlm.default_rungs()["L2"].grid
    with pytest.raises(ValueError, match="unknown rung"):
        mode_l.fit_final(w, cat, qc, "L9", 1)


# ------------------------------------------------------------------ frozen-models manifest

def test_manifest_lists_every_frozen_model_and_the_predictions_sha(env):
    paths, _ = env
    out = predict(env)
    man = json.loads(out["manifest"].read_text())
    assert out["manifest"] == paths.out_dir / lb.MANIFEST
    assert man["predictions_sha256"] == lb.runlog.file_hash(out["path"]) == out["rec"]["predictions_sha256"]
    assert set(man["mode_l"]) == set(mlm.default_rungs()) and len(man["mode_a"]) == 7
    recs = ri.read_records(paths.runs_log)
    for name, v in man["mode_l"].items():
        rec = ri.latest_record(recs, command="final-mode-l", part="fit_final", rung=name, seed=MASTER)
        assert v == {"sha256": rec["sha256"], "run_id": rec["run_id"]}
    assert out["rec"]["mode_l_frozen"] == man["mode_l"] and out["rec"]["manifest_sha256"] == lb.runlog.file_hash(out["manifest"])
    assert man["mode_a"]["M1Ridge/energy_mJ"]["role"] == "primary" and man["mode_a"]["M3Factored/energy_mJ"]["role"] == "secondary"


def test_predict_refuses_when_a_mode_l_rung_is_not_frozen_and_writes_nothing(env):
    paths, _ = env
    recs = [x for x in paths.runs_log.read_text().splitlines() if '"rung": "L4-trees"' not in x]
    paths.runs_log.write_text("\n".join(recs) + "\n")
    with pytest.raises(lb.LockboxError, match="L4-trees.*final-mode-l"):
        predict(env)
    assert not list(paths.out_dir.glob("lockbox_*")) if paths.out_dir.exists() else True


def test_predict_refuses_a_changed_mode_l_pickle(env, tmp_path):
    paths, _ = env
    victim = ri.latest_record(ri.read_records(paths.runs_log), command="final-mode-l", part="fit_final", rung="L0")
    with pytest.raises(lb.LockboxError, match="changed after it was frozen"):
        lb.check_sha({**victim, "sha256": "d" * 64})
    assert lb.check_sha(victim) == victim["sha256"]
