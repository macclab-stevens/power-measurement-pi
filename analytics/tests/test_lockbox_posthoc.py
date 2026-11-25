"""Post hoc (R2) lockbox scoring on tiny synthetic inputs: isolation (no lockbox row in any fit), regimes, intervals, scoring,
Mode L finals under new pickle names, and the written outputs."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli
import lockbox as lb
import lockbox_posthoc as ph
import lockbox_score as ls
import mode_a
import mode_l
import mode_l_models as mlm
import report_inputs as ri
import settings
from synth_lockbox import MASTER, TRAIN
from test_lockbox_score import inputs
from test_mode_a import SMALL_GRIDS, make_dataset
from test_mode_l import make_windows

NAN = float("nan")


# ------------------------------------------------------------------ Mode A

def test_fit_candidates_matches_fit_final_has_intervals_and_writes_no_pickle(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t, desc = make_dataset(TRAIN[:5])
    for regime in mode_a.REGIMES:
        res = mode_a.fit_candidates(t, "energy_mJ", MASTER, desc, regime, SMALL_GRIDS)
        assert set(res) == set(mode_a.CANDIDATES_BY_TARGET["energy_mJ"]) | {mode_a.AUTO}
        assert res[mode_a.AUTO]["chosen"] in mode_a.AUTO_AMONG["energy_mJ"]
        assert res[mode_a.AUTO]["score"] == min(res[c]["score"] for c in mode_a.AUTO_AMONG["energy_mJ"])
        assert all(f["q5"] < f["q95"] and f["n_resid"] >= 2 for f in res.values())
        assert np.isfinite(np.asarray(res["M1Ridge"]["model"].predict(t.head(3)))).all()
    assert not (settings.derived_dir() / "models").exists()                       # no pickle
    _, info = mode_a.fit_final(t, "M1Ridge", "energy_mJ", MASTER, desc, SMALL_GRIDS, regime="seen-model")
    assert mode_a.fit_candidates(t, "energy_mJ", MASTER, desc, "seen-model", SMALL_GRIDS)["M1Ridge"]["params"] == info["params"]


def test_fit_candidates_rejects_a_bad_regime_or_target():
    t, desc = make_dataset(TRAIN[:4])
    with pytest.raises(ValueError):
        mode_a.fit_candidates(t, "energy_mJ", 1, desc, "nope", SMALL_GRIDS)
    with pytest.raises(ValueError):
        mode_a.fit_candidates(t, "dyn_energy", 1, desc, "new-model", SMALL_GRIDS)


def lock_feats(lockbox_env, configs=("yolo12n:640", "yolov8n:480", "resnet18:224")):
    feats, _ = lb.feature_rows([lb.parse_config(c) for c in configs], lockbox_env.static_dir)
    return feats


def test_regimes_follow_the_base_model_and_every_fit_sees_training_keys_only(lockbox_env):
    t, desc = lockbox_env.t, lockbox_env.desc
    feats = lock_feats(lockbox_env)
    calls = []

    def fake(tt, target, seed, d, regime, grids):
        calls.append((target, regime, frozenset(tt["config_key"])))
        model = type("M", (), {"predict": lambda self, df: np.full(len(df), 100.0)})()
        return {c: {"model": model, "params": {}, "score": 0.1, "q5": -0.2, "q95": 0.3, "n_resid": 9, "chosen": c}
                for c in (*mode_a.CANDIDATES_BY_TARGET[target], mode_a.AUTO)}
    p = ph.mode_a_predictions(t, feats, lb.descriptor(desc, feats), MASTER, targets=["energy_mJ", "latency_ms"], fit=fake)
    assert {(c[0], c[1]) for c in calls} == {(tg, r) for tg in ("energy_mJ", "latency_ms") for r in mode_a.REGIMES}
    assert all(c[2] == frozenset(t["config_key"]) and not c[2] & set(feats["config_key"]) for c in calls)
    reg = p.drop_duplicates("config_key").set_index("model")["regime"].to_dict()
    assert reg == {"yolo12n": "new-model", "yolov8n": "seen-model", "resnet18": "seen-model"}     # yolov8n is a training base model
    a = p[(p["candidate"] == "AUTO") & (p["target"] == "energy_mJ")]
    assert len(a) == 3 and np.allclose(a["lo90"], 100.0 * np.exp(-0.2)) and np.allclose(a["hi90"], 100.0 * np.exp(0.3))
    assert set(p["candidate"]) == set(mode_a.CANDIDATES_BY_TARGET["energy_mJ"]) | {mode_a.AUTO}


def test_a_lockbox_row_in_the_training_input_is_refused_before_any_fit(lockbox_env):
    t, desc = lockbox_env.t, lockbox_env.desc
    feats = lock_feats(lockbox_env)

    def boom(*a, **k):
        raise AssertionError("a fit was reached")
    with pytest.raises(lb.LockboxError, match="lockbox configuration"):
        ph.mode_a_predictions(pd.concat([t, feats.iloc[:1]], ignore_index=True), feats, desc, MASTER, fit=boom)
    with pytest.raises(lb.LockboxError, match="source lockbox"):
        ph.check_training_only(t.assign(source="lockbox"))
    with pytest.raises(lb.LockboxError, match="cell_ids"):
        ph.check_training_only(t.assign(cell_ids="a_r0;lockbox_yolo12n_640_t4_2400_r00"))
    with pytest.raises(lb.LockboxError, match="cell_ids"):
        ph.check_training_only(t.assign(cell_ids="a_r0;b_r0"), lockbox_cells=["b_r0"])
    with pytest.raises(lb.LockboxError, match="window of a lockbox cell"):
        ph.check_training_only(t, w=pd.DataFrame({"cell_id": ["tr_r0", "lockbox_x_r0"]}))
    ph.check_training_only(t, feats, pd.DataFrame({"cell_id": ["tr_r0"]}), ["other"])        # clean input passes


def test_score_and_summary_numbers():
    pred = pd.DataFrame({"config_key": ["a", "b", "c"], "model": ["m1", "m1", "m2"], "imgsz": 640, "regime": ["seen-model"] * 2 + ["new-model"],
                         "group": ["seen model", "seen model", "new model"], "target": "energy_mJ", "candidate": "AUTO", "chosen": "M1Ridge",
                         "pred": [110.0, 80.0, 300.0], "lo90": [50.0, 50.0, 100.0], "hi90": [200.0, 70.0, 600.0], "score": 0.1, "params": "{}"})
    truth = pd.DataFrame({"config_key": ["a", "b", "c"], "energy_mJ": [100.0, 100.0, 200.0]})
    s = ph.score_mode_a(pred, truth)
    assert s["ape"].tolist() == pytest.approx([0.1, 0.2, 0.5]) and s["inside"].tolist() == [True, False, True]
    sm = ph.summarise(s).set_index("group")
    assert sm.loc["all", "mdape"] == pytest.approx(0.2) and sm.loc["all", "maxape"] == pytest.approx(0.5)
    assert sm.loc["all", "mape"] == pytest.approx(0.8 / 3) and sm.loc["all", "inside_90"] == pytest.approx(2 / 3)
    assert sm.loc["all", "width"] == pytest.approx(np.median([4.0, 1.4, 6.0])) and sm.loc["new model", "n"] == 1
    assert sm.loc["seen model", "mdape"] == pytest.approx(0.15) and sm.loc["seen model", "bias_pct"] == pytest.approx((0.1 - 0.2) / 2)


def test_summarise_skips_configurations_without_truth():
    pred = pd.DataFrame({"config_key": ["a", "z"], "model": "m", "imgsz": 1, "regime": "seen-model", "group": "seen model", "target": "energy_mJ",
                         "candidate": "AUTO", "chosen": "M1Ridge", "pred": 1.0, "lo90": 0.5, "hi90": 2.0, "score": 0.0, "params": "{}"})
    s = ph.score_mode_a(pred, pd.DataFrame({"config_key": ["a"], "energy_mJ": [1.0]}))
    assert np.isnan(s["truth"].iloc[1]) and ph.summarise(s)["n"].iloc[0] == 1


def test_truth_check_against_the_frozen_table():
    a = pd.DataFrame({"config_key": ["a", "b"], "energy_mJ": [1.0, 2.0], "latency_ms": [3.0, 4.0], "mean_P_W": [5.0, 6.0]})
    assert ph.check_truth_matches_frozen(a, None) is None
    assert ph.check_truth_matches_frozen(a, a.iloc[::-1]) is True
    with pytest.raises(lb.LockboxError, match="differs"):
        ph.check_truth_matches_frozen(a, a.assign(energy_mJ=[1.0, 2.1]))


def test_calculator_comparison_puts_r1_beside_r2():
    cat_l, der, qc_l, pred, idle_t, train = __import__("test_lockbox_score")._gap_frames()
    r2 = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle_t, train_cells=train)
    r1 = r2.assign(ape_known=0.01, ape_oh0=0.02, ape_est=0.03)
    c = ph.compare_calc(r2, r1)
    assert list(c.columns) == ["variant", "n", "R2 median", "R2 mean", "R2 max", "R1 median", "R1 mean", "R1 max"]
    assert c["R1 median"].tolist() == pytest.approx([0.01, 0.02, 0.03]) and list(ph.compare_calc(r2, None).columns)[-1] == "R2 max"
    auto = ph.auto_primary(pd.DataFrame({"model": ["m"] * 3, "imgsz": 1, "candidate": ["AUTO"] * 2 + ["M1Ridge"], "target": ["energy_mJ", "mean_P_W", "energy_mJ"],
                                         "pred": 1.0, "lo90": 0.5, "hi90": 2.0}), {"m": 2.5})
    assert len(auto) == 1 and auto["role"].iloc[0] == "primary" and auto["novelty"].iloc[0] == 2.5 and not auto["flagged"].iloc[0]


# ------------------------------------------------------------------ Mode L

def train_windows(prefix="tr_"):
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"), cells_per_model=2, seed=1)
    w["cell_id"], cat["cell_id"] = prefix + w["cell_id"], prefix + cat["cell_id"]
    cat = cat.assign(source="v2", path="", task="image", has_counters=True, serve_fps=NAN, soak_minutes=NAN, concurrent_mode=None,
                     input_source="repeat", quant="fp32", n_quantized_modules=0)
    return w, cat


RUNGS = mlm.first_point({k: v for k, v in mlm.default_rungs().items() if k in ("L0", "L3-trees", "L5-cmos")})


def test_mode_l_finals_get_new_names_and_never_overwrite_a_frozen_pickle(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    w, cat = train_windows()
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    frozen = settings.derived_dir() / "models"
    frozen.mkdir(parents=True)
    (frozen / "mode_l_L0.pkl").write_bytes(b"frozen R1")
    out = ph.fit_mode_l(w, cat, qc, MASTER, RUNGS)
    assert set(out) == set(RUNGS)
    assert sorted(p.name for p in frozen.glob("*.pkl")) == ["mode_l_L0.pkl", "mode_l_L0_r2.pkl", "mode_l_L3-trees_r2.pkl", "mode_l_L5-cmos_r2.pkl"]
    assert (frozen / "mode_l_L0.pkl").read_bytes() == b"frozen R1"
    assert all(Path(i["path"]).name.endswith("_r2.pkl") and i["n_cells"] == 6 for _, i in out.values())
    _, info = mode_l.fit_final(w, cat, qc, "L0", MASTER, RUNGS)                              # the default name is still the R1 one
    assert Path(info["path"]).name == "mode_l_L0.pkl"


def test_mode_l_fit_refuses_lockbox_windows_and_drops_lockbox_catalogue_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    w, cat = train_windows()
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    lw, lcat = train_windows("lockbox_")
    seen = []
    real = mode_l.fit_final
    monkeypatch.setattr(mode_l, "fit_final", lambda ww, cc, *a, **k: (seen.append(set(ww["cell_id"])), real(ww, cc, *a, **k))[1])
    with pytest.raises(lb.LockboxError, match="window of a lockbox cell"):
        ph.fit_mode_l(pd.concat([w, lw]), pd.concat([cat, lcat.assign(source="lockbox")]), qc, MASTER, RUNGS)
    assert not seen                                                                          # refused before any fit
    ph.fit_mode_l(w, pd.concat([cat, lcat.assign(source="lockbox")]), qc, MASTER, {"L0": RUNGS["L0"]})
    assert seen and not any(c.startswith("lockbox") for c in seen[0])


def test_mode_l_family_summary_pools_forward_windows_and_r1_sits_beside_it():
    w, cat = make_windows(models=("yolo12n", "yolov8n", "resnet18", "mbv3"), cells_per_model=3, seed=5)
    models = {n: r.make(0).fit(w, r.grid[0], mlm.cell_weights(w)) for n, r in RUNGS.items()}
    cells = ls.mode_l_cells(w, cat.assign(imgsz=640), models)
    s = ph.summarise_mode_l(cells)
    assert set(s["family"]) == {"YOLO", "non-YOLO"} and set(s["rung"]) == set(RUNGS)
    y = s[(s["family"] == "YOLO") & (s["rung"] == "L0")].iloc[0]
    g = cells[(cells["rung"] == "L0") & cells["model"].isin(["yolo12n", "yolov8n"])]
    assert y["n_cells"] == 6 and y["whole"] == pytest.approx(g["ape"].median()) and y["fwd_mae_W"] == pytest.approx(g["sae_fwd"].sum() / g["n_fwd"].sum())
    assert y["fwd_r2"] == pytest.approx(mode_l.pooled_window_metrics(g)["r2_fwd"])
    r1 = ph.r1_mode_l(cells)
    assert list(r1.columns) == ["family", "rung", "R1 whole", "R1 inference", "R1 1 s block"] and len(r1) == len(s)


# ------------------------------------------------------------------ the run

@pytest.fixture
def world(lockbox_env, tmp_path, monkeypatch):
    """Synthetic catalogue (training cells with serve phases, 15 lockbox cells), training and lockbox windows, runlog and folders."""
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "data"))
    cat, der, qc_t, qc_l, w_l = inputs()
    w_t, cat_w = train_windows()
    cat = pd.concat([cat, cat_w], ignore_index=True)
    der_w = pd.DataFrame({"cell_id": cat_w["cell_id"], "mode": "forward", "batch": 1, "energy_mJ_per_image": 1000.0, "latency_ms_per_image": 100.0,
                          "mean_P_W": 7.0, "baseline_W": 5.0, "serve_n_requests": 10, "serve_dur_ms": 3100.0, "serve_E_mJ": 20000.0})
    der = pd.concat([der, der_w], ignore_index=True)
    qc_t = pd.concat([qc_t, pd.DataFrame({"cell_id": cat_w["cell_id"], "primary": True})], ignore_index=True)
    paths = dataclasses.replace(lockbox_env.paths, runs_log=tmp_path / "runs.jsonl", out_dir=tmp_path / "out")
    paths.out_dir.mkdir()
    kw = dict(cat=cat, der=der, qc_train=qc_t, qc_l=qc_l, t=lockbox_env.t, desc_train=lockbox_env.desc, w_train=w_t,
              static_dir=lockbox_env.static_dir, w_lockbox=w_l, grids=SMALL_GRIDS, rungs=RUNGS)
    return paths, kw, lockbox_env


def test_run_posthoc_end_to_end_on_training_data_only_with_outputs(world, monkeypatch):
    paths, kw, env = world
    fits = []
    real_fit = mode_a.fit_candidates
    monkeypatch.setattr(mode_a, "fit_candidates", lambda tt, *a, **k: (fits.append(frozenset(tt["config_key"])), real_fit(tt, *a, **k))[1])
    lfits = []
    real_l = mode_l.fit_final
    monkeypatch.setattr(mode_l, "fit_final", lambda ww, cc, *a, **k: (lfits.append((set(ww["cell_id"]), set(cc["cell_id"]))), real_l(ww, cc, *a, **k))[1])
    out = ph.run_posthoc(paths, MASTER, **kw)
    lock_cells = {c for c in kw["cat"]["cell_id"] if kw["cat"].set_index("cell_id").loc[c, "source"] == "lockbox"}
    assert fits and all(f == frozenset(env.t["config_key"]) for f in fits)
    assert lfits and all(not (w & lock_cells) and not (c & lock_cells) for w, c in lfits)
    d = paths.out_dir / "r2"
    assert sorted(p.name for p in d.iterdir()) == sorted(["lockbox_posthoc.md", "lockbox_posthoc_mode_a.csv", "lockbox_posthoc_mode_a_summary.csv",
                                                          "lockbox_posthoc_mode_l.csv", "lockbox_posthoc_mode_l_summary.csv", "lockbox_posthoc_calc.csv"])
    assert not (paths.out_dir / "lockbox.md").exists() and not (settings.derived_dir() / "lockbox").exists()
    text = (d / "lockbox_posthoc.md").read_text()
    assert ph.LABEL in text and "Mode L by family" in text and "Deployment calculator" in text
    assert text.index("Mode A: AUTO and M4Physics") < text.index("Per configuration")                      # summaries first
    a = pd.read_csv(d / "lockbox_posthoc_mode_a.csv")
    assert list(a.columns) == ph.MODE_A_COLS and a["config_key"].nunique() == 4 and {"AUTO", "M4Physics"} <= set(a["candidate"])
    regime = a.drop_duplicates("model").set_index("model")["regime"].to_dict()
    assert regime == {"yolo12n": "new-model", "yolov8n": "seen-model", "resnet18": "seen-model", "mbv3": "seen-model"}
    assert np.isfinite(a["ape"]).all()
    assert len(pd.read_csv(d / "lockbox_posthoc_calc.csv")) == 11                                              # 12 lockbox cells, one not QC-primary
    l = pd.read_csv(d / "lockbox_posthoc_mode_l.csv")
    assert set(l["rung"]) == set(RUNGS) and set(l["family"]) == {"YOLO", "non-YOLO"}
    assert all(Path(i["path"]).name.endswith("_r2.pkl") for _, i in out["fits"].values())
    rec = json.loads(paths.runs_log.read_text().splitlines()[-1])
    assert rec["command"] == "lockbox-posthoc" and rec["label"] == ph.LABEL and set(rec["mode_l_models"]) == set(RUNGS)
    assert rec["truth_matches_frozen"] is None and set(rec["calculator_median_ape"]) == {v for _, v in ls.POSTHOC_VARIANTS}


def test_run_posthoc_puts_the_frozen_r1_numbers_beside_and_checks_the_truth(world):
    paths, kw, env = world
    cat_l = kw["cat"][kw["cat"]["source"] == "lockbox"]
    truth = ls.truth_table(cat_l, kw["der"], kw["qc_l"])
    fdir = settings.derived_dir() / "lockbox"
    fdir.mkdir(parents=True)
    frozen = truth.rename(columns={"base_model": "model"}).melt(id_vars=["model", "imgsz", "config_key"], value_vars=list(ls.TARGET_COL), var_name="target",
                                                               value_name="truth")
    frozen = frozen.assign(candidate="M1Ridge", role="primary", pred=frozen["truth"] * 1.1, lo90=frozen["truth"] * 0.5, hi90=frozen["truth"] * 2,
                           ape=0.1, inside=True, new_model=frozen["model"] == "yolo12n")
    truth.to_parquet(fdir / "frozen_truth.parquet", index=False)
    frozen.to_parquet(fdir / "frozen_mode_a_scores.parquet", index=False)
    paths.runs_log.write_text(json.dumps({"run_id": "frozen", "command": "lockbox-score"}) + "\n")
    out = ph.run_posthoc(paths, MASTER, **kw)
    assert out["r1_mode_a_summary"] is not None and "M1Ridge (primary)" in set(out["r1_mode_a_summary"]["candidate"])
    text = (paths.out_dir / "r2" / "lockbox_posthoc.md").read_text()
    assert "| R1 |" in text and "R1 (frozen)" not in text and "truth equals the frozen truth table" in text and ": yes" in text
    # a frozen truth table that differs is refused
    pd.read_parquet(fdir / "frozen_truth.parquet").assign(energy_mJ=lambda d: d["energy_mJ"] * 1.5).to_parquet(fdir / "frozen_truth.parquet", index=False)
    with pytest.raises(lb.LockboxError, match="differs"):
        ph.run_posthoc(paths, MASTER, **kw)


def test_run_posthoc_refuses_a_polluted_training_table(world):
    paths, kw, env = world
    bad = pd.concat([kw["t"], lock_feats(env).iloc[:1]], ignore_index=True)
    with pytest.raises(lb.LockboxError, match="lockbox configuration"):
        ph.run_posthoc(paths, MASTER, **{**kw, "t": bad})
    wl = kw["w_lockbox"]
    with pytest.raises(lb.LockboxError, match="window of a lockbox cell"):
        ph.run_posthoc(paths, MASTER, **{**kw, "w_train": pd.concat([kw["w_train"], wl.iloc[:3]], ignore_index=True)})
    assert not (paths.out_dir / "r2").exists()


def test_the_command_is_registered_and_the_frozen_outputs_are_not_in_its_paths():
    a = cli.build_parser().parse_args(["lockbox-posthoc", "--n-jobs", "3", "--no-probe"])
    assert a.command == "lockbox-posthoc" and a.n_jobs == 3 and a.no_probe
    assert "post hoc" in cli.COMMANDS["lockbox-posthoc"][0]
