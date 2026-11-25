"""Mode L ladder: window models, leave-one-model-out folds (with the no-test-data-in-a-fit spy),
the PMIC leave-one-workload-out check and probe mode, all on synthetic windows."""
import json

import numpy as np
import pandas as pd
import pytest

import mode_l
import mode_l_models as mm

NAN = float("nan")
N_WIN = 220


def make_windows(models=("yolov8n", "yolov8s", "yolov8m"), cells_per_model=2, seed=0, pmic=True, slope=5.0, icpt=4.0):
    """Synthetic 100 ms windows: P = icpt + slope * busy. Cell ids <model>_r<k>."""
    rng = np.random.default_rng(seed)
    rows, cat = [], []
    for mi, m in enumerate(models):
        for r in range(cells_per_model):
            cid = f"{m}_r{r}"
            busy = np.clip(rng.uniform(0, 1, N_WIN) + 0.05 * mi, 0, 1)
            P = icpt + slope * busy
            ph = np.where(np.arange(N_WIN) < 20, "baseline_A", "forward")
            ph[20:30] = "warmup"
            d = pd.DataFrame({
                "cell_id": cid, "w_idx": np.arange(N_WIN), "dur_s": 0.1, "P_W": P, "E_mJ": P * 100.0,
                "phase": ph, "counted": 1, "busy": busy, "cycles_rate": busy * 2.4e9,
                "instr_rate": busy * 3e9, "ipc": 1.2 + 0.0 * busy, "cache_miss_rate": 0.01,
                "branch_mpki": 2.0, "stall_frac": 0.3, "llc_mpki": 0.5, "ctx_sw_rate": 0.0,
                "pf_rate": 10.0, "llc_miss_rate": 1e6 * busy, "f_eff_mhz": 2400.0, "vcore_eff": 0.9294,
                "v2_cycles_rate": 0.9294 ** 2 * busy * 2.4e9, "v2_instr_rate": 0.9294 ** 2 * busy * 3e9,
                "busy_l1": np.r_[0, busy[:-1]], "busy_l2": np.r_[0, 0, busy[:-2]],
                "busy_l3": np.r_[0, 0, 0, busy[:-3]], "cycles_rate_l1": 1.0, "cycles_rate_l2": 1.0,
                "cycles_rate_l3": 1.0, "ipc_l1": 1.2, "ipc_l2": 1.2, "ipc_l3": 1.2, "freq_mhz": 2400.0,
                "temp_C": 55.0 + 5 * busy, "fan_state": 4.0, "pmic_W": (P - icpt) / 2 + 1.0 if pmic else NAN,
                "pmic_age_s": 2.0, "cpu_key": "2400", "fan_mode": "full", "governor": "pinned"})
            rows.append(d)
            cat.append({"cell_id": cid, "base_model": m, "model_key": m, "family": "A", "variant": m,
                        "precision": "fp32", "imgsz": 640, "batch": 1, "threads": 4, "cpu_mhz": 2400,
                        "rig": "nov2026", "governor": "pinned", "fan_mode": "full",
                        "mode_a_eligible": True})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cat)


# ------------------------------------------------------------------ window models

def test_l2_recovers_linear_busy_power():
    w, _ = make_windows()
    m = mm.L2Busy().fit(w, {"alpha": 1e-2}, np.ones(len(w)))
    assert np.abs(m.predict(w) - w["P_W"]).max() < 0.02


def test_l0_lookup_and_fallbacks():
    w, _ = make_windows(cells_per_model=1)
    w["cpu_key"] = np.where(w["cell_id"] == "yolov8n_r0", "1500", "2400")
    w["busy"] = np.where(w["cell_id"] == "yolov8n_r0", 0.9, 0.1)
    w["P_W"] = np.where(w["cell_id"] == "yolov8n_r0", 10.0, 6.0)
    m = mm.L0Lookup().fit(w, {}, np.ones(len(w)))
    q = w.iloc[:3].copy()
    q["cpu_key"], q["fan_mode"], q["busy"] = ["1500", "2400", "2400"], ["full", "full", "off"], 0.9
    # (1500, full, busy>0.5) known -> 10; (2400, full, busy>0.5) unseen at that level -> cpu_key mean 6;
    # (2400, off, busy>0.5) unseen -> cpu_key 2400 mean 6
    assert m.predict(q).tolist() == pytest.approx([10.0, 6.0, 6.0])
    q["cpu_key"] = ["9999", "9999", "9999"]       # unseen clock -> global weighted mean
    assert m.predict(q) == pytest.approx(np.full(3, w["P_W"].mean()))


def test_l0_weights_cells_equally():
    a = pd.DataFrame({"cell_id": "a", "cpu_key": "x", "fan_mode": "f", "busy": 0.9, "P_W": [10.0] * 90})
    b = pd.DataFrame({"cell_id": "b", "cpu_key": "x", "fan_mode": "f", "busy": 0.9, "P_W": [4.0] * 10})
    w = pd.concat([a, b], ignore_index=True)
    sw = mm.cell_weights(w)
    assert sw.sum() == pytest.approx(2.0)
    assert mm.L0Lookup().fit(w, {}, sw).predict(w)[0] == pytest.approx(7.0)


def test_l1_ols_and_l0_fallback_with_share():
    w, _ = make_windows()
    w["pmic_W"] = (w["P_W"] - 1.0) / 2.0          # P = 1 + 2 pmic
    m = mm.L1Pmic().fit(w, {}, mm.cell_weights(w))
    assert m.predict(w) == pytest.approx(w["P_W"].to_numpy())
    t = w.iloc[:10].copy()
    t.loc[t.index[:4], "pmic_W"] = NAN
    p = m.predict(t)
    assert p[4:] == pytest.approx(t["P_W"].to_numpy()[4:])
    assert np.isfinite(p).all()                   # NaN windows come from the L0 lookup
    assert m.fallback_frac(t) == pytest.approx(0.4)


def test_l3_l4_ridge_and_trees_fit_predict_with_missing_values():
    w, _ = make_windows(pmic=True)
    w.loc[w.index[::7], ["ipc", "llc_mpki"]] = NAN
    w.loc[w.index[::5], "pmic_W"] = NAN
    sw = mm.cell_weights(w)
    for cls in (mm.L3Counters, mm.L4CountersPmic):
        r = cls("ridge", seed=1).fit(w, {"alpha": 0.01}, sw)
        assert np.isfinite(r.predict(w)).all()
        assert np.abs(r.predict(w) - w["P_W"]).mean() < 0.2
        grid = {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50, "min_samples_leaf": 20}
        t = cls("trees", seed=1).fit(w, grid, sw)
        assert np.isfinite(t.predict(w)).all()
        t2 = cls("trees", seed=1).fit(w, grid, sw)
        assert t.predict(w) == pytest.approx(t2.predict(w))


def test_l3_features_have_no_pmic_and_l4_adds_it():
    assert "pmic_W" not in mm.L3Counters("ridge").feature_names()
    f4 = mm.L4CountersPmic("ridge").feature_names()
    assert "pmic_W" in f4 and "pmic_age_s" in f4 and "fan_full" in f4 and "gov_ondemand" in f4


def test_grids_match_the_protocol():
    g = mm.default_rungs()
    assert list(g) == ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees", "L5-cmos",
                       "L5-cmos-trees", "L3k-trees", "L5k-trees"]
    assert g["L5-cmos"].grid == g["L3-ridge"].grid and g["L5-cmos-trees"].grid == g["L3-trees"].grid
    assert [p["alpha"] for p in g["L2"].grid] == pytest.approx(list(np.logspace(-2, 4, 7)))
    assert len(g["L3-ridge"].grid) == 7 and len(g["L0"].grid) == 1 and len(g["L1"].grid) == 1
    tg = g["L4-trees"].grid
    assert {(p["max_depth"], p["min_samples_leaf"]) for p in tg} == {(3, 50), (3, 200), (5, 50), (5, 200)}
    assert {p["learning_rate"] for p in tg} == {0.1} and {p["max_iter"] for p in tg} == {200}


def test_no_rung_uses_ctx_sw_rate_and_l5_adds_the_five_cmos_features():
    for name, rung in mm.default_rungs().items():
        m = rung.make(0)
        if hasattr(m, "feature_names") and name not in ("L3k-trees", "L5k-trees"):   # R4 kernel rungs, v4 cells only
            assert "ctx_sw_rate" not in m.feature_names(), name
    assert mm.L3Kernel("trees").feature_names() == mm.L3Counters("trees").feature_names() + mm.KERNEL
    assert mm.default_rungs()["L3k-trees"].exploratory and mm.default_rungs()["L5k-trees"].exploratory
    assert "ctx_sw_rate" not in mm.COUNTERS
    l3, l5 = mm.L3Counters("ridge").feature_names(), mm.L5Cmos("ridge").feature_names()
    five = ["llc_miss_rate", "f_eff_mhz", "vcore_eff", "v2_cycles_rate", "v2_instr_rate"]
    assert l5 == l3 + five and not set(five) & set(l3)
    assert mm.L5Cmos("trees").feature_names() == l5 and "pmic_W" not in l5 and "pmic_missing" not in l5
    assert "pmic_missing" in mm.L4CountersPmic("ridge").feature_names()


def test_l5_fits_and_uses_the_new_columns():
    w, _ = make_windows()
    sw = np.ones(len(w))
    for variant, params in (("ridge", {"alpha": 1.0}), ("trees", mm.TREE_GRID[0])):
        m = mm.L5Cmos(variant, seed=0).fit(w, params, sw)
        assert np.isfinite(m.predict(w)).all()
        w2 = w.drop(columns=["v2_cycles_rate"])
        with pytest.raises(KeyError):
            m.predict(w2)


def test_unpickled_r1_model_keeps_its_ctx_sw_rate_feature():
    w, _ = make_windows()
    m = mm.L3Counters("ridge").fit(w, {"alpha": 1.0}, np.ones(len(w)))
    del m.counters                                  # what a pickle from before R2 looks like
    assert "ctx_sw_rate" in m.feature_names() and m._X(w).shape[1] == len(mm.COUNTERS) + 1 + len(mm.ONEHOT)
    w = w.drop(columns=["ctx_sw_rate"])
    with pytest.raises(KeyError):
        m.predict(w)


def test_trees_seed_is_the_run_seed():
    r = mm.default_rungs()["L3-trees"].make(12345)
    assert r.seed == 12345


# ------------------------------------------------------------------ spy: no test data in any fit

class Spy:
    """Window model that records what it was fitted on and what it predicts for."""
    log = []

    def fit(self, train_w, params, sample_weight):
        cells = set(train_w["cell_id"])
        Spy.log.append(("fit", cells, float(np.sum(sample_weight)), params))
        assert len(sample_weight) == len(train_w)
        self.cells = cells
        self.mean = float(np.average(train_w["P_W"], weights=sample_weight))
        return self

    def predict(self, w):
        Spy.log.append(("predict", set(w["cell_id"]), self.cells))
        return np.full(len(w), self.mean)


def test_spy_no_test_cell_ever_in_a_fit_and_weights_sum_to_cells():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m", "yolo11n"))
    Spy.log = []
    rungs = {"spy": mm.Rung("spy", lambda seed: Spy(), [{"k": 1}, {"k": 2}])}
    res = mode_l.run_lomo(w, cat, seed=1, rungs=rungs)
    assert set(res["cell_id"]) == set(w["cell_id"])
    cells_of = dict(zip(cat["cell_id"], cat["base_model"]))
    fits = [e for e in Spy.log if e[0] == "fit"]
    preds = [e for e in Spy.log if e[0] == "predict"]
    assert len(fits) > 0 and len(preds) > 0
    for _, cells, wsum, _p in fits:
        assert wsum == pytest.approx(len(cells))                 # weights sum to the number of cells
    for _, test_cells, fit_cells in preds:
        assert not (test_cells & fit_cells), "a predicted cell was in the fit that predicts it"
        # whole models: a model is never split between train and the cells it is scored on
        assert {cells_of[c] for c in test_cells}.isdisjoint({cells_of[c] for c in fit_cells})
    # run each outer fold alone: its held-out model's cells appear in no fit at all
    for held in ("yolov8n", "yolov8s", "yolov8m", "yolo11n"):
        held_cells = {c for c, m in cells_of.items() if m == held}
        Spy.log = []
        mode_l.run_lomo(w, cat, seed=1, rungs=rungs, folds=[held])
        assert Spy.log and all(not (e[1] & held_cells) for e in Spy.log if e[0] == "fit")
        assert any(e[0] == "predict" and e[1] == held_cells for e in Spy.log)
    Spy.log = []
    mode_l.run_lomo(w, cat, seed=1, rungs=rungs)
    fits = [e for e in Spy.log if e[0] == "fit"]
    # the inner folds ran: every fit set is a union of whole models over 3 or 2 models
    sizes = sorted({len(c) // 2 for _, c, _, _ in fits})
    assert sizes == [2, 3]


def test_spy_inner_selection_uses_only_training_models():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    Spy.log = []
    rungs = {"spy": mm.Rung("spy", lambda seed: Spy(), [{"k": 1}])}
    mode_l.run_lomo(w, cat, seed=1, rungs=rungs)
    # a one-point grid has no inner loop: exactly one fit per outer fold
    assert len([e for e in Spy.log if e[0] == "fit"]) == 3


# ------------------------------------------------------------------ run_lomo on synthetic data

SMALL = None


def small_rungs():
    """Every rung but the trees (sklearn's weighted HistGradientBoosting binning costs about a second
    even on tiny data); the trees get their own test."""
    r = mm.default_rungs()
    return {k: mm.Rung(k, v.make, v.grid[:2]) for k, v in r.items() if "trees" not in k}


def test_run_lomo_columns_and_recovery():
    w, cat = make_windows()
    res = mode_l.run_lomo(w, cat, seed=3, rungs=small_rungs())
    assert len(res) == 6 * len(small_rungs())
    need = {"fold", "cell_id", "rung", "params", "E_true_J", "E_pred_J", "E_true_meas_J", "E_pred_meas_J",
            "ape", "ape_meas", "ape_1s", "ape_10s", "mae_window_W", "bias_pct", "mean_P_pred_meas_W",
            "l1_fallback_frac", "ape_phase_idle", "ape_phase_load_weights", "ape_phase_warmup",
            "ape_phase_forward", "ape_phase_serve"}
    assert need <= set(res.columns)
    l2 = res[res["rung"] == "L2"]
    assert (l2["ape"] < 0.01).all() and (l2["ape_1s"] < 0.02).all()
    # E_true is the sum of E_mJ over all windows of the cell, in J
    c = w[w["cell_id"] == "yolov8n_r0"]
    row = res[(res["cell_id"] == "yolov8n_r0") & (res["rung"] == "L2")].iloc[0]
    assert row["E_true_J"] == pytest.approx(c["E_mJ"].sum() / 1000)
    assert row["E_true_meas_J"] == pytest.approx(c.loc[c["phase"] == "forward", "E_mJ"].sum() / 1000)
    assert row["mean_P_pred_meas_W"] == pytest.approx(c.loc[c["phase"] == "forward", "P_W"].mean(), rel=1e-2)
    assert np.isnan(row["ape_phase_serve"]) and row["ape_phase_forward"] < 0.01
    assert row["ape_phase_idle"] < 0.05           # baseline_A windows count as idle
    assert res.loc[res["rung"] == "L1", "l1_fallback_frac"].eq(0).all()
    assert res.loc[res["rung"] == "L2", "l1_fallback_frac"].isna().all()


def test_block_horizons():
    # 220 windows -> 22 blocks of 10 and 2 blocks of 100 (20 windows dropped)
    p = np.full(220, 5.0)
    e = np.full(220, 5.0 * 0.1 * 1000)
    p[:10] = 10.0                                    # first 1 s block is 100 % over
    dur = np.full(220, 0.1)
    assert mode_l.block_ape(p, dur, e, 10) == pytest.approx(0.0)    # median of 22 blocks
    assert mode_l.block_ape(p, dur, e, 100) == pytest.approx(0.05)  # (1.0 s extra of 10 s)/ ... first block
    assert np.isnan(mode_l.block_ape(p[:50], dur[:50], e[:50], 100))


def test_run_lomo_trees_rungs_run_end_to_end():
    w, cat = make_windows()
    tree = {"max_depth": 3, "learning_rate": 0.1, "max_iter": 10, "min_samples_leaf": 20}
    r = mm.default_rungs()
    rungs = {k: mm.Rung(k, r[k].make, [tree]) for k in ("L3-trees", "L4-trees")}
    res = mode_l.run_lomo(w, cat, seed=7, rungs=rungs, folds=["yolov8n"])
    assert set(res["rung"]) == {"L3-trees", "L4-trees"} and len(res) == 4
    assert (res["ape"] < 0.05).all()
    assert {json.loads(p)["max_iter"] for p in res["params"]} == {10}


def test_l1_fallback_share_reported():
    w, cat = make_windows()
    w.loc[w["cell_id"] == "yolov8n_r0", "pmic_W"] = np.where(np.arange(N_WIN) < 22, NAN, w.loc[w["cell_id"] == "yolov8n_r0", "pmic_W"])
    res = mode_l.run_lomo(w, cat, seed=3, rungs={"L1": mm.default_rungs()["L1"]})
    r = res[res["cell_id"] == "yolov8n_r0"].iloc[0]
    assert r["l1_fallback_frac"] == pytest.approx(22 / N_WIN)


def test_run_lomo_identical_for_any_n_jobs():
    w, cat = make_windows()
    a = mode_l.run_lomo(w, cat, seed=3, rungs=small_rungs(), n_jobs=1)
    b = mode_l.run_lomo(w, cat, seed=3, rungs=small_rungs(), n_jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_run_lomo_qc_filter_and_fold_selection():
    w, cat = make_windows()
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": [c != "yolov8s_r1" for c in cat["cell_id"]]})
    res = mode_l.run_lomo(w, cat, seed=3, qc=qc, rungs={"L0": mm.default_rungs()["L0"]}, folds=["yolov8s"])
    assert set(res["cell_id"]) == {"yolov8s_r0"} and set(res["fold"]) == {"yolov8s"}


def test_grid_point_chosen_by_inner_median_ape():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m", "yolo11n"))

    class Const:
        def fit(self, train_w, params, sample_weight):
            self.c = params["c"]
            return self

        def predict(self, x):
            return np.full(len(x), self.c)
    # truth mean is ~6.5 W: c = 6.5 beats 3 and 12 in every inner fold
    rungs = {"c": mm.Rung("c", lambda seed: Const(), [{"c": 3.0}, {"c": 6.5}, {"c": 12.0}])}
    res = mode_l.run_lomo(w, cat, seed=0, rungs=rungs)
    assert {json.loads(p)["c"] for p in res["params"]} == {6.5}


def _select_workload():
    """Windows where the inference segment (forward) is 10 W and the long idle part 2 W: the grid point
    that is best on the whole cell (c = 3.2: overall mean) is not the best on the inference segment (c = 10)."""
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    fwd = w["w_idx"] >= 170
    w["phase"] = np.where(fwd, "forward", "baseline_A")
    w["P_W"] = np.where(fwd, 10.0, 2.0)
    w["E_mJ"] = w["P_W"] * 100.0
    w["base_model"] = w["cell_id"].str.split("_").str[0]
    n_fwd = fwd.sum() / len(w)
    return w, 10.0 * n_fwd + 2.0 * (1 - n_fwd)


class _Const:
    def fit(self, train_w, params, sample_weight):
        self.c = params["c"]
        return self

    def predict(self, x):
        return np.full(len(x), self.c)


def test_select_uses_the_inference_segment_ape_not_the_whole_cell():
    w, whole_mean = _select_workload()
    grid = [{"c": whole_mean}, {"c": 10.0}, {"c": 2.0}]
    rung = mm.Rung("c", lambda seed: _Const(), grid)
    te = w[w["base_model"] == "yolov8n"]
    assert np.nanmedian(mode_l._cell_apes(te, np.full(len(te), whole_mean))) < 0.01      # whole-cell prefers c = mean
    assert np.nanmedian(mode_l._cell_apes(te, np.full(len(te), 10.0))) > 0.2
    assert mode_l._select(rung, 0, w) == {"c": 10.0}                                      # inference segment: c = 10


def test_select_falls_back_to_whole_cell_without_forward_windows():
    w, whole_mean = _select_workload()
    w["phase"] = "idle"
    rung = mm.Rung("c", lambda seed: _Const(), [{"c": 10.0}, {"c": whole_mean}, {"c": 2.0}])
    assert mode_l._select(rung, 0, w) == {"c": whole_mean}


def test_score_cell_forward_window_sums_and_pooled_metrics():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"), cells_per_model=1)
    res = mode_l.run_lomo(w, cat, seed=0, rungs={"c": mm.Rung("c", lambda seed: _Const(), [{"c": 6.0}])})
    pm = mode_l.pooled_window_metrics(res)
    d = w.merge(cat[["cell_id", "base_model"]], on="cell_id")
    f = d[d["phase"] == "forward"]
    assert pm["n_win_fwd"] == len(f) and pm["n_win_all"] == len(d)
    assert pm["mae_fwd_W"] == pytest.approx(np.abs(6.0 - f["P_W"]).mean())
    r2f = 1 - ((f["P_W"] - 6.0) ** 2).sum() / ((f["P_W"] - f["P_W"].mean()) ** 2).sum()
    r2a = 1 - ((d["P_W"] - 6.0) ** 2).sum() / ((d["P_W"] - d["P_W"].mean()) ** 2).sum()
    assert pm["r2_fwd"] == pytest.approx(r2f) and pm["r2_all"] == pytest.approx(r2a)
    one = res.iloc[0]
    cf = w[(w["cell_id"] == one["cell_id"]) & (w["phase"] == "forward")]
    assert one["mae_fwd_W"] == pytest.approx(np.abs(6.0 - cf["P_W"]).mean())


# ------------------------------------------------------------------ pmic leave-one-workload-out

def test_run_pmic_v1_per_cell_and_family():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    pw = w.rename(columns={})[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]]
    pw = pw[pw.groupby("cell_id").cumcount() % 10 == 0].reset_index(drop=True)
    cat.loc[cat["cell_id"].str.startswith("yolov8m"), "family"] = "B"
    res = mode_l.run_pmic_v1(pw, cat)
    assert set(res["cell_id"]) == set(cat["cell_id"]) and len(res) == 12 and set(res["rung"]) == {"L1", "L1-cell"}
    assert (res["ape"] < 1e-6).all()                 # P is exactly linear in pmic_W
    assert set(res["family"]) == {"A", "B"}
    assert {"E_true_J", "E_pred_J", "n_windows", "model_key"} <= set(res.columns)


def test_run_pmic_v1_leaves_whole_workload_out_and_filters_qc():
    pw = pd.DataFrame({"cell_id": ["a1"] * 5 + ["a2"] * 5 + ["b1"] * 5 + ["x1"] * 5,
                       "dur_s": 5.0, "P_W": np.tile([5.0, 6, 7, 8, 9], 4),
                       "E_mJ": np.tile([5.0, 6, 7, 8, 9], 4) * 5000, "pmic_W": np.tile([1.0, 2, 3, 4, 5], 4),
                       "cpu_key": "2400", "fan_mode": "full", "phase": "forward"})
    cat = pd.DataFrame({"cell_id": ["a1", "a2", "b1", "x1"], "model_key": ["a", "a", "b", "x"],
                        "family": ["A", "A", "B", "B"]})
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": [True, True, True, False]})
    res = mode_l.run_pmic_v1(pw, cat, qc=qc)
    assert set(res["cell_id"]) == {"a1", "a2", "b1"} and set(res["rung"]) == {"L1", "L1-cell"}


# ------------------------------------------------------------------ probe

def _der(cat, lat=100.0):
    rows = []
    for i, c in enumerate(cat["cell_id"]):
        rows.append({"cell_id": c, "mode": "forward", "energy_mJ_per_image": 700.0 + 10 * i,
                     "latency_ms_per_image": lat + i})
    return pd.DataFrame(rows)


def test_probe_pairs_values_and_mode_a_columns():
    import deploy
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    der = _der(cat)
    lomo = mode_l.run_lomo(w, cat, seed=2, rungs={"L2": mm.default_rungs()["L2"]})
    keys = {r["cell_id"]: deploy.mode_a_key_for_cell(r) for _, r in cat.iterrows()}
    pred = pd.DataFrame({"config_key": sorted(set(keys.values())), "e_img_mJ": 650.0})
    pr = mode_l.probe(w, der, cat, pred, seed=2, lomo=lomo)
    assert len(pr) == 3 * 2                          # 3 configs, 2 ordered pairs each
    assert (pr["j"] != pr["k"]).all()
    assert set(pr["rung"]) == {"L2"}
    r = pr[(pr["j"] == "yolov8n_r0") & (pr["k"] == "yolov8n_r1")].iloc[0]
    lat_j = der.loc[der["cell_id"] == "yolov8n_r0", "latency_ms_per_image"].iloc[0]
    p_mean = w[(w["cell_id"] == "yolov8n_r0") & (w["phase"] == "forward")]["P_W"].mean()
    assert r["probe_mJ"] == pytest.approx(p_mean * lat_j, rel=1e-2)
    true_k = der.loc[der["cell_id"] == "yolov8n_r1", "energy_mJ_per_image"].iloc[0]
    assert r["true_mJ"] == true_k and r["ape"] == pytest.approx(abs(r["probe_mJ"] - true_k) / true_k)
    assert r["mode_a_mJ"] == 650.0 and r["mode_a_ape"] == pytest.approx(abs(650 - true_k) / true_k)
    assert r["config_key"] == keys["yolov8n_r0"]


def test_probe_skips_single_cell_configs_ineligible_cells_and_missing_mode_a():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    cat.loc[cat["cell_id"] == "yolov8s_r1", "mode_a_eligible"] = False      # leaves yolov8s with one cell
    lomo = mode_l.run_lomo(w, cat, seed=2, rungs={"L0": mm.default_rungs()["L0"]})
    pr = mode_l.probe(w, _der(cat), cat, pd.DataFrame({"config_key": [], "e_img_mJ": []}), seed=2, lomo=lomo)
    assert {c.split("_")[0] for c in pr["j"]} == {"yolov8n", "yolov8m"}
    assert pr["mode_a_mJ"].isna().all() and pr["mode_a_ape"].isna().all()


# ------------------------------------------------------------------ outputs

def test_run_all_writes_outputs_and_runlog(tmp_path):
    w, cat = make_windows()
    pw = w[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]].iloc[::10]
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    log = tmp_path / "runs.jsonl"
    out = mode_l.run_all(w, pw, cat, _der(cat), qc, seed=5, n_jobs=1, mode_a_pred=None,
                         out_dir=tmp_path / "mode_l", rungs=small_rungs(), runlog_path=log)
    files = sorted(p.name for p in (tmp_path / "mode_l").iterdir())
    rid = out["run_id"]
    assert files == [f"{rid}.parquet", f"{rid}_pmic_v1.parquet"]      # no probe without Mode A
    assert out["probe_skipped"]
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert {r["command"] for r in recs} == {"mode-l"}
    lomo = next(r for r in recs if r["part"] == "lomo")
    assert set(lomo["metrics"]) == set(small_rungs()) and "wall_s" in lomo
    assert lomo["seed"] == 5


def test_run_all_with_mode_a_writes_probe(tmp_path):
    import deploy
    w, cat = make_windows()
    pw = w[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]].iloc[::10]
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    pred = pd.DataFrame({"config_key": sorted({deploy.mode_a_key_for_cell(r) for _, r in cat.iterrows()}),
                         "e_img_mJ": 700.0})
    out = mode_l.run_all(w, pw, cat, _der(cat), qc, seed=5, n_jobs=1, mode_a_pred=pred,
                         out_dir=tmp_path / "mode_l", rungs=small_rungs(), runlog_path=tmp_path / "r.jsonl")
    assert (tmp_path / "mode_l" / f"{out['run_id']}_probe.parquet").is_file() and not out["probe_skipped"]


def test_cli_mode_l_is_registered_and_fails_without_qc(monkeypatch, tmp_path, capsys):
    import cli
    assert "mode-l" in cli.COMMANDS
    monkeypatch.setattr(cli, "REPORTS", tmp_path / "none")
    assert cli.main(["mode-l"]) == 1
    assert "qc_cells.csv" in capsys.readouterr().err


# ------------------------------------------------------------------ L1-cell (exploratory)

def snapshot_windows(n_cells=8, n_win=1600, hold=50, seed=0, c=0.5, noise=0.15):
    """Windows whose pmic_W is a noisy 5 s (50 windows) zero-order-hold snapshot of c * P: the true
    relation is P = pmic / c on cell means, while per-window P fluctuates around the cell mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cells):
        mu = rng.uniform(4, 9)
        P = mu + rng.normal(0, 1.5, n_win)
        snap = np.repeat(c * (mu + rng.normal(0, 1.5, n_win // hold)) + rng.normal(0, noise, n_win // hold), hold)
        rows.append(pd.DataFrame({"cell_id": f"c{i}", "w_idx": np.arange(n_win), "dur_s": 0.1, "P_W": P,
                                  "E_mJ": P * 100.0, "pmic_W": snap, "cpu_key": "2400", "fan_mode": "full",
                                  "busy": 0.6, "phase": "forward"}))
    return pd.concat(rows, ignore_index=True)


def test_l1_cell_recovers_the_slope_and_l1_is_attenuated():
    w = snapshot_windows()
    sw = mm.cell_weights(w)
    b_cell = mm.L1CellPmic().fit(w, {}, sw).b
    b_win = mm.L1Pmic().fit(w, {}, sw).b
    assert b_cell == pytest.approx(2.0, rel=0.10)
    assert b_win < 0.85 * 2.0 and b_win < b_cell


def test_l1_cell_is_calibrated_on_equally_weighted_cell_means_with_l0_fallback():
    w, _ = make_windows()
    w["pmic_W"] = (w["P_W"] - 1.0) / 2.0
    big = w[w["cell_id"] == "yolov8n_r0"]
    w = pd.concat([w, big, big], ignore_index=True)        # a cell with more windows does not count more
    m = mm.L1CellPmic().fit(w, {}, mm.cell_weights(w))
    assert (m.a, m.b) == pytest.approx((1.0, 2.0))
    t = w.iloc[:6].copy()
    t.loc[t.index[:2], "pmic_W"] = NAN
    assert np.isfinite(m.predict(t)).all() and m.fallback_frac(t) == pytest.approx(2 / 6)


def test_l1_cell_rung_is_exploratory_in_lomo_and_pmic_outputs():
    w, cat = make_windows()
    res = mode_l.run_lomo(w, cat, seed=1, rungs={k: v for k, v in mm.default_rungs().items() if k.startswith("L1")})
    assert set(res.loc[res["exploratory"], "rung"]) == {"L1-cell"}
    assert not res.loc[res["rung"] == "L1", "exploratory"].any()
    pw = w[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]].iloc[::10]
    pm = mode_l.run_pmic_v1(pw, make_windows()[1])
    assert set(pm.loc[pm["exploratory"], "rung"]) == {"L1-cell"}
    assert pm.groupby("rung")["cell_id"].nunique().eq(6).all()


# ------------------------------------------------------------------ hardening

@pytest.mark.parametrize("cls,args", [(mm.L2Busy, ()), (mm.L3Counters, ("ridge",)), (mm.L3Counters, ("trees",)),
                                      (mm.L4CountersPmic, ("ridge",)), (mm.L4CountersPmic, ("trees",))])
def test_feature_models_raise_keyerror_on_a_missing_column(cls, args):
    w, _ = make_windows()
    grid = {"alpha": 1.0, "max_depth": 3, "learning_rate": 0.1, "max_iter": 5, "min_samples_leaf": 20}
    for col in ("busy", "cycles_rate", "freq_mhz", "ipc_l2"):
        if cls is mm.L2Busy and col == "ipc_l2":
            continue
        with pytest.raises(KeyError):
            cls(*args).fit(w.drop(columns=[col]), grid, mm.cell_weights(w))
    if cls is mm.L4CountersPmic:
        with pytest.raises(KeyError):
            cls(*args).fit(w.drop(columns=["pmic_age_s"]), grid, mm.cell_weights(w))
    m = cls(*args).fit(w, grid, mm.cell_weights(w))
    with pytest.raises(KeyError):
        m.predict(w.drop(columns=["cycles_rate"]))


def test_l0_and_l1_tolerate_pmic_windows_without_counter_columns():
    w, _ = make_windows()
    pw = w[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]]
    sw = mm.cell_weights(pw)
    assert np.isfinite(mm.L0Lookup().fit(pw, {}, sw).predict(pw)).all()
    assert np.isfinite(mm.L1Pmic().fit(pw, {}, sw).predict(pw)).all()
    assert np.isfinite(mm.L1CellPmic().fit(pw, {}, sw).predict(pw)).all()


def test_run_lomo_uses_every_vision_base_model_and_reports_the_rest():
    """R3: every vision model with counters is an outer fold (before v3 only YOLO models had counters); a model
    outside data.VISION_MODELS (here an LM key) is excluded and reported."""
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m", "effb0", "lm64"))
    res = mode_l.run_lomo(w, cat, seed=1, rungs={"L0": mm.default_rungs()["L0"]})
    assert set(res["fold"]) == {"yolov8n", "yolov8s", "yolov8m", "effb0"}
    assert not any(c.startswith("lm64") for c in res["cell_id"])
    assert res.attrs["excluded"] == {"lm64": {"windows": 2 * N_WIN, "cells": 2}}


def test_cell_apes_zero_truth_is_nan_not_inf():
    w = pd.DataFrame({"cell_id": ["a", "a", "b"], "dur_s": 0.1, "E_mJ": [0.0, 0.0, 100.0]})
    a = mode_l._cell_apes(w, np.array([5.0, 5.0, 1.0]))
    assert np.isnan(a[0]) and a[1] == pytest.approx(0.0)


def _write_mode_a_runs(tmp_path, specs):
    """specs: (run_id, target, seed) in log order; each gets a parquet with one AUTO row."""
    d = tmp_path / "mode_a"
    d.mkdir(exist_ok=True)
    lines = []
    for rid, target, seed in specs:
        pd.DataFrame({"split": ["S-LOMO"], "candidate": ["AUTO"], "config_key": ["k"], "y_pred": [float(len(rid))],
                      "lo90": [0.0], "hi90": [1.0], "flagged": [False]}).to_parquet(d / f"{rid}.parquet")
        lines.append(json.dumps({"run_id": rid, "command": "mode-a", "split": "S-LOMO", "target": target,
                                 "seed": seed}))
    (tmp_path / "runs.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path / "runs.jsonl", d


def test_load_mode_a_preds_prefers_the_run_with_the_same_seed(tmp_path):
    import deploy
    log, d = _write_mode_a_runs(tmp_path, [("e1", "energy_mJ", 7), ("t1", "latency_ms", 7),
                                           ("e22", "energy_mJ", 8), ("t22", "latency_ms", 8)])
    p = deploy.load_mode_a_preds(log, d, seed=7)
    assert p.attrs["run_ids"] == {"energy_mJ": "e1", "latency_ms": "t1"}
    assert p.attrs["seed_matched"] == {"energy_mJ": True, "latency_ms": True}
    q = deploy.load_mode_a_preds(log, d, seed=99)                  # no run with that seed: the latest
    assert q.attrs["run_ids"]["energy_mJ"] == "e22" and q.attrs["seed_matched"]["energy_mJ"] is False
    assert deploy.load_mode_a_preds(log, d).attrs["run_ids"]["energy_mJ"] == "e22"


def test_probe_and_runlog_record_the_mode_a_run(tmp_path):
    import deploy
    w, cat = make_windows()
    pw = w[["cell_id", "dur_s", "P_W", "E_mJ", "pmic_W", "cpu_key", "fan_mode", "phase"]].iloc[::10]
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    pred = pd.DataFrame({"config_key": sorted({deploy.mode_a_key_for_cell(r) for _, r in cat.iterrows()}),
                         "e_img_mJ": 700.0})
    pred.attrs["run_ids"] = {"energy_mJ": "abc123", "latency_ms": "def456"}
    pred.attrs["seed_matched"] = {"energy_mJ": True, "latency_ms": False}
    log = tmp_path / "r.jsonl"
    out = mode_l.run_all(w, pw, cat, _der(cat), qc, seed=5, n_jobs=1, mode_a_pred=pred,
                         out_dir=tmp_path / "ml", rungs={"L0": mm.default_rungs()["L0"]}, runlog_path=log)
    assert set(out["probe"]["mode_a_run_id"]) == {"abc123"} and out["probe"]["mode_a_seed_matched"].all()
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    probe_rec = next(r for r in rec if r["part"] == "probe")
    assert probe_rec["mode_a_run_ids"]["energy_mJ"] == "abc123" and probe_rec["mode_a_seed_matched"]["energy_mJ"]
    assert next(r for r in rec if r["part"] == "lomo")["excluded_non_yolo"] == {}
