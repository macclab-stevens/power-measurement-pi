"""R4 additions to Mode A: conformal interval columns, ENS / STACK combinations, M5Pooled in the runner, AUTO_R4."""
import json

import numpy as np
import pandas as pd
import pytest

import mode_a
import mode_a_models as mm
import splits
from test_mode_a import MODELS, SMALL_GRIDS, lomo, make_dataset

LEGACY = ["B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear", "M1Ridge", "M2Trees", "M3Factored",
          "M4Physics", "AUTO"]


# ---- conformal_halfwidth ----------------------------------------------------------------------------
def test_conformal_equal_weights_picks_the_expected_order_statistic():
    # K folds of one residual each: level (0.9)(1 + 1/K), the ceil(level * K)-th smallest score (the max once level is 1)
    for k, rank in ((9, 9), (10, 10), (19, 18), (29, 27), (39, 36)):
        r = np.random.default_rng(k).permutation(np.arange(1, k + 1)) * 0.01 * np.where(np.arange(k) % 2, -1, 1)
        q = mode_a.conformal_halfwidth(r, np.ones(k))
        assert q == pytest.approx(rank * 0.01), (k, rank)
    # 9 folds: level exactly 1 (not 0.99999...): the maximum, never beyond it
    assert mode_a.conformal_halfwidth(np.arange(9.0), np.ones(9)) == 8.0


def test_conformal_uses_absolute_scores_and_ties():
    assert mode_a.conformal_halfwidth([-3.0, 1.0, 2.0, 0.5], np.ones(4)) == 3.0
    assert mode_a.conformal_halfwidth([1.0] * 5, np.ones(5)) == 1.0


def test_conformal_unequal_fold_sizes_weights_make_each_fold_count_equally():
    # fold 0: 100 residuals of 0.1 (weight 1/100 each); 19 more folds of one residual each, scores 1..19 (weight 1)
    resid = np.concatenate([np.full(100, 0.1), np.arange(1.0, 20.0) * np.where(np.arange(19) % 2, -1, 1)])
    w = np.concatenate([np.full(100, 0.01), np.ones(19)])
    # K = 20, level 0.9 * 21 / 20 = 0.945: 18.9 fold-units; the score s sits at s + 1 units (fold 0 is one unit)
    assert mode_a.conformal_halfwidth(resid, w) == 18.0
    # pooling the 119 rows with equal weights would let the 100-row fold decide (113th of 119 -> score 13)
    assert mode_a.conformal_halfwidth(resid, np.ones(resid.size), n_folds=20) != 18.0
    # the weights agree with _residuals: it weighs each inner fold's residuals 1 / n_fold
    r, ww = mode_a._residuals([np.ones(3), np.ones(1)], [np.exp(-np.array([.1, .2, .3])), np.exp([-0.5])])
    np.testing.assert_allclose(ww, [1 / 3] * 3 + [1.0])
    assert mode_a.conformal_halfwidth(r, ww) == pytest.approx(0.5)      # K = 2: level 1 -> the largest score


def test_conformal_nan_with_fewer_than_two_residuals():
    assert np.isnan(mode_a.conformal_halfwidth([0.2], [1.0])) and np.isnan(mode_a.conformal_halfwidth([], []))
    assert mode_a.conformal_halfwidth([0.2, 0.3], [1.0, 1.0]) == 0.3


# ---- the runner -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runs():
    t, desc = make_dataset(MODELS)
    return t, desc, {tg: mode_a.run_split(t, lomo(t), tg, 7, desc, grids=SMALL_GRIDS) for tg in mm.TARGETS}


def _by(res, cand):
    return res[res["candidate"] == cand].sort_values(["fold", "config_key"]).reset_index(drop=True)


def test_extra_columns_follow_hi90_and_bracket_the_prediction(runs):
    _, _, res = runs
    assert mode_a.OUT_COLS.index("lo90_cf") == mode_a.OUT_COLS.index("hi90") + 1
    assert mode_a.OUT_COLS.index("hi90_cf") == mode_a.OUT_COLS.index("lo90_cf") + 1
    for tg, r in res.items():
        assert list(r.columns) == mode_a.OUT_COLS
        assert r["lo90_cf"].notna().all() and (r["lo90_cf"] <= r["y_pred"]).all() and (r["y_pred"] <= r["hi90_cf"]).all()
        # symmetric in log space around the prediction
        np.testing.assert_allclose(np.log(r["y_pred"] / r["lo90_cf"]), np.log(r["hi90_cf"] / r["y_pred"]))


def test_conformal_interval_is_built_from_the_inner_residuals_of_the_block(runs):
    t, desc, res = runs
    r = _by(res["energy_mJ"], "M1Ridge")
    fold = r[r["fold"] == "m2"]
    train = t[t["base_model"] != "m2"]
    search = mode_a._Search(train, desc, SMALL_GRIDS, 7)
    s = search.select("M1Ridge", "energy_mJ")
    resid, wts = mode_a._residuals(search.truth("energy_mJ"), s["preds"])
    q = mode_a.conformal_halfwidth(resid, wts)
    assert len(search.folds) == train["base_model"].nunique()          # K = one inner fold per base model
    np.testing.assert_allclose(np.log(fold["y_pred"] / fold["lo90_cf"]), q)
    np.testing.assert_allclose(np.log(fold["hi90_cf"] / fold["y_pred"]), q)


def test_existing_candidates_are_unchanged_by_the_r4_additions(runs, monkeypatch):
    t, desc, res = runs
    legacy = {tg: [c for c in mode_a.CANDIDATES_BY_TARGET[tg] if c not in mode_a.R4_EXTRA] for tg in mm.TARGETS}
    monkeypatch.setattr(mode_a, "CANDIDATES_BY_TARGET", legacy)
    monkeypatch.setattr(mode_a, "R4_EXTRA", [])
    for tg in mm.TARGETS:
        old = mode_a.run_split(t, lomo(t), tg, 7, desc, grids=SMALL_GRIDS)
        new = res[tg]
        assert set(old["candidate"]) - {"AUTO_R4"} == set(legacy[tg]) | {"AUTO"}
        keep = new[new["candidate"].isin(set(old["candidate"]) - {"AUTO_R4"})].reset_index(drop=True)
        old = old[old["candidate"] != "AUTO_R4"].reset_index(drop=True)
        pd.testing.assert_frame_equal(old, keep, check_exact=True)


@pytest.mark.parametrize("target", mm.TARGETS)
def test_ens_is_the_geometric_mean_of_the_selected_m1_and_m4(runs, target):
    _, _, res = runs
    r = res[target]
    m1, m4, ens = _by(r, "M1Ridge"), _by(r, "M4Physics"), _by(r, "ENS")
    np.testing.assert_allclose(ens["y_pred"], np.sqrt(m1["y_pred"] * m4["y_pred"]), rtol=1e-12)
    p = json.loads(ens["params"].iloc[0])
    assert set(p) == {"M1Ridge", "M4Physics"}


@pytest.mark.parametrize("target", mm.TARGETS)
def test_stack_blends_in_log_space_with_the_chosen_weight(runs, target):
    _, _, res = runs
    r = res[target]
    m1, m4, st = _by(r, "M1Ridge"), _by(r, "M4Physics"), _by(r, "STACK")
    for fold, g in st.groupby("fold"):
        p = json.loads(g["params"].iloc[0])
        assert p["w"] in mode_a.STACK_W and {"M1Ridge", "M4Physics"} <= set(p)
        a, b = m1[m1["fold"] == fold], m4[m4["fold"] == fold]
        np.testing.assert_allclose(g["y_pred"], np.exp(p["w"] * np.log(a["y_pred"]) + (1 - p["w"]) * np.log(b["y_pred"])),
                                   rtol=1e-12)
        assert p["M1Ridge"] == json.loads(a["params"].iloc[0]) and p["M4Physics"] == json.loads(b["params"].iloc[0])


def test_search_ens_and_stack_inner_predictions_score_and_w_choice(runs):
    t, desc, _ = runs
    train = t[t["base_model"] != "m1"]
    s = mode_a._Search(train, desc, SMALL_GRIDS, 7)
    for target in ("energy_mJ", "mean_P_W"):
        a, b = s.select("M1Ridge", target), s.select("M4Physics", target)
        ens, st = s.select("ENS", target), s.select("STACK", target)
        y = s.truth(target)
        for pe, pa, pb in zip(ens["preds"], a["preds"], b["preds"]):
            np.testing.assert_allclose(pe, np.sqrt(pa * pb))
        assert ens["score"] == mode_a._balanced(y, ens["preds"])
        scores = [mode_a._balanced(y, s._stack(w, a["preds"], b["preds"])) for w in mode_a.STACK_W]
        assert st["params"]["w"] == mode_a.STACK_W[int(np.argmin(scores))] and st["score"] == min(scores)
        # w = 1 is M1Ridge, w = 0 is M4Physics
        for pe, pa in zip(s._stack(1.0, a["preds"], b["preds"]), a["preds"]):
            np.testing.assert_allclose(pe, pa)
        # the preds_of route of the chosen params reproduces them
        for name, sel in (("ENS", ens), ("STACK", st)):
            for x, y_ in zip(s.preds_of(name, sel["params"], target), sel["preds"]):
                np.testing.assert_allclose(x, y_)


def test_stack_ties_keep_the_first_grid_point():
    t, desc = make_dataset(MODELS[:4])
    s = mode_a._Search(t, desc, SMALL_GRIDS, 1)
    # identical parts: every w gives the same predictions, so the first w wins
    sel = {"params": {}, "preds": s.select("M1Ridge", "mean_P_W")["preds"], "score": 0.0, "mdape": 0.0}
    s.select = lambda name, target, _o=s.select: sel if name in mode_a.COMBO_PARTS else _o(name, target)
    assert mode_a._Search.select(s, "STACK", "mean_P_W")["params"]["w"] == mode_a.STACK_W[0]


def test_m5pooled_is_a_normal_factored_candidate_in_the_runner(runs):
    t, desc, res = runs
    r = _by(res["energy_mJ"], "M5Pooled")
    p = json.loads(r["params"].iloc[0])
    assert set(p) == {"latency", "power"} and "beta" in p["latency"] and "lam" in p["latency"]
    assert "alpha" in p["power"] and "lam" in p["power"]
    for tg, key in (("latency_ms", "lam"), ("mean_P_W", "lam")):
        q = json.loads(_by(res[tg], "M5Pooled")["params"].iloc[0])
        assert key in q and q in mm.GRIDS["M5Pooled"][tg]
    # the energy prediction is the model refitted with the chosen parts
    fold = "m3"
    train, test = t[t["base_model"] != fold], t[t["base_model"] == fold]
    p3 = json.loads(r[r["fold"] == fold]["params"].iloc[0])
    pred = mm.M5Pooled().fit(train, "energy_mJ", p3).predict(test.sort_values("config_key"))
    np.testing.assert_allclose(r[r["fold"] == fold]["y_pred"], pred, rtol=1e-12)


@pytest.mark.parametrize("target", mm.TARGETS)
def test_auto_r4_is_the_lowest_inner_score_among_auto_candidates_and_the_extras(runs, target):
    t, desc, res = runs
    r = res[target]
    pool = mode_a.AUTO_AMONG[target] + mode_a.R4_EXTRA
    assert mode_a.R4_EXTRA == ["ENS", "STACK", "M5Pooled"]
    for fold in sorted(set(r["fold"])):
        train = t[t["base_model"] != fold]
        s = mode_a._Search(train, desc, SMALL_GRIDS, 7)
        sc = {n: s.select(n, target)["score"] for n in pool}
        want = min(pool, key=lambda n: sc[n])
        want_auto = min(mode_a.AUTO_AMONG[target], key=lambda n: sc[n])
        g = r[r["fold"] == fold]
        a4, a = g[g["candidate"] == "AUTO_R4"], g[g["candidate"] == "AUTO"]
        assert json.loads(a4["params"].iloc[0])["candidate"] == want
        assert json.loads(a["params"].iloc[0])["candidate"] == want_auto          # AUTO unchanged
        src = g[g["candidate"] == want]
        np.testing.assert_array_equal(a4["y_pred"].to_numpy(), src["y_pred"].to_numpy())
        for c in ("lo90", "hi90", "lo90_cf", "hi90_cf"):
            np.testing.assert_array_equal(a4[c].to_numpy(), src[c].to_numpy())


def test_candidates_by_target_lists_extras_after_the_existing_ones():
    for tg, names in mode_a.CANDIDATES_BY_TARGET.items():
        assert names[-3:] == ["ENS", "STACK", "M5Pooled"] and names.index("M4Physics") < names.index("ENS")
        assert set(mode_a.AUTO_AMONG[tg]).isdisjoint(mode_a.R4_EXTRA)
    assert mode_a.CANDIDATES_BY_TARGET["energy_mJ"][-4] == "M4Physics"


def test_seen_model_fold_has_the_extra_blocks_too():
    t, desc = make_dataset(MODELS[:5])
    keys = sorted(t["config_key"])
    te = [k for k in keys if k.startswith("m1|") and k.endswith("|2000")]
    fold = splits.Fold("S", "a", "c", [k for k in keys if k not in te], te)
    res = mode_a.run_split(t, [fold], "energy_mJ", 3, desc, grids=SMALL_GRIDS)
    assert {"ENS", "STACK", "M5Pooled", "AUTO_R4"} <= set(res["candidate"])
    assert res["lo90_cf"].notna().all() and (res["lo90_cf"] < res["hi90_cf"]).all()


# ---- frozen models ------------------------------------------------------------------------------------
def test_fit_final_rejects_combinations_and_freezes_m5pooled(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t, desc = make_dataset(MODELS[:5])
    for name in ("ENS", "STACK"):
        with pytest.raises(ValueError, match="combination"):
            mode_a.fit_final(t, name, "energy_mJ", 1, desc, grids=SMALL_GRIDS)
        with pytest.raises(ValueError):
            mode_a._make(name, desc)
    model, info = mode_a.fit_final(t, "M5Pooled", "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert isinstance(model, mm.M5Pooled) and set(info["params"]) == {"latency", "power"}
    assert np.all(model.predict(t) > 0)


def test_fit_candidates_keeps_the_frozen_candidate_set_unless_r4(runs):
    t, desc, _ = runs
    a = mode_a.fit_candidates(t, "mean_P_W", 1, desc, "new-model", SMALL_GRIDS)
    assert set(a) == set(LEGACY) - {"M3Factored"}
    b = mode_a.fit_candidates(t, "mean_P_W", 1, desc, "new-model", SMALL_GRIDS, r4=True)
    assert set(b) == set(a) | {"M5Pooled"}
    assert b["AUTO"]["chosen"] == a["AUTO"]["chosen"] and b["M5Pooled"]["model"].predict(t).shape == (len(t),)
