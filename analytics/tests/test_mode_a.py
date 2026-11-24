"""Mode A nested evaluation on synthetic data: leakage spy, accuracy, shape, determinism, curves."""
import hashlib
import json
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import evaluate
import mode_a
import mode_a_models as mm
import settings
import splits
import tables

DESC_COLS = ["log_params", "log_macs_per_px", "depth", "frac_macs_conv", "frac_macs_linear",
             "frac_macs_norm", "frac_macs_act", "frac_macs_pool", "frac_macs_other"]
SMALL_GRIDS = {**mm.GRIDS, "M1Ridge": [{"alpha": 0.01}, {"alpha": 1.0}, {"alpha": 100.0}],
               "M2Trees": [{"max_depth": 2, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 3},
                           {"max_depth": 3, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 3}]}


def make_dataset(models, seed=3, noise=0.01, alien=None, clocks=(1500, 2000, 2400), threads=(1, 4)):
    """energy = k x macs^0.9 x clock factor x (1 + noise); latency, power follow."""
    rng = np.random.default_rng(seed)
    rows, desc = [], {}
    for i, m in enumerate(models):
        macs0 = float(np.exp(rng.uniform(18, 22)))
        fr = np.array([0.9, 0.03, 0.03, 0.02, 0.01, 0.01])   # same op mix in every model
        # size-like features track macs, as in the real table
        info = {"log_weight_bytes": np.log(macs0) - 3 + 0.0,
                "log_act_bytes": np.log(macs0) - 2 + 0.0, "depth": 20.0}
        desc[m] = {"log_params": np.log(macs0) - 3 + rng.normal(0, 0.2), "log_macs_per_px": np.log(macs0 / 640 ** 2),
                   "depth": info["depth"], **{f"frac_macs_{g}": f for g, f in
                                              zip(["conv", "linear", "norm", "act", "pool", "other"], fr)}}
        if m == alien:
            desc[m] = {k: v + 60.0 for k, v in desc[m].items()}
        for prec in ("fp32", "int8"):
            for th in threads:
                for mhz in clocks:
                    macs = macs0
                    e = 1e-3 * macs ** 0.9 * (1500 / mhz) ** 0.3 * (0.7 if prec == "int8" else 1.0) / th ** 0.1
                    e *= 1 + noise * rng.normal()
                    lat = 1e-5 * macs ** 0.8 * (1500 / mhz) / th ** 0.5 * (1 + noise * rng.normal())
                    r = {"config_key": f"{m}|{prec}|{th}|{mhz}", "base_model": m, "variant": f"{m}_{prec}",
                         "precision": prec, "imgsz": 640, "batch": 1, "threads": th, "cpu_mhz": mhz,
                         "rig": "oct2026", "n_quantized_modules": 0 if prec == "fp32" else 50, "macs": macs,
                         "log_macs": np.log(macs), "log_params": desc[m]["log_params"], "cpu_ghz": mhz / 1000,
                         "is_int8": int(prec == "int8"), "is_fp16": 0, **info,
                         "act_bytes": float(np.exp(info["log_act_bytes"])), "n_module_calls": float(10 * macs0 ** 0.15),
                         "log_n_calls": float(np.log(10 * macs0 ** 0.15)),
                         "v2f": float(tables.vcore_v(mhz) ** 2 * mhz / 1000),
                         "energy_mJ": e, "latency_ms": lat, "mean_P_W": e / lat}
                    r.update({f"frac_macs_{g}": f for g, f in zip(["conv", "linear", "norm", "act", "pool", "other"], fr)})
                    rows.append(r)
    d = pd.DataFrame.from_dict(desc, orient="index", columns=DESC_COLS)
    d.index.name = "base_model"
    return pd.DataFrame(rows), d


def lomo(t):
    out = []
    for m in sorted(set(t["base_model"])):
        te = sorted(t.loc[t["base_model"] == m, "config_key"])
        tr = sorted(t.loc[t["base_model"] != m, "config_key"])
        out.append(splits.Fold("S-LOMO", m, "new-architecture", tr, te))
    return out


MODELS = [f"m{i}" for i in range(6)]


# ---- leakage spy ------------------------------------------------------------------------------
def test_spy_no_test_key_reaches_any_fit(monkeypatch):
    t, desc = make_dataset(MODELS[:5])
    folds = lomo(t)[:3]
    log = []   # (fit keys, predicted keys, target)

    def spy(base):
        class Spy(base):
            def fit(self, train, target, params=None):
                self._keys, self._target = frozenset(train["config_key"]), target
                self._params = json.dumps(params, sort_keys=True)
                return super().fit(train, target, params)

            def predict(self, test):
                log.append((self._keys, frozenset(test["config_key"]), self._target, self._params))
                return super().predict(test)
        return Spy

    for name in ("M1Ridge", "M2Trees"):
        monkeypatch.setitem(mm.CANDIDATES, name, spy(mm.CANDIDATES[name]))
    res = mode_a.run_split(t, folds, "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert len(res) and log
    owners = []
    for f in folds:
        train, test = frozenset(f.train_keys), frozenset(f.test_keys)
        inner = splits.inner_folds(t[t["config_key"].isin(train)])
        owners.append((train, test, {(frozenset(tr), frozenset(va)) for tr, va in inner}))
    for fk, pk, tg, _ in log:
        # every fit belongs to exactly one outer fold: its outer refit, or one of its inner fits
        mine = [(train, test) for train, test, inner in owners
                if (fk == train and pk <= test) or (fk, pk) in inner]
        assert len(mine) == 1, "a fit/predict pair that matches no outer fold's training protocol"
        train, test = mine[0]
        assert fk <= train and not (fk & test), "an outer test key reached a fit"
        assert not (fk & pk) or pk <= test and not (fk & pk), "predicted keys were in the fit"
    # each outer fold's refit and inner fits were all exercised
    for train, test, inner in owners:
        assert any(fk == train and pk <= test for fk, pk, *_ in log)
        assert {(fk, pk) for fk, pk, *_ in log} >= inner


def test_inner_predictions_are_cached_one_fit_per_point_and_target(monkeypatch):
    t, desc = make_dataset(MODELS[:5])
    seen = Counter()

    class Spy(mm.M1Ridge):
        def fit(self, train, target, params=None):
            seen[(frozenset(train["config_key"]), target, json.dumps(params, sort_keys=True))] += 1
            return super().fit(train, target, params)

    monkeypatch.setitem(mm.CANDIDATES, "M1Ridge", Spy)
    mode_a.run_split(t, lomo(t)[:1], "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert seen and max(seen.values()) == 1


# ---- accuracy, shape --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_result():
    t, desc = make_dataset(MODELS)
    return t, desc, mode_a.run_split(t, lomo(t), "energy_mJ", 7, desc, grids=SMALL_GRIDS)


def test_auto_accuracy_and_interval_coverage(synth_result):
    t, desc, res = synth_result
    auto = res[res["candidate"] == "AUTO"]
    assert evaluate.metrics(auto["y_true"], auto["y_pred"])["mdape"] < 0.1
    assert 0.6 <= evaluate.coverage(auto["y_true"], auto["lo90"], auto["hi90"]) <= 1.0
    assert (auto["lo90"] <= auto["y_pred"]).all() and (auto["y_pred"] <= auto["hi90"]).all()


def test_each_config_once_per_fold_and_candidate(synth_result):
    t, desc, res = synth_result
    expect = {"B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear", "M1Ridge", "M2Trees",
              "M3Factored", "M4Physics", "AUTO"}
    assert set(res["candidate"]) == expect
    assert not res.duplicated(["fold", "config_key", "candidate"]).any()
    assert len(res) == len(t) * len(expect)
    assert list(res.columns) == mode_a.OUT_COLS
    assert set(res["target"]) == {"energy_mJ"} and set(res["seed"]) == {7}
    for p in res["params"]:
        json.loads(p)
    m3 = json.loads(res.loc[res["candidate"] == "M3Factored", "params"].iloc[0])
    assert set(m3) == {"latency", "power"} and m3["latency"]["name"] in ("M1Ridge", "M2Trees")
    auto = json.loads(res.loc[res["candidate"] == "AUTO", "params"].iloc[0])
    assert auto["candidate"] in ("M1Ridge", "M2Trees", "M3Factored", "M4Physics")


def test_latency_and_power_targets_have_no_m3():
    t, desc = make_dataset(MODELS[:5])
    res = mode_a.run_split(t, lomo(t)[:1], "mean_P_W", 1, desc, grids=SMALL_GRIDS)
    assert "M3Factored" not in set(res["candidate"]) and "AUTO" in set(res["candidate"])
    assert (res["y_pred"] > 0).all()


def test_alien_model_is_flagged_and_normal_models_are_not():
    t, desc = make_dataset(MODELS, alien="m5")
    res = mode_a.run_split(t, lomo(t), "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    auto = res[res["candidate"] == "AUTO"]
    assert auto.loc[auto["base_model"] == "m5", "flagged"].all()
    assert (auto.loc[auto["base_model"] == "m5", "novelty"] > 10).all()
    normal = auto[auto["base_model"] != "m5"]
    assert len(normal) and not normal["flagged"].any()
    assert (auto["threshold"] >= 0).all()


def test_novelty_zero_when_test_model_is_in_training_and_not_flagged():
    t, desc = make_dataset(MODELS[:5])
    keys = sorted(t["config_key"])
    fold = splits.Fold("RANDOM", "k0", "sanity", keys[3:], keys[:3])
    res = mode_a.run_split(t, [fold], "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert (res["novelty"] == 0.0).all() and not res["flagged"].any()


def test_threshold_rule():
    assert mode_a._threshold([1.0, 2.0, 3.0], [0.05, 0.10, 0.40]) == 2.0
    assert mode_a._threshold([1.0, 2.0], [0.15, 0.9]) == 0.0   # 0.15 is not below 0.15


def test_ties_pick_first_grid_point_and_few_residuals_give_nan_interval():
    lo, hi = mode_a._interval(np.array([2.0, 3.0]), np.array([0.1]))
    assert np.isnan(lo).all() and np.isnan(hi).all()
    lo, hi = mode_a._interval(np.array([2.0]), np.array([-0.1, 0.0, 0.1]))
    assert lo[0] < 2.0 < hi[0]
    t, desc = make_dataset(MODELS[:4])
    grids = {**SMALL_GRIDS, "M1Ridge": [{"alpha": 1.0}, {"alpha": 1.0}]}   # identical points tie
    s = mode_a._Search(t, desc, grids)
    assert s.select("M1Ridge", "energy_mJ")["params"] == {"alpha": 1.0}


def test_bad_folds_are_rejected():
    t, desc = make_dataset(MODELS[:4])
    keys = sorted(t["config_key"])
    with pytest.raises(ValueError):
        mode_a.run_split(t, [splits.Fold("X", "a", "c", keys[:5], keys[3:6])], "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    with pytest.raises(ValueError):
        mode_a.run_split(t, [splits.Fold("X", "a", "c", keys[:5], ["nope"])], "energy_mJ", 1, desc, grids=SMALL_GRIDS)


# ---- seed reaches the estimator -----------------------------------------------------------------
@pytest.fixture
def hgb_seeds(monkeypatch):
    """Record the random_state of every HistGradientBoostingRegressor built by M2Trees."""
    seen = []
    real = mm.HistGradientBoostingRegressor

    def spy(**kw):
        seen.append(kw.get("random_state"))
        return real(**kw)
    monkeypatch.setattr(mm, "HistGradientBoostingRegressor", spy)
    return seen


def test_seed_reaches_m2_in_run_split_fit_final_and_learning_curve(hgb_seeds, tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    assert settings.seed() != 7
    t, desc = make_dataset(MODELS[:5])
    mode_a.run_split(t, lomo(t)[:2], "energy_mJ", 7, desc, grids=SMALL_GRIDS)
    assert hgb_seeds and set(hgb_seeds) == {7}                      # M2 and M3's M2 parts, inner and outer
    hgb_seeds.clear()
    model, _ = mode_a.fit_final(t, "M2Trees", "energy_mJ", 11, desc, grids=SMALL_GRIDS)
    assert hgb_seeds and set(hgb_seeds) == {11} and model.model_.random_state == 11
    hgb_seeds.clear()
    mode_a.fit_final(t, "M3Factored", "energy_mJ", 13, desc, grids=SMALL_GRIDS)
    assert hgb_seeds and set(hgb_seeds) == {13}
    hgb_seeds.clear()
    ty, dy = make_dataset(YOLO_MODELS, clocks=(1500, 2400), threads=(4,))
    params = SMALL_GRIDS["M2Trees"][0]
    mode_a.learning_curve(ty, "energy_mJ", 17, "M2Trees", params, dy, max_subsets=2)
    assert hgb_seeds and set(hgb_seeds) == {17}


def test_seed_default_for_a_bare_candidate_is_the_config_seed(hgb_seeds):
    t, _ = make_dataset(MODELS[:4])
    mm.M2Trees().fit(t, "energy_mJ", SMALL_GRIDS["M2Trees"][0])
    assert hgb_seeds == [settings.seed()]


def test_seed_leaves_results_unchanged_without_subsampling():
    # HGB without early stopping or subsampling is deterministic: the seed changes nothing here
    t, desc = make_dataset(MODELS[:5])
    folds = lomo(t)[:2]
    a = mode_a.run_split(t, folds, "energy_mJ", settings.seed(), desc, grids=SMALL_GRIDS)
    b = mode_a.run_split(t, folds, "energy_mJ", 7, desc, grids=SMALL_GRIDS)
    cols = [c for c in mode_a.OUT_COLS if c != "seed"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)


# ---- n_resid ------------------------------------------------------------------------------------
def test_n_resid_counts_the_finite_inner_residuals_behind_each_interval():
    t, desc = make_dataset(MODELS[:5])
    res = mode_a.run_split(t, lomo(t)[:2], "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert "n_resid" in res.columns and res["n_resid"].dtype.kind == "i"
    for fold, g in res.groupby("fold"):
        train = t[t["base_model"] != fold]
        n_expect = len(train)   # leave-one-model-out inner folds cover every training row once
        assert set(g["n_resid"]) == {n_expect}
    assert (res["n_resid"] >= 2).all() and res["lo90"].notna().all()


# ---- parallel determinism ---------------------------------------------------------------------
def test_n_jobs_1_and_2_give_identical_results():
    t, desc = make_dataset(MODELS[:5])
    folds = lomo(t)[:3]
    a = mode_a.run_split(t, folds, "energy_mJ", 1, desc, n_jobs=1, grids=SMALL_GRIDS)
    b = mode_a.run_split(t, folds, "energy_mJ", 1, desc, n_jobs=2, grids=SMALL_GRIDS)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ---- fit_final --------------------------------------------------------------------------------
def test_fit_final_pickles_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t, desc = make_dataset(MODELS[:5])
    model, info = mode_a.fit_final(t, "M1Ridge", "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    path = tmp_path / "derived" / "models" / "mode_a_energy_mJ_M1Ridge_new-model.pkl"
    assert info["path"] == str(path) and path.is_file()
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info["params"] in SMALL_GRIDS["M1Ridge"]
    np.testing.assert_allclose(pickle.loads(path.read_bytes()).predict(t), model.predict(t))
    _, info3 = mode_a.fit_final(t, "M3Factored", "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    assert set(info3["params"]) == {"latency", "power"}


# ---- learning curve ---------------------------------------------------------------------------
YOLO_MODELS = list(splits.YOLO) + ["effb0", "mbv3"]


def test_learning_curve_counts_membership_and_determinism(monkeypatch):
    t, desc = make_dataset(YOLO_MODELS, clocks=(1500, 2400), threads=(4,))
    log = []

    class Spy(mm.B0Median):
        def fit(self, train, target, params=None):
            log.append(sorted(set(train["base_model"])))
            return super().fit(train, target, params)

    monkeypatch.setitem(mm.CANDIDATES, "B0Median", Spy)
    full = mode_a.learning_curve(t, "energy_mJ", 5, "B0Median", {}, desc, max_subsets=10)
    assert len(full) == 6 * (10 + 10 + 5 + 1)
    assert set(full["k"]) == {2, 3, 4, 5}
    for _, r in full.iterrows():
        sub = r["subset"].split(";")
        assert len(sub) == r["k"] and r["heldout"] not in sub and set(sub) <= set(splits.YOLO)
    for models in log:   # non-YOLO always present
        assert {"effb0", "mbv3"} <= set(models)
    small_a = mode_a.learning_curve(t, "energy_mJ", 5, "B0Median", {}, desc, max_subsets=3)
    small_b = mode_a.learning_curve(t, "energy_mJ", 5, "B0Median", {}, desc, max_subsets=3)
    assert len(small_a) == 6 * (3 + 3 + 3 + 1)
    pd.testing.assert_frame_equal(small_a, small_b)
    assert small_a["mdape"].notna().all()


# ---- real data smoke --------------------------------------------------------------------------
def test_real_data_one_slomo_fold_energy(real_data):
    import data
    import static
    table = settings.HERE / "reports" / "mode_a_configs.csv"
    if not table.is_file():
        pytest.skip("reports/mode_a_configs.csv absent")
    t = pd.read_csv(table)
    desc = static.model_descriptor(static.static_table(data.catalogue()))
    fold = next(f for f in splits.make_splits(t, settings.seed())["S-LOMO"] if f.fold == "yolo11s")
    t0 = time.perf_counter()
    res = mode_a.run_split(t, [fold], "energy_mJ", settings.seed(), desc, n_jobs=1)
    wall = time.perf_counter() - t0
    print(f"real-data S-LOMO yolo11s energy: {wall:.1f} s")   # printed, not asserted: it depends on the machine
    assert len(res) == len(fold.test_keys) * 9
    assert res["y_pred"].notna().all() and (res["y_pred"] > 0).all()


# ---- CLI --------------------------------------------------------------------------------------
def test_cli_mode_a_writes_parquet_and_runlog(tmp_path, monkeypatch):
    import cli
    import runlog
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t, desc = make_dataset(MODELS[:4])
    monkeypatch.setattr(cli, "_mode_a_inputs", lambda: (t, desc))
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    t.to_csv(tmp_path / "mode_a_configs.csv", index=False)
    monkeypatch.setattr(mm, "GRIDS", SMALL_GRIDS)
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(runlog, "DEFAULT_LOG", log)
    args = cli.build_parser().parse_args(["mode-a", "--split", "S-LOMO", "--target", "mean_P_W", "--n-jobs", "1"])
    assert args.seed is None and args.n_jobs == 1
    assert cli.cmd_mode_a(args) == 0
    rec = json.loads(log.read_text().splitlines()[-1])
    assert (rec["command"], rec["split"], rec["target"], rec["seed"]) == ("mode-a", "S-LOMO", "mean_P_W", settings.seed())
    assert rec["table_sha256"] == hashlib.sha256((tmp_path / "mode_a_configs.csv").read_bytes()).hexdigest()
    assert {"AUTO", "B0Median"} <= set(rec["metrics"]) and rec["wall_s"] >= 0
    df = pd.read_parquet(tmp_path / "derived" / "mode_a" / f"{rec['run_id']}.parquet")
    assert list(df.columns) == mode_a.OUT_COLS and set(df["split"]) == {"S-LOMO"}


# ---- R2: balanced inner selection ---------------------------------------------------------------
def test_balanced_is_mean_over_folds_of_fold_median_abs_log_ratio():
    y = [np.array([1.0, 1.0, 1.0]), np.array([2.0])]
    p = [np.array([np.e, 1.0, 1.0]), np.array([2.0 * np.e ** 0.5])]
    # fold medians: median(1, 0, 0) = 0 ; 0.5 -> mean 0.25
    assert mode_a._balanced(y, p) == pytest.approx(0.25)


def test_balanced_is_inf_for_non_finite_or_non_positive_predictions_and_for_no_folds():
    y = [np.array([1.0, 2.0]), np.array([3.0])]
    assert mode_a._balanced(y, [np.array([1.0, np.nan]), np.array([3.0])]) == float("inf")
    assert mode_a._balanced(y, [np.array([1.0, 2.0]), np.array([-3.0])]) == float("inf")
    assert mode_a._balanced([], []) == float("inf")


class _Stub:
    """Predicts y x (1 + err[test model]) with the error table taken from its params (a test double)."""
    def fit(self, train, target, params=None):
        self.err_ = params["err"]
        return self

    def predict(self, test):
        return test["energy_mJ"].to_numpy(float) * np.array([1 + self.err_[m] for m in test["base_model"]])


def _unbalanced_table():
    rows = [{"config_key": f"big|{i}", "base_model": "big", "energy_mJ": 100.0 + i} for i in range(30)]
    rows += [{"config_key": f"s{j}|{i}", "base_model": f"s{j}", "energy_mJ": 50.0 + i} for j in (1, 2) for i in range(3)]
    return pd.DataFrame(rows)


def test_balanced_selection_picks_a_grid_point_pooled_mdape_would_not(monkeypatch):
    """One many-row model (30 rows) and two few-row models (3 rows each). Point P is accurate on the big model
    only (1 % there, 40 % on the small ones); point Q is 5 % everywhere. Pooled MdAPE (30 of 36 rows are the big
    model) prefers P; the balanced error (each held-out model once) prefers Q."""
    monkeypatch.setitem(mm.CANDIDATES, "Stub", _Stub)
    P = {"err": {"big": 0.01, "s1": 0.4, "s2": 0.4}}
    Q = {"err": {"big": 0.05, "s1": 0.05, "s2": 0.05}}
    grids = {"Stub": [P, Q]}
    search = mode_a._Search(_unbalanced_table(), pd.DataFrame(), grids, 1)
    y = search.truth("energy_mJ")
    pooled = [mode_a._mdape(mode_a._pooled(y), mode_a._pooled(search.inner_pred("Stub", g, "energy_mJ"))) for g in (P, Q)]
    assert pooled[0] < pooled[1]                                  # the R1 rule would choose P
    name, params, preds, score = search._best(["Stub"], "energy_mJ")
    assert params == Q and score == pytest.approx(np.log(1.05))
    sel = search.select("Stub", "energy_mJ")
    assert sel["params"] == Q and sel["score"] == pytest.approx(np.log(1.05))
    assert sel["mdape"] == pytest.approx(0.05)


def test_ties_in_the_balanced_score_keep_the_first_grid_point(monkeypatch):
    monkeypatch.setitem(mm.CANDIDATES, "Stub", _Stub)
    A = {"err": {"big": 0.05, "s1": 0.05, "s2": 0.05}, "tag": "first"}
    B = {"err": {"big": 0.05, "s1": 0.05, "s2": 0.05}, "tag": "second"}
    search = mode_a._Search(_unbalanced_table(), pd.DataFrame(), {"Stub": [A, B]}, 1)
    assert search._best(["Stub"], "energy_mJ")[1]["tag"] == "first"


def test_m4_is_a_candidate_and_in_auto_for_every_target_and_m3_only_for_energy():
    for target in ("energy_mJ", "latency_ms", "mean_P_W"):
        assert "M4Physics" in mode_a.CANDIDATES_BY_TARGET[target] and "M4Physics" in mode_a.AUTO_AMONG[target]
    assert "M3Factored" in mode_a.AUTO_AMONG["energy_mJ"]
    assert "M3Factored" not in mode_a.AUTO_AMONG["latency_ms"] + mode_a.AUTO_AMONG["mean_P_W"]


def test_m4_energy_params_are_the_parts_chosen_on_their_own_targets(synth_result):
    t, desc, res = synth_result
    p = json.loads(res.loc[res["candidate"] == "M4Physics", "params"].iloc[0])
    assert set(p) == {"latency", "power"} and p["latency"]["beta"] in (0.5, 0.75, 1.0)
    assert p["power"]["alpha"] in (0.01, 0.1, 1, 10, 100)


# ---- R2: interval weights, threshold folds, regimes ---------------------------------------------------
def test_wquantile_equal_weights_known_positions_and_weighting():
    v = np.array([3.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(mode_a.wquantile(v, np.ones(4), [0.375, 0.5, 0.625]), [2.0, 2.5, 3.0])
    np.testing.assert_allclose(mode_a.wquantile(v, np.ones(4), [0.0, 1.0]), [1.0, 4.0])   # clamped
    # a heavy weight on the largest value pulls the median to it
    assert mode_a.wquantile(v, np.array([1.0, 1.0, 1.0, 10.0]), 0.5)[0] > 3.5
    # scaling all weights changes nothing
    np.testing.assert_allclose(mode_a.wquantile(v, np.full(4, 7.0), 0.3), mode_a.wquantile(v, np.ones(4), 0.3))


def test_residuals_weight_every_inner_fold_equally():
    y = [np.ones(3), np.ones(1), np.array([1.0, 1.0])]
    p = [np.exp([0.1, 0.2, 0.3]), np.array([np.nan]), np.exp([-0.5, -0.4])]
    r, w = mode_a._residuals(y, p)
    np.testing.assert_allclose(r, [-0.1, -0.2, -0.3, 0.5, 0.4])      # the NaN fold contributes nothing
    np.testing.assert_allclose(w, [1 / 3] * 3 + [1 / 2] * 2)
    assert w[:3].sum() == pytest.approx(w[3:].sum())
    # the 2-residual fold weighs as much as the 3-residual one, so the weighted median is above the plain one
    assert mode_a.wquantile(r, w, 0.5)[0] > mode_a.wquantile(r, np.ones(5), 0.5)[0] + 0.1


def _seen_fold(t):
    """Every 4th configuration held out: all base models stay in training (seen-model regime)."""
    keys = sorted(t["config_key"])
    te = keys[::4]
    return splits.Fold("S-LCO", "k0", "seen", [k for k in keys if k not in set(te)], te)


def test_regime_decides_the_inner_folds_and_the_threshold_always_uses_leave_one_model_out(monkeypatch):
    t, desc = make_dataset(MODELS[:5])
    seen = _seen_fold(t)
    new = lomo(t)[0]
    used = []
    orig = mode_a._Search.__init__

    def spy(self, train_t, desc_, grids, seed=None, folds=None):
        used.append(None if folds is None else frozenset(map(lambda f: frozenset(f[1]), folds)))
        orig(self, train_t, desc_, grids, seed, folds)

    monkeypatch.setattr(mode_a._Search, "__init__", spy)
    train = t[t["config_key"].isin(seen.train_keys)].reset_index(drop=True)
    res = mode_a.run_split(t, [seen], "energy_mJ", 3, desc, grids=SMALL_GRIDS)
    want_random = frozenset(frozenset(va) for _, va in splits.random_inner_folds(train, 3, 0))
    lomo_sets = lambda tr: frozenset(frozenset(va) for _, va in splits.inner_folds(tr))   # noqa: E731
    assert used[0] == want_random and used[1] is None              # selection on random folds, threshold on LOMO
    auto = res[res["candidate"] == "AUTO"]
    spec = json.loads(auto["params"].iloc[0])
    monkeypatch.undo()
    s = mode_a._Search(train, desc, SMALL_GRIDS, 3)                # leave-one-model-out
    preds = s.preds_of(spec["candidate"], spec["params"], "energy_mJ")
    novs = [max(mode_a._model_novelty(desc, m, tr["base_model"]) for m in set(va["base_model"])) for tr, va in s.folds]
    want = mode_a._threshold(novs, [mode_a._mdape(y, p) for y, p in zip(s.truth("energy_mJ"), preds)])
    assert set(auto["threshold"]) == {want}
    used.clear()
    monkeypatch.setattr(mode_a._Search, "__init__", spy)
    mode_a.run_split(t, [new], "energy_mJ", 3, desc, grids=SMALL_GRIDS)
    train_new = t[t["config_key"].isin(new.train_keys)].reset_index(drop=True)
    assert used == [lomo_sets(train_new)]                          # new model: leave-one-model-out, one search only


def test_seen_model_fold_is_deterministic_across_n_jobs_and_n_resid_counts_training_rows():
    t, desc = make_dataset(MODELS[:5])
    folds = [_seen_fold(t)]
    a = mode_a.run_split(t, folds, "energy_mJ", 1, desc, n_jobs=1, grids=SMALL_GRIDS)
    b = mode_a.run_split(t, folds, "energy_mJ", 1, desc, n_jobs=2, grids=SMALL_GRIDS)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(a["n_resid"]) == {len(folds[0].train_keys)}         # every training row validated once


def test_fit_final_regimes_name_the_pickle_and_select_on_their_own_inner_folds(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t, desc = make_dataset(MODELS[:5])
    _, new = mode_a.fit_final(t, "M1Ridge", "energy_mJ", 1, desc, grids=SMALL_GRIDS)
    _, seen = mode_a.fit_final(t, "M1Ridge", "energy_mJ", 1, desc, SMALL_GRIDS, regime="seen-model")
    assert new["regime"] == "new-model" and seen["regime"] == "seen-model"
    assert new["path"].endswith("mode_a_energy_mJ_M1Ridge_new-model.pkl")
    assert seen["path"].endswith("mode_a_energy_mJ_M1Ridge_seen-model.pkl") and new["path"] != seen["path"]
    with pytest.raises(ValueError, match="regime"):
        mode_a.fit_final(t, "M1Ridge", "energy_mJ", 1, desc, SMALL_GRIDS, regime="other")
    # the seen-model choice is the one a random-5-fold search makes
    folds = splits.random_inner_folds(t, 1)
    assert seen["params"] == mode_a._Search(t, desc, SMALL_GRIDS, 1, folds).select("M1Ridge", "energy_mJ")["params"]
