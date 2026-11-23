"""Evaluation utilities: hand-computed metrics, deterministic bootstrap, Wilcoxon, Holm, coverage."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import evaluate


# ---- metrics ----------------------------------------------------------------------------------
def test_metrics_hand_computed():
    y = [100.0, 200.0, 400.0]
    yhat = [110.0, 180.0, 400.0]
    m = evaluate.metrics(y, yhat)
    # signed % errors: +0.10, -0.10, 0.0 ; abs: 0.10, 0.10, 0.0
    assert m["n"] == 3
    assert m["mdape"] == pytest.approx(0.10)
    assert m["mape"] == pytest.approx(0.2 / 3)
    assert m["mae"] == pytest.approx((10 + 20 + 0) / 3)
    assert m["rmse"] == pytest.approx(math.sqrt((100 + 400 + 0) / 3))
    assert m["bias_pct"] == pytest.approx(0.0)
    assert m["mdale"] == pytest.approx(np.median([abs(math.log(1.1)), abs(math.log(0.9)), 0.0]))
    assert set(m) == {"n", "mdape", "mape", "mae", "rmse", "bias_pct", "mdale"}


def test_metrics_signed_bias_and_median():
    y = np.array([10.0, 10.0, 10.0, 10.0])
    yhat = np.array([12.0, 13.0, 15.0, 30.0])  # +0.2 +0.3 +0.5 +2.0
    m = evaluate.metrics(y, yhat)
    assert m["bias_pct"] == pytest.approx(0.4)      # median of 0.3, 0.5
    assert m["mdape"] == pytest.approx(0.4)
    assert m["mape"] == pytest.approx(0.75)
    assert m["bias_pct"] > 0


def test_metrics_underprediction_is_negative_bias():
    m = evaluate.metrics([10.0, 20.0], [5.0, 10.0])
    assert m["bias_pct"] == pytest.approx(-0.5)
    assert m["mdale"] == pytest.approx(math.log(2))


def test_metrics_drops_nonfinite_pairs_and_reports_n_used():
    y = [1.0, 2.0, np.nan, 4.0, 5.0]
    yhat = [1.0, np.nan, 3.0, 4.0, np.inf]
    m = evaluate.metrics(y, yhat)
    assert m["n"] == 2
    assert m["mape"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_metrics_nonpositive_truth_raises():
    with pytest.raises(ValueError):
        evaluate.metrics([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        evaluate.metrics([1.0, -2.0], [1.0, 1.0])


def test_metrics_nonpositive_truth_dropped_pair_does_not_raise():
    # a non-finite pair is dropped before the positivity check
    m = evaluate.metrics([1.0, np.nan], [1.0, 1.0])
    assert m["n"] == 1


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate.metrics([1.0, 2.0], [1.0])


def test_metrics_empty_is_nan_with_n_zero():
    m = evaluate.metrics([np.nan], [1.0])
    assert m["n"] == 0
    assert all(math.isnan(m[k]) for k in m if k != "n")


def test_metrics_nonpositive_prediction_gives_nan_mdale_only():
    m = evaluate.metrics([10.0, 10.0], [0.0, 10.0])
    assert math.isnan(m["mdale"])
    assert m["mape"] == pytest.approx(0.5)


def test_metrics_accepts_series_and_lists():
    s = pd.Series([1.0, 2.0])
    assert evaluate.metrics(s, s)["mape"] == 0.0
    assert evaluate.metrics([1.0], [2.0])["mape"] == pytest.approx(1.0)


# ---- coverage ---------------------------------------------------------------------------------
def test_coverage_inclusive_bounds():
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [1.0, 2.5, 2.0, 0.0]
    hi = [1.0, 3.0, 3.0, 3.9]
    # in: yes (equal), no (below lo), yes (equal hi), no (above hi)
    assert evaluate.coverage(y, lo, hi) == pytest.approx(0.5)


def test_coverage_drops_nonfinite_and_empty_is_nan():
    assert evaluate.coverage([1.0, 2.0], [0.0, np.nan], [2.0, 3.0]) == pytest.approx(1.0)
    assert math.isnan(evaluate.coverage([np.nan], [0.0], [1.0]))


def test_coverage_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate.coverage([1.0, 2.0], [0.0], [1.0, 2.0])


# ---- cluster bootstrap ------------------------------------------------------------------------
def _frame():
    return pd.DataFrame({"g": ["a", "a", "b", "b", "b", "c", "d", "d"],
                         "x": [1.0, 3.0, 2.0, 4.0, 6.0, 10.0, 5.0, 7.0]})


def _mean(d):
    return float(d["x"].mean())


def test_bootstrap_deterministic_for_fixed_seed():
    a = evaluate.cluster_bootstrap(_frame(), "g", _mean, n=300, seed=7)
    b = evaluate.cluster_bootstrap(_frame(), "g", _mean, n=300, seed=7)
    c = evaluate.cluster_bootstrap(_frame(), "g", _mean, n=300, seed=8)
    assert a == b
    assert a != c


def test_bootstrap_interval_contains_point_estimate():
    df = _frame()
    lo, hi = evaluate.cluster_bootstrap(df, "g", _mean, n=500, seed=1)
    assert lo <= _mean(df) <= hi
    assert lo < hi


def test_bootstrap_resamples_whole_groups_with_replacement():
    """A statistic that is the row count shows groups are drawn whole and as many as exist."""
    df = _frame()                      # group sizes a=2 b=3 c=1 d=2
    seen = set()

    def stat(d):
        seen.add(len(d))
        return float(len(d))

    lo, hi = evaluate.cluster_bootstrap(df, "g", stat, n=400, seed=3)
    assert min(seen) >= 4 * 1 and max(seen) <= 4 * 3   # four draws of sizes 1..3
    assert len(seen) > 3                               # sizes vary: groups repeat
    assert 4 <= lo <= hi <= 12


def test_bootstrap_known_draw_matches_manual_resample():
    df = _frame()
    n = 50
    rng = np.random.default_rng(11)
    groups = sorted(df["g"].unique())
    manual = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        manual.append(_mean(pd.concat([df[df["g"] == groups[i]] for i in pick])))
    expect = (float(np.percentile(manual, 2.5)), float(np.percentile(manual, 97.5)))
    got = evaluate.cluster_bootstrap(df, "g", _mean, n=n, seed=11)
    assert got == pytest.approx(expect)


def test_bootstrap_single_group_returns_point_twice():
    df = pd.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 6.0]})
    assert evaluate.cluster_bootstrap(df, "g", _mean, n=50, seed=0) == (3.0, 3.0)


def test_bootstrap_does_not_mutate_input_and_handles_nonsorted_index():
    df = _frame().sample(frac=1.0, random_state=0)
    before = df.copy()
    evaluate.cluster_bootstrap(df, "g", _mean, n=20, seed=0)
    pd.testing.assert_frame_equal(df, before)


def test_bootstrap_empty_or_null_groups_raise():
    with pytest.raises(ValueError):
        evaluate.cluster_bootstrap(_frame().iloc[0:0], "g", _mean, n=10, seed=0)
    bad = _frame()
    bad.loc[0, "g"] = None
    with pytest.raises(ValueError):
        evaluate.cluster_bootstrap(bad, "g", _mean, n=10, seed=0)


# ---- paired Wilcoxon --------------------------------------------------------------------------
def test_wilcoxon_matches_scipy_and_median_diff_sign():
    a = np.array([1.0, 2.0, 3.5, 4.0, 6.0, 7.5, 8.0])
    b = np.array([0.5, 1.0, 3.0, 3.0, 4.0, 6.0, 5.0])
    r = evaluate.paired_wilcoxon(a, b)
    ref = stats.wilcoxon(a - b, zero_method="wilcox")
    assert r["p"] == pytest.approx(ref.pvalue)
    assert r["stat"] == pytest.approx(ref.statistic)
    assert r["median_diff"] == pytest.approx(np.median(a - b))
    assert r["median_diff"] > 0
    assert set(r) == {"stat", "p", "median_diff"}


def test_wilcoxon_all_zero_differences():
    r = evaluate.paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["p"] == 1.0
    assert math.isnan(r["stat"])
    assert r["median_diff"] == 0.0


def test_wilcoxon_fewer_than_two_nonzero_differences():
    r = evaluate.paired_wilcoxon([1.0, 2.0, 5.0], [1.0, 2.0, 4.0])
    assert r["p"] == 1.0 and math.isnan(r["stat"])
    assert r["median_diff"] == 0.0


def test_wilcoxon_drops_nonfinite_pairs():
    a = [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 8.0, 9.0]
    b = [0.0, 0.5, 1.0, 1.0, np.nan, 3.0, 5.0, 4.0]
    r = evaluate.paired_wilcoxon(a, b)
    keep = [0, 1, 2, 5, 6, 7]
    d = np.array(a)[keep] - np.array(b)[keep]
    assert r["p"] == pytest.approx(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    assert r["median_diff"] == pytest.approx(np.median(d))


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate.paired_wilcoxon([1.0, 2.0], [1.0])


# ---- Holm -------------------------------------------------------------------------------------
def test_holm_brief_example():
    assert evaluate.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_preserves_order_and_caps_at_one():
    out = evaluate.holm([0.5, 0.001, 0.9, 0.04])
    # sorted: .001*4=.004, .04*3=.12, .5*2=1.0, .9*1=.9 -> max-accumulated .9 -> 1.0
    assert out == pytest.approx([1.0, 0.004, 1.0, 0.12])
    assert all(0 <= p <= 1 for p in out)


def test_holm_monotone_and_not_below_raw():
    raw = [0.2, 0.01, 0.03, 0.03, 0.5]
    adj = evaluate.holm(raw)
    assert all(a >= r - 1e-15 for a, r in zip(adj, raw))
    order = np.argsort(raw, kind="stable")
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-15 for i in range(len(raw) - 1))


def test_holm_edge_cases():
    assert evaluate.holm([]) == []
    assert evaluate.holm([0.2]) == [0.2]
    with pytest.raises(ValueError):
        evaluate.holm([0.1, float("nan")])
    with pytest.raises(ValueError):
        evaluate.holm([1.2])


# ---- subgroup table ---------------------------------------------------------------------------
def _pred():
    return pd.DataFrame({
        "model": ["b", "a", "b", "a", "a", "c"],
        "clock": [1, 1, 2, 2, 1, 1],
        "y_true": [10.0, 20.0, 40.0, 20.0, 10.0, 5.0],
        "y_pred": [11.0, 20.0, 36.0, 30.0, 10.0, 5.0],
    })


def test_subgroup_table_rows_sorted_and_metrics():
    t = evaluate.subgroup_table(_pred(), ["model"])
    assert list(t["model"]) == ["a", "b", "c"]
    assert list(t["n"]) == [3, 2, 1]
    a = t.set_index("model").loc["a"]
    assert a["mape"] == pytest.approx((0 + 0.5 + 0) / 3)
    assert a["mae"] == pytest.approx(10 / 3)
    assert "coverage90" not in t.columns
    assert list(t.columns[:1]) == ["model"]
    assert {"mdape", "mape", "mae", "rmse", "bias_pct", "mdale"} <= set(t.columns)


def test_subgroup_table_multi_key_sorted_and_custom_columns():
    p = _pred().rename(columns={"y_true": "E", "y_pred": "Ehat"})
    t = evaluate.subgroup_table(p, ["model", "clock"], y="E", yhat="Ehat")
    assert list(zip(t["model"], t["clock"])) == [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)]
    assert t.index.tolist() == list(range(len(t)))


def test_subgroup_table_coverage_when_both_bounds_present():
    p = _pred()
    p["lo90"] = p["y_pred"] * 0.95
    p["hi90"] = p["y_pred"] * 1.05
    t = evaluate.subgroup_table(p, ["model"]).set_index("model")
    assert t.loc["a", "coverage90"] == pytest.approx(2 / 3)   # 20 vs 30 misses
    assert t.loc["c", "coverage90"] == pytest.approx(1.0)
    only_lo = p.drop(columns="hi90")
    assert "coverage90" not in evaluate.subgroup_table(only_lo, ["model"]).columns


def test_subgroup_table_accepts_string_by():
    t = evaluate.subgroup_table(_pred(), "model")
    assert list(t["model"]) == ["a", "b", "c"]
