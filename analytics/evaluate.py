"""Evaluation utilities that score every predictor: error metrics, cluster bootstrap, paired
Wilcoxon, Holm correction, interval coverage and subgroup tables. Pure functions, no I/O.
Percent quantities are fractions (0.05 = 5 %)."""
import math

import numpy as np
import pandas as pd
from scipy import stats

METRIC_KEYS = ["n", "mdape", "mape", "mae", "rmse", "bias_pct", "mdale", "maxape", "p90ape", "r2_log"]


def _pair(a, b, what="arrays"):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"{what} differ in length: {a.size} vs {b.size}")
    return a, b


def metrics(y, yhat) -> dict:
    """Error metrics of predictions `yhat` against positive truth `y`.

    Pairs with a non-finite value on either side are dropped and `n` is the number used. Raises
    ValueError if any remaining `y <= 0`. `mae`/`rmse` are in the target's units; `mdape`, `mape`,
    `bias_pct` (median of signed `(yhat - y) / y`) are fractions; `mdale` is the median of
    `|log(yhat / y)|` and is NaN when any retained prediction is <= 0 (log undefined). `maxape` and
    `p90ape` are the maximum and the 90th percentile (linear interpolation) of `|pct|`; `r2_log` is
    `1 - SSE/SST` of `log y` against `log yhat` (NaN when a prediction is <= 0 or `log y` is constant)."""
    y, yhat = _pair(y, yhat, "y and yhat")
    keep = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[keep], yhat[keep]
    if y.size == 0:
        return {"n": 0, **{k: float("nan") for k in METRIC_KEYS[1:]}}
    if (y <= 0).any():
        raise ValueError("percentage errors need positive truth: y contains values <= 0")
    err = yhat - y
    pct = err / y
    if (yhat <= 0).any():
        mdale = r2_log = float("nan")
    else:
        mdale = float(np.median(np.abs(np.log(yhat / y))))
        sst = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
        r2_log = float(1.0 - np.sum((np.log(y) - np.log(yhat)) ** 2) / sst) if sst > 0 else float("nan")
    return {"n": int(y.size),
            "mdape": float(np.median(np.abs(pct))),
            "mape": float(np.mean(np.abs(pct))),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(math.sqrt(np.mean(err ** 2))),
            "bias_pct": float(np.median(pct)),
            "mdale": mdale,
            "maxape": float(np.max(np.abs(pct))),
            "p90ape": float(np.percentile(np.abs(pct), 90)),
            "r2_log": r2_log}


def coverage(y, lo, hi) -> float:
    """Fraction of `y` inside the closed interval [lo, hi]; rows with a non-finite value are
    dropped; NaN when none remain."""
    y, lo = _pair(y, lo, "y and lo")
    y, hi = _pair(y, hi, "y and hi")
    keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return float("nan")
    return float(np.mean((y[keep] >= lo[keep]) & (y[keep] <= hi[keep])))


def interval_width(lo, hi) -> float:
    """Median of hi / lo over the rows where both bounds are finite and positive (NaN if none): the
    typical multiplicative width of a prediction interval."""
    lo, hi = _pair(lo, hi, "lo and hi")
    keep = np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
    return float(np.median(hi[keep] / lo[keep])) if keep.any() else float("nan")


def cluster_bootstrap(df: pd.DataFrame, group_col: str, stat, n: int = 2000, *, seed: int):
    """Percentile (2.5, 97.5) bootstrap interval of `stat(df)` resampling whole groups.

    Each of the `n` resamples draws as many groups as there are distinct groups, with replacement
    (`numpy.random.default_rng(seed)`), and concatenates their rows; a group drawn twice
    contributes twice. `stat` takes a DataFrame and returns a float. One group gives
    `(point, point)`. Non-finite resampled statistics are ignored."""
    if len(df) == 0:
        raise ValueError("cluster_bootstrap needs at least one row")
    if df[group_col].isna().any():
        raise ValueError(f"group column {group_col!r} contains nulls")
    df = df.reset_index(drop=True)
    members = [np.asarray(ix) for _, ix in sorted(df.groupby(group_col).indices.items())]
    k = len(members)
    if k == 1:
        point = float(stat(df))
        return point, point
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, size=k)
        vals[i] = stat(df.iloc[np.concatenate([members[j] for j in pick])])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_wilcoxon(err_a, err_b) -> dict:
    """Two-sided paired Wilcoxon signed-rank test of `err_a - err_b` (`zero_method="wilcox"`).

    Non-finite pairs are dropped. `median_diff` is the median of the differences (positive means
    A has the larger error). With fewer than 2 non-zero differences the test is undefined: p = 1.0
    and stat NaN."""
    a, b = _pair(err_a, err_b, "err_a and err_b")
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    median_diff = float(np.median(d)) if d.size else float("nan")
    if np.count_nonzero(d) < 2:
        return {"stat": float("nan"), "p": 1.0, "median_diff": median_diff}
    res = stats.wilcoxon(d, zero_method="wilcox")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "median_diff": median_diff}


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order (monotone, capped at 1)."""
    p = np.asarray(list(pvals), dtype=float)
    if p.size == 0:
        return []
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be finite and within [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return [float(v) for v in out]


def subgroup_table(pred: pd.DataFrame, by, y: str = "y_true", yhat: str = "y_pred") -> pd.DataFrame:
    """One row per group of `by`: the `by` columns, the `metrics` keys and, when both `lo90` and
    `hi90` exist, `coverage90`. Sorted by `by`."""
    by = [by] if isinstance(by, str) else list(by)
    for col in by:
        if pred[col].isna().any():
            raise ValueError(f"group column {col!r} contains nulls")
    with_cov = "lo90" in pred.columns and "hi90" in pred.columns
    rows = []
    for key, g in pred.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(metrics(g[y], g[yhat]))
        if with_cov:
            row["coverage90"] = coverage(g[y], g["lo90"], g["hi90"])
        rows.append(row)
    cols = by + METRIC_KEYS + (["coverage90"] if with_cov else [])
    return pd.DataFrame(rows, columns=cols)
