"""Exploration report, trace level (spec section 5, items 6-8): computations on the 100 ms windows and the 5 s
PMIC windows. Descriptive only: no predictor is fitted for use anywhere; the OLS lines are descriptions of
the data they are fitted on. Writing (markdown, figures) lives in eda_trace_write.py."""
import numpy as np
import pandas as pd
from scipy import stats

import data
import settings
import tables

NAN = float("nan")
FEATURES = ["busy", "cycles_rate", "instr_rate", "ipc", "cache_miss_rate", "branch_mpki", "stall_frac", "llc_mpki",
            "ctx_sw_rate", "pf_rate", "freq_mhz", "temp_C"]
RATIO_FEATURES = ["ipc", "cache_miss_rate", "branch_mpki", "stall_frac", "llc_mpki"]   # undefined when not counted
CORR_PHASES = {"forward": ["forward"], "idle": ["baseline_A", "baseline_B"], "serve": ["serve"], "gap": ["gap"]}
SOAK_CELLS = ("yolo11n_soak30_640_t4_2400_r00", "yolo11m_soak30_640_t4_2400_r00")
WINDOW_COLS = ["cell_id", "w_idx", "t_start_ns", "P_W", "phase", "counted", *FEATURES]
ACF_FAMILIES = ["A", "B", "C", "D", "soak"]
LOW_RANGE_C = 2.0          # a configuration whose forward windows span less than this does not identify a slope


# ------------------------------------------------------------------ OLS and temperature slopes

def ols(x, y) -> dict:
    """Simple least squares y = a + b x with the textbook standard error of b (n - 2 df)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[m], y[m], int(m.sum())
    out = dict(slope=NAN, intercept=NAN, se=NAN, r2=NAN, resid_sd=NAN, n=n)
    sxx = float(((x - x.mean()) ** 2).sum()) if n else 0.0
    if n < 3 or sxx <= 0:
        return out
    b = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    a = float(y.mean() - b * x.mean())
    rss = float(((y - a - b * x) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    out.update(slope=b, intercept=a, se=float(np.sqrt(rss / (n - 2) / sxx)), r2=1 - rss / tss if tss > 0 else NAN,
               resid_sd=float(np.sqrt(rss / (n - 2))))
    return out


def within_slope(x, y, g) -> dict:
    """Slope after subtracting each group's mean of x and y (fixed-effects slope); SE uses n - groups - 1 df."""
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "g": np.asarray(g)}).dropna()
    d = d[d.groupby("g")["x"].transform("size") >= 2]
    k, n = d["g"].nunique(), len(d)
    out = dict(slope=NAN, se=NAN, n=n, k=k)
    if n - k - 1 < 1:
        return out
    dx, dy = d["x"] - d.groupby("g")["x"].transform("mean"), d["y"] - d.groupby("g")["y"].transform("mean")
    sxx = float((dx ** 2).sum())
    if sxx <= 0:
        return out
    b = float((dx * dy).sum() / sxx)
    rss = float(((dy - b * dx) ** 2).sum())
    out.update(slope=b, se=float(np.sqrt(rss / (n - k - 1) / sxx)))
    return out


def thermal_key(row) -> str:
    """Configuration for the thermal analysis: the Mode A key plus governor, fan mode and the special-run tag."""
    tag = [f"{k}={row[k]}" for k in ("soak_minutes", "serve_fps", "concurrent_mode") if pd.notna(row[k])]
    if row["input_source"] != "repeat":
        tag.append(f"input={row['input_source']}")
    return "|".join([tables.config_key(row), str(row["governor"]), str(row["fan_mode"]), *tag])


def temp_slopes(w: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Per configuration, OLS of window P_W on temp_C over forward-phase windows. `pooled` pools all the
    configuration's cells on the raw values; `within` subtracts each cell's means first. Standard errors treat
    windows as independent (they are not: 100 ms windows are strongly autocorrelated), so they are lower bounds.
    `info` has one row per cell: cell_id, family, config (key). Columns: config, family, n_cells, n_windows,
    slope_pooled, se_pooled, r2_pooled, slope_within, se_within, temp_min, temp_max, temp_range, low_range."""
    f = w[w["phase"] == "forward"].merge(info[["cell_id", "family", "config"]], on="cell_id")
    f = f[np.isfinite(f["temp_C"]) & np.isfinite(f["P_W"])]
    rows = []
    for (cfg, fam), g in f.groupby(["config", "family"], sort=True):
        p = ols(g["temp_C"], g["P_W"])
        wi = within_slope(g["temp_C"], g["P_W"], g["cell_id"])
        rng = float(g["temp_C"].max() - g["temp_C"].min())
        rows.append(dict(config=cfg, family=fam, n_cells=g["cell_id"].nunique(), n_windows=len(g),
                         slope_pooled=p["slope"], se_pooled=p["se"], r2_pooled=p["r2"], slope_within=wi["slope"],
                         se_within=wi["se"], temp_min=float(g["temp_C"].min()), temp_max=float(g["temp_C"].max()),
                         temp_range=rng, low_range=rng < LOW_RANGE_C))
    return pd.DataFrame(rows)


def slope_family_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """Per family: configurations, cells, windows, and the distribution (median, min, max) of the pooled and
    within-cell slopes over the configurations with a temperature range of at least LOW_RANGE_C."""
    rows = []
    for fam, g in ts.groupby("family", sort=True):
        ok = g[~g["low_range"] & g["slope_pooled"].notna()]
        rows.append(dict(family=fam, n_configs=len(g), n_used=len(ok), n_cells=int(g["n_cells"].sum()),
                         n_windows=int(g["n_windows"].sum()),
                         pooled_median=ok["slope_pooled"].median(), pooled_min=ok["slope_pooled"].min(),
                         pooled_max=ok["slope_pooled"].max(), within_median=ok["slope_within"].median(),
                         within_min=ok["slope_within"].min(), within_max=ok["slope_within"].max(),
                         share_positive_pct=100 * (ok["slope_pooled"] > 0).mean() if len(ok) else NAN))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ soak

def soak_series(w: pd.DataFrame, cell_id: str, smooth_s: float = 10.0) -> pd.DataFrame:
    """Forward windows of a cell: t_s since the start of the forward phase, P_W, temp_C and their centred rolling
    means over smooth_s."""
    f = w[(w["cell_id"] == cell_id) & (w["phase"] == "forward")].sort_values("t_start_ns")
    t = (f["t_start_ns"].to_numpy() - f["t_start_ns"].iloc[0]) / 1e9 if len(f) else np.array([])
    n = max(int(round(smooth_s / np.median(np.diff(t)))), 1) if len(t) > 1 else 1
    s = pd.DataFrame({"t_s": t, "P_W": f["P_W"].to_numpy(), "temp_C": f["temp_C"].to_numpy()})
    s["P_smooth"] = s["P_W"].rolling(n, center=True, min_periods=1).mean()
    s["T_smooth"] = s["temp_C"].rolling(n, center=True, min_periods=1).mean()
    return s


def time_to_fraction(t, p, frac: float = 0.95, final_s: float = 300.0, smooth_s: float = 10.0) -> dict:
    """First time (s, from t[0]) at which the centred rolling mean of p reaches frac * the mean of p over the last
    final_s seconds. NaN if it never does (for example a series that falls from above)."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    out = dict(final_mean=NAN, threshold=NAN, t_frac_s=NAN)
    if len(t) < 3:
        return out
    n = max(int(round(smooth_s / np.median(np.diff(t)))), 1)
    sm = pd.Series(p).rolling(n, center=True, min_periods=1).mean().to_numpy()
    fm = float(p[t >= t[-1] - final_s].mean())
    hit = np.flatnonzero(sm >= frac * fm)
    out.update(final_mean=fm, threshold=frac * fm, t_frac_s=float(t[hit[0]] - t[0]) if len(hit) else NAN)
    return out


def soak_summary(s: pd.DataFrame, first_s: float = 60.0, final_s: float = 300.0) -> dict:
    """Time to 95 % of the final-5-minute mean power, cold (first minute) vs hot (last 5 minutes) power and
    temperature. One cell per model: n = 1."""
    t = s["t_s"].to_numpy()
    cold, hot = s[s["t_s"] < first_s], s[s["t_s"] >= t[-1] - final_s]
    r = time_to_fraction(t, s["P_W"].to_numpy(), 0.95, final_s)
    return dict(n_windows=len(s), duration_s=float(t[-1] - t[0]), t95_s=r["t_frac_s"], final_P_W=r["final_mean"],
                cold_P_W=float(cold["P_W"].mean()), hot_P_W=float(hot["P_W"].mean()),
                delta_P_W=float(hot["P_W"].mean() - cold["P_W"].mean()),
                delta_pct=100 * float(hot["P_W"].mean() / cold["P_W"].mean() - 1),
                cold_T_C=float(cold["temp_C"].mean()), hot_T_C=float(hot["temp_C"].mean()),
                n_cold=len(cold), n_hot=len(hot))


# ------------------------------------------------------------------ cell order

def started_map(root=None) -> pd.Series:
    """cell_id -> `started` timestamp (dataset_index.csv of every source)."""
    root = root or settings.data_root()
    parts = []
    for src in data.SOURCES:
        p = root / src / "dataset_index.csv"
        if p.is_file():
            parts.append(pd.read_csv(p, dtype={"cell_id": str}, usecols=["cell_id", "started"]))
    if not parts:
        return pd.Series(dtype="datetime64[ns]")
    d = pd.concat(parts).drop_duplicates("cell_id", keep="last")
    return pd.Series(pd.to_datetime(d["started"]).to_numpy(), index=d["cell_id"].to_numpy())


def order_residuals(cells: pd.DataFrame, started: pd.Series) -> pd.DataFrame:
    """Relative residual of each cell's energy per image from its configuration mean, for configurations with at
    least two cells, scaled by sqrt(n / (n - 1)) so a residual from a mean that includes the cell has unit-ratio
    variance; plus the cell's start order within its collection day (rank, and rank / (cells that day)).
    `cells` has cell_id, config_key, energy_mJ_per_image."""
    c = cells[["cell_id", "config_key", "energy_mJ_per_image"]].copy()
    c["started"] = c["cell_id"].map(started)
    c = c.dropna(subset=["started", "energy_mJ_per_image"])
    g = c.groupby("config_key")["energy_mJ_per_image"]
    n = g.transform("size")
    c = c[n >= 2].copy()
    n = n[c.index]
    c["resid"] = (c["energy_mJ_per_image"] / g.transform("mean")[c.index] - 1) * np.sqrt(n / (n - 1))
    c["day"] = c["started"].dt.date.astype(str)
    c["order"] = c.groupby("day")["started"].rank(method="first")
    c["order_frac"] = c["order"] / c.groupby("day")["order"].transform("max")
    return c.reset_index(drop=True)


def order_spearman(c: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho and p of residual against start order, per day and pooled (order as a within-day fraction)."""
    rows = []
    for day, g in list(c.groupby("day")) + [("pooled", c)]:
        if len(g) >= 3:
            r = stats.spearmanr(g["order_frac"], g["resid"])
            rows.append(dict(day=day, n_cells=len(g), n_configs=g["config_key"].nunique(), rho=float(r.statistic),
                             p=float(r.pvalue)))
    return pd.DataFrame(rows, columns=["day", "n_cells", "n_configs", "rho", "p"])


# ------------------------------------------------------------------ counter correlations

def demeaned(df: pd.DataFrame, cols, by) -> pd.DataFrame:
    """cols with each group's mean subtracted; groups with a single row are dropped."""
    d = df[df.groupby(by)[cols[0]].transform("size") >= 2]
    return d[cols] - d.groupby(by)[cols].transform("mean")


def corr_pair(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = dict(spearman=NAN, pearson=NAN, n=int(m.sum()))
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        out.update(spearman=float(stats.spearmanr(x, y).statistic), pearson=float(np.corrcoef(x, y)[0, 1]))
    return out


def counter_correlations(w: pd.DataFrame, features=FEATURES, phases=CORR_PHASES, target: str = "P_W") -> pd.DataFrame:
    """Spearman and Pearson of window power with each feature per phase group and pooled over the groups, raw and
    within-cell demeaned (each (cell, phase) group's mean of both variables removed). Ratio features use counted
    windows only (they are undefined otherwise). Columns: phase, feature, n_cells, n_raw, spearman_raw, pearson_raw,
    n_dm, spearman_dm, pearson_dm."""
    rows = []
    lab = {p: g for g, ps_ in phases.items() for p in ps_}
    d = w[w["phase"].isin(lab)].assign(group=lambda x: x["phase"].map(lab))
    for grp, g in [(k, d[d["group"] == k]) for k in phases] + [("pooled", d)]:
        for f in features:
            h = g[g["counted"] == 1] if f in RATIO_FEATURES else g
            h = h[np.isfinite(h[f]) & np.isfinite(h[target])].assign(grp_key=lambda x: x["cell_id"] + "/" + x["phase"])
            raw = corr_pair(h[f], h[target])
            dm = demeaned(h, [f, target], "grp_key")
            dmc = corr_pair(dm[f], dm[target])
            rows.append(dict(phase=grp, feature=f, n_cells=h["cell_id"].nunique(), n_raw=raw["n"],
                             spearman_raw=raw["spearman"], pearson_raw=raw["pearson"], n_dm=dmc["n"],
                             spearman_dm=dmc["spearman"], pearson_dm=dmc["pearson"]))
    return pd.DataFrame(rows)


def perf_overhead(cells: pd.DataFrame) -> pd.DataFrame:
    """Forward power of `ref` cells (perf counters off) against the same plain configuration in the other v2
    families (counters on), per clock; descriptive, family and run order are confounded with the counters switch."""
    c = cells[(cells["rig"] == "nov2026") & cells["serve_fps"].isna() & cells["soak_minutes"].isna()
              & cells["concurrent_mode"].isna() & (cells["input_source"] == "repeat") & (cells["precision"] == "fp32")
              & (cells["variant"] == cells["base_model"]) & (cells["batch"] == 1)].copy()
    c["counters"] = np.where(c["family"] == "ref", "off", "on")
    k = ["base_model", "imgsz", "threads", "cpu_mhz"]
    g = c.groupby(k + ["counters"])["mean_P_W"].agg(["mean", "size"]).unstack("counters")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.dropna().reset_index()
    g["delta_W"] = g["mean_off"] - g["mean_on"]
    g["delta_pct"] = 100 * g["delta_W"] / g["mean_on"]
    return g.rename(columns={"mean_off": "P_off_W", "mean_on": "P_on_W", "size_off": "n_off", "size_on": "n_on"})


# ------------------------------------------------------------------ PMIC against Monsoon

def pmic_family(info: pd.DataFrame) -> pd.Series:
    """Group label per cell: v1 by workload task, v2 by family."""
    return pd.Series(np.where(info["source"] == "v1", "v1 " + info["task"], "v2 " + info["family"].astype(str)),
                     index=info["cell_id"].to_numpy())


def pmic_fits(pw: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Per group, OLS P_W = a + b pmic_W over the 5 s PMIC windows (all phases), plus the forward-phase R2.
    Columns: group, n_cells, n_windows, a, b, se_b, r2, resid_sd_W, mean_P_W, mean_pmic_W, n_forward, r2_forward."""
    d = pw.assign(group=pw["cell_id"].map(group)).dropna(subset=["group", "pmic_W", "P_W"])
    rows = []
    for gname, g in d.groupby("group", sort=True):
        f, fw = ols(g["pmic_W"], g["P_W"]), g[g["phase"] == "forward"]
        rows.append(dict(group=gname, n_cells=g["cell_id"].nunique(), n_windows=f["n"], a=f["intercept"], b=f["slope"],
                         se_b=f["se"], r2=f["r2"], resid_sd_W=f["resid_sd"], mean_P_W=float(g["P_W"].mean()),
                         mean_pmic_W=float(g["pmic_W"].mean()), n_forward=len(fw),
                         r2_forward=ols(fw["pmic_W"], fw["P_W"])["r2"] if len(fw) >= 3 else NAN))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ autocorrelation

def acf(x, nlags: int) -> np.ndarray:
    """Sample autocorrelation at lags 0..nlags (mean removed, divisor n, so acf[0] = 1)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    nlags = min(nlags, n - 1)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    return c / c[0] if c[0] > 0 else np.full(nlags + 1, NAN)


def first_lag_below(a: np.ndarray, thr: float = 0.1) -> int | None:
    """First lag >= 1 with acf < thr, or None if not reached."""
    hit = np.flatnonzero(np.asarray(a)[1:] < thr)
    return int(hit[0] + 1) if len(hit) else None


def acf_cells(info: pd.DataFrame, avail: set) -> list[str]:
    """One cell per family A, B, C, D, soak: the r00 cell of a yolo11n configuration (first by id), else the first
    cell of the family that has windows."""
    out = []
    for fam in ACF_FAMILIES:
        g = info[(info["family"] == fam) & info["cell_id"].isin(avail)].sort_values("cell_id")
        pref = g[(g["base_model"] == "yolo11n") & (g["repeat"] == 0)]
        pick = pref if len(pref) else g
        if len(pick):
            out.append(pick["cell_id"].iloc[0])
    return out


def acf_summary(w: pd.DataFrame, cell_id: str, phase: str = "forward", nlags: int = 600, thr: float = 0.1) -> dict:
    """ACF of 100 ms window power of one cell's phase windows (ordered by time) and its first lag below thr."""
    f = w[(w["cell_id"] == cell_id) & (w["phase"] == phase)].sort_values("t_start_ns")
    dt = float(np.median(np.diff(f["t_start_ns"])) / 1e9) if len(f) > 1 else NAN
    a = acf(f["P_W"].to_numpy(), nlags) if len(f) > 3 else np.array([1.0])
    lag = first_lag_below(a, thr)
    above = np.flatnonzero(a[1:] >= thr)
    return dict(cell_id=cell_id, phase=phase, n_windows=len(f), dt_s=dt, max_lag=len(a) - 1, lag_windows=lag,
                lag_s=lag * dt if lag is not None else NAN, acf1=float(a[1]) if len(a) > 1 else NAN,
                last_above=int(above[-1] + 1) if len(above) else 0, acf=a)
