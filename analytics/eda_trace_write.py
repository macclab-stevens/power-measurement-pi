"""Markdown and figures of the trace-level exploration report (sections 6-8; eda_trace.py holds the computations).
write_eda_trace() returns the text that eda.splice_trace() puts into eda.md."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import deploy
import eda
import eda_trace as et
import eda_write as ew
import plotstyle as ps
import settings
import windows


def trace_info(cat: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """QC-primary catalogue cells with their thermal configuration key."""
    c = deploy._primary(cat, qc).copy()
    c["config"] = [et.thermal_key(r) for _, r in c.iterrows()]
    return c


def load_windows(cat: pd.DataFrame, cell_ids) -> pd.DataFrame:
    """Column-projected 100 ms windows of the given cells (cache rebuilt only if missing or stale)."""
    path = settings.derived_dir() / windows.WINDOWS_CACHE
    if not path.is_file():
        windows.build_windows(cat)
    cols = [c for c in et.WINDOW_COLS if c in pq.read_schema(path).names]
    w = pq.read_table(path, columns=cols).to_pandas()
    return w[w["cell_id"].isin(set(cell_ids))].reset_index(drop=True)


def load_pmic(cat: pd.DataFrame, cell_ids) -> pd.DataFrame:
    path = settings.derived_dir() / windows.PMIC_CACHE
    if not path.is_file():
        windows.build_pmic_windows(cat)
    p = pq.read_table(path, columns=["cell_id", "phase", "P_W", "pmic_W"]).to_pandas()
    return p[p["cell_id"].isin(set(cell_ids))].reset_index(drop=True)


# ------------------------------------------------------------------ figures

def fig_temp_slopes(ts, figdir):
    d = ts[~ts["low_range"] & ts["slope_pooled"].notna()]
    fams = sorted(d["family"].unique())
    fig, ax = ps.subplots(size=(6.4, 3.4))
    for i, f in enumerate(fams):
        g = d[d["family"] == f]
        x = np.full(len(g), i, float)
        ax.scatter(x - 0.12, g["slope_pooled"], s=16, color=ps.COLORS[0], alpha=0.7, label="pooled" if i == 0 else None)
        ax.scatter(x + 0.12, g["slope_within"], s=16, color=ps.COLORS[1], alpha=0.7, marker="s",
                   label="within-cell" if i == 0 else None)
    ax.axhline(0, color=ps.MUTED, lw=0.8)
    ax.set_xticks(range(len(fams)), [f"{f} ({int((d['family'] == f).sum())})" for f in fams])
    ax.set(xlabel="family (configurations)", ylabel="dP/dT, forward windows (W/degC)")
    ax.legend()
    return ps.save(fig, "eda_temp_slopes", figdir)


def fig_soak(series: dict, figdir):
    fig, axes = ps.subplots(2, len(series), size=(4.0 * len(series), 5.0), squeeze=False, sharex="col")
    for j, (cid, s) in enumerate(series.items()):
        t = s["t_s"] / 60
        axes[0, j].plot(t, s["P_W"], color=ps.COLORS[0], alpha=0.25, lw=0.5, label="100 ms windows")
        axes[0, j].plot(t, s["P_smooth"], color=ps.COLORS[0], label="10 s mean")
        axes[1, j].plot(t, s["T_smooth"], color=ps.COLORS[1], label="10 s mean")
        axes[0, j].set(title=f"{cid} (n = 1)", ylabel="power (W)")
        axes[1, j].set(xlabel="minutes since start of forward phase", ylabel="SoC temperature (degC)")
        axes[0, j].legend(loc="lower right")
    return ps.save(fig, "eda_soak", figdir)


def fig_pmic(pw, grp, fits, figdir):
    names = list(fits["group"])
    nc = 5
    fig, axes = ps.subplots((len(names) + nc - 1) // nc, nc, size=(3.0 * nc, 2.8 * ((len(names) + nc - 1) // nc)), squeeze=False)
    d = pw.assign(group=pw["cell_id"].map(grp))
    for ax, n in zip(axes.ravel(), names):
        g, f = d[d["group"] == n], fits[fits["group"] == n].iloc[0]
        ax.scatter(g["pmic_W"], g["P_W"], s=3, alpha=0.3, color=ps.COLORS[0], rasterized=True)
        x = np.array([g["pmic_W"].min(), g["pmic_W"].max()])
        ax.plot(x, f["a"] + f["b"] * x, color=ps.COLORS[1], lw=1.2)
        ax.set_title(f"{n}  R2 {f['r2']:.2f}, n {int(f['n_windows'])}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.supxlabel("PMIC rail-sum power (W)")
    fig.supylabel("Monsoon power over the 5 s window (W)")
    return ps.save(fig, "eda_pmic_scatter", figdir)


def fig_acf(res, figdir):
    fig, ax = ps.subplots(size=(6.4, 3.6))
    for i, r in enumerate(res):
        a = r["acf"]
        ax.plot(np.arange(len(a)) * r["dt_s"], a, color=ps.COLORS[i % len(ps.COLORS)], lw=1.0, label=r["cell_id"])
    ax.axhline(0.1, color=ps.MUTED, ls="--", lw=0.8)
    ax.set(xlabel="lag (s)", ylabel="autocorrelation of 100 ms window power", xlim=(0, 30))
    ax.legend(fontsize=6)
    return ps.save(fig, "eda_acf", figdir)


# ------------------------------------------------------------------ report

def _fmt_corr(df, phases=("forward", "idle", "serve", "gap", "pooled")):
    t = df.rename(columns={"phase": "phase", "n_cells": "n cells", "n_raw": "n raw", "spearman_raw": "Spearman raw",
                           "pearson_raw": "Pearson raw", "n_dm": "n dm", "spearman_dm": "Spearman dm",
                           "pearson_dm": "Pearson dm"})
    return ew.md_table(t, {c: "{:+.3f}" for c in t.columns if "Spearman" in c or "Pearson" in c})


def write_eda_trace(cat, der, qc, report_dir, w=None, pw=None) -> str:
    """Render sections 6-8 (figures under report_dir/fig); returns the markdown. `w` / `pw` can be passed in
    (tests); otherwise the cached windows are loaded with column projection."""
    figdir = Path(report_dir) / "fig"
    figdir.mkdir(parents=True, exist_ok=True)
    info = trace_info(cat, qc)
    ctr = info[info["has_counters"]]
    w = load_windows(cat, ctr["cell_id"]) if w is None else w
    pw = load_pmic(cat, info["cell_id"]) if pw is None else pw
    wc = ctr[ctr["cell_id"].isin(set(w["cell_id"]))]
    ver = "dataset v2 (nov2026 rig; v1 has no counters or sysfs)"
    L = []

    # 6 ------------------------------------------------------------ thermal
    ts = et.temp_slopes(w, info)
    fs = et.slope_family_summary(ts)
    fw = w[w["phase"] == "forward"]
    L += ["## 6. Thermal behaviour and run order\n", "### 6.1 Power against temperature in forward windows\n",
          "Per configuration (Mode A key plus governor, fan mode and the soak / serve / concurrent / input tag), "
          "ordinary least squares of the 100 ms window power `P_W` on `temp_C` (1 Hz sysfs temperature, latest reading at "
          "the window end) over its forward-phase windows. `pooled` uses all the configuration's cells on raw values; "
          "`within` subtracts each cell's means first. The standard errors treat windows as independent; consecutive "
          "100 ms windows are strongly autocorrelated (section 7.3), so they are lower bounds. Temperature is confounded "
          "with elapsed time and with the clock-pinned load, so a slope is a description, not a causal estimate. "
          f"Configurations whose forward windows span less than {et.LOW_RANGE_C:g} degC do not identify a slope and are "
          f"excluded from the family summaries (still listed in the detail table below). {ew.prov(wc)}; "
          f"{ew.plural(len(ts), 'configuration')}, {len(fw)} forward windows; {ver}.\n",
          ew.md_table(fs.rename(columns={"n_configs": "configs", "n_used": "used", "n_cells": "cells", "n_windows": "windows",
                                         "pooled_median": "pooled med", "pooled_min": "pooled min", "pooled_max": "pooled max",
                                         "within_median": "within med", "within_min": "within min", "within_max": "within max",
                                         "share_positive_pct": "% pooled > 0"}),
                      {c: "{:+.3f}" for c in ("pooled med", "pooled min", "pooled max", "within med", "within min", "within max")}
                      | {"% pooled > 0": "{:.0f}"})]
    ok = ts[~ts["low_range"]]
    if len(ok):
        L.append(f"Slopes over the {len(ok)} identified configurations (W/degC): pooled {ok['slope_pooled'].min():+.3f} to "
                 f"{ok['slope_pooled'].max():+.3f} (median {ok['slope_pooled'].median():+.3f}), within-cell "
                 f"{ok['slope_within'].min():+.3f} to {ok['slope_within'].max():+.3f} (median {ok['slope_within'].median():+.3f}).\n")
        fig_temp_slopes(ts, figdir)
        L.append("![Power-temperature slope per configuration](fig/eda_temp_slopes.png)\n")
    L += ["Per configuration (slope W/degC, standard error, forward windows, cells, temperature range degC):\n",
          ew.md_table(ts[["family", "config", "n_cells", "n_windows", "slope_pooled", "se_pooled", "slope_within", "se_within",
                          "temp_range"]].assign(config=lambda d: d["config"].str.replace("|nov2026|", "|", regex=False)),
                      {"slope_pooled": "{:+.3f}", "se_pooled": "{:.3f}", "slope_within": "{:+.3f}", "se_within": "{:.3f}",
                       "temp_range": "{:.1f}"})]

    # 6.2 soak
    series = {c: et.soak_series(w, c) for c in et.SOAK_CELLS if c in set(w["cell_id"])}
    L += ["### 6.2 Soak trajectories\n",
          "The two 30-minute soak cells (yolo11n and yolo11m, 640 px, 4 threads, 2400 MHz, fan full): one cell per model, "
          "**n = 1 each, so nothing here carries a confidence interval**. Forward-phase windows, time from the start of "
          "the forward phase, 10 s centred rolling mean. Time to 95 % is the first time the rolling-mean power reaches 95 % "
          "of the mean power over the last 5 minutes (the rolling mean is centred; a trailing 10 s mean, which lags, gives "
          "27.7 s for yolo11n and 13.4 s for yolo11m). Cold = first minute, hot = last 5 minutes. "
          f"{ew.prov(wc[wc['cell_id'].isin(series)])}; {ver}.\n"]
    rows = [{"cell": c, **et.soak_summary(s)} for c, s in series.items()]
    if rows:
        t = pd.DataFrame(rows)
        L.append(ew.md_table(t[["cell", "n_windows", "duration_s", "t95_s", "final_P_W", "cold_P_W", "hot_P_W", "delta_P_W",
                                "delta_pct", "cold_T_C", "hot_T_C"]],
                             {"duration_s": "{:.0f}", "t95_s": "{:.0f}", "final_P_W": "{:.3f}", "cold_P_W": "{:.3f}",
                              "hot_P_W": "{:.3f}", "delta_P_W": "{:+.3f}", "delta_pct": "{:+.2f}", "cold_T_C": "{:.1f}",
                              "hot_T_C": "{:.1f}"}))
        fig_soak(series, figdir)
        L.append("![Soak power and temperature](fig/eda_soak.png)\n")
    else:
        L.append("_Soak cells not present._\n")

    # 6.3 cell order
    cells = deploy.cell_table(cat, der, qc)
    cells = cells.assign(config_key=[et.thermal_key(r) for _, r in cells.iterrows()])
    od = et.order_residuals(cells, et.started_map())
    sp = et.order_spearman(od)
    L += ["### 6.3 Cell order\n",
          "Residual of each cell's energy per image from its configuration mean (relative, scaled by sqrt(n/(n-1)); "
          "configurations (the key of section 6.1) with at least two QC-primary cells) against the cell's start order within its collection day (rank / cells that day). The "
          "matrix order was shuffled, so a Spearman rho far from 0 would indicate drift (temperature, supply, "
          "clock) that survives the shuffle; the test cannot see an effect that the shuffle spread evenly. Each version "
          f"was collected on one day. {ew.prov(cells[cells['cell_id'].isin(od['cell_id'])])}; "
          f"{ew.plural(od['config_key'].nunique(), 'configuration')}.\n", ew.md_table(sp, {"rho": "{:+.3f}", "p": "{:.3g}"})]

    # 7 -------------------------------------------------------------- counters
    cc = et.counter_correlations(w)
    cs = deploy.cell_table(cat, der, qc)
    L += ["## 7. Counters and window power\n", "### 7.1 Correlations with 100 ms window power\n",
          "Spearman and Pearson correlation of `P_W` with each counter feature, per phase group (`forward`, `idle` = "
          "`baseline_A` + `baseline_B`, `serve`, `gap`) and pooled over the four. `raw` uses all windows; `dm` subtracts, "
          "for each cell and phase, the mean of both variables first (so only within-cell, within-phase co-movement "
          "remains). Ratio features (`ipc`, `cache_miss_rate`, `branch_mpki`, `stall_frac`, `llc_mpki`) use counted "
          "windows only (undefined otherwise); rate features use every window. `n raw` / `n dm` are windows. A blank is an undefined "
          "correlation: `ctx_sw_rate` is 0 in every window (the context-switch counter never ticks for the attached "
          "process), and `freq_mhz` is constant within a pinned-clock cell and phase. "
          f"Windows: {ew.prov(wc)}; {ver}. **Perf overhead caveat**: perf counters were enabled in every v2 cell except "
          "`ref`, so power measured here includes the cost of counting (section 7.2).\n"]
    for ph in ("forward", "idle", "serve", "gap", "pooled"):
        g = cc[cc["phase"] == ph]
        L += [f"**{ph}** ({int(g['n_raw'].max())} windows at most, {int(g['n_cells'].max())} cells):\n", _fmt_corr(g.drop(columns="phase"))]
    fwd = cc[(cc["phase"] == "forward")].copy()
    fwd["abs_dm"] = fwd["spearman_dm"].abs()
    top = fwd.sort_values("abs_dm", ascending=False).head(5)
    L.append("Forward phase, largest within-cell |Spearman|: " + "; ".join(
        f"`{r.feature}` raw {r.spearman_raw:+.2f} / dm {r.spearman_dm:+.2f}" for r in top.itertuples()) + ".\n")

    po = et.perf_overhead(cs)
    L += ["### 7.2 Forward power with counters off (`ref`) against counters on\n",
          "Mean forward power of plain `ref` cells (counters off) against the same configuration in the other v2 families "
          "(counters on), nov2026 rig, batch 1, fp32. Family, collection order and the counters switch are confounded, so "
          "this bounds the perf overhead only loosely. "
          f"{ew.plural(len(po), 'configuration')}, {int(po['n_off'].sum() + po['n_on'].sum()) if len(po) else 0} cells (dataset v2).\n",
          ew.md_table(po.rename(columns={"delta_W": "off_minus_on_W", "delta_pct": "off_minus_on_pct"}), {"off_minus_on_W": "{:+.3f}", "off_minus_on_pct": "{:+.2f}", "P_off_W": "{:.3f}", "P_on_W": "{:.3f}", })]

    avail = set(w.loc[w["phase"] == "forward", "cell_id"])
    ids = et.acf_cells(info, avail)
    res = [et.acf_summary(w, c) for c in ids]
    L += ["### 7.3 Autocorrelation of window power\n",
          "ACF of the 100 ms window power within the forward phase of one cell per family (the r00 cell of a yolo11n "
          "configuration where there is one), windows ordered by time, mean removed; the first lag where ACF < 0.1. "
          "The ACF of a frame-periodic signal oscillates (one frame is about 3 windows at 290 ms), so the first lag "
          "below 0.1 can be 1 even when correlation returns at the next frame period; `ACF lag 1` and `last lag >= 0.1` "
          "(the last lag, within the max lag computed, at which the ACF is still at least 0.1) are shown beside it. "
          "A soak cell's slow thermal trend is left in. One cell per family (n = 1 each); "
          f"{ver}.\n",
          ew.md_table(pd.DataFrame([{"cell": r["cell_id"], "family": info.set_index("cell_id").loc[r["cell_id"], "family"],
                                     "n windows": r["n_windows"], "max lag": r["max_lag"],
                                     "lag < 0.1 (windows)": "not reached" if r["lag_windows"] is None else r["lag_windows"],
                                     "lag < 0.1 (s)": "" if r["lag_windows"] is None else f"{r['lag_s']:.2f}",
                                     "ACF lag 1": f"{r['acf1']:+.2f}", "last lag >= 0.1": r["last_above"]} for r in res]))]
    if res:
        fig_acf(res, figdir)
        L.append("![ACF of window power](fig/eda_acf.png)\n")

    # 8 -------------------------------------------------------------- PMIC
    grp = et.pmic_family(info)
    fits = et.pmic_fits(pw, grp)
    L += ["## 8. PMIC rail-sum power against Monsoon power\n",
          "`pmic_W` is the sum of the PMIC rail powers from the `context.csv` read that was available at the window "
          "start (a 5 s snapshot, dated by availability, not an average); `P_W` is the Monsoon mean power over the 5 s "
          "window. All phases pooled within a group; v1 groups are by workload task, v2 by family. Per group an OLS "
          "`P_W = a + b * pmic_W` (a description on these windows; the calibrated PMIC model is fitted later inside "
          "folds), with `R2` and residual SD; `R2 fwd` is the forward-phase subset. "
          f"{ew.prov(info[info['cell_id'].isin(pw['cell_id'])])}; {len(pw)} windows; datasets v1 (oct2026) and v2 (nov2026).\n",
          ew.md_table(fits.rename(columns={"n_cells": "cells", "n_windows": "windows", "se_b": "se b", "resid_sd_W": "resid SD W",
                                           "mean_P_W": "mean P W", "mean_pmic_W": "mean pmic W", "n_forward": "n fwd",
                                           "r2_forward": "R2 fwd"}),
                      {"a": "{:+.3f}", "b": "{:.3f}", "se b": "{:.3f}", "r2": "{:.3f}", "R2 fwd": "{:.3f}",
                       "resid SD W": "{:.3f}", "mean P W": "{:.3f}", "mean pmic W": "{:.3f}"})]
    if len(fits):
        fig_pmic(pw, grp, fits, figdir)
        L.append("![PMIC against Monsoon power](fig/eda_pmic_scatter.png)\n")
    return "\n".join(L)
