"""Markdown and figures of the configuration-level exploration report (eda.py holds the computations)."""
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

import data
import deploy
import eda
import plotstyle as ps

STEP_ORDER = ["preprocess", "forward", "postprocess", "profiler", "dataload", "zero_grad", "loss", "backward",
              "optimizer_step", "(none)"]
YOLO = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]
SIZE_MARK = {320: "o", 480: "s", 640: "^", 960: "D"}


# ------------------------------------------------------------------ markdown helpers

def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    """GitHub table; floats default to 4 significant digits, NaN blank, per-column format via `fmt`."""
    if df is None or len(df) == 0:
        return "_No data._\n"
    fmt = fmt or {}

    def cell(c, v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.get(c, "{:.4g}").format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(c, v) for c, v in zip(df.columns, row)) + " |\n"
                          for row in df.itertuples(index=False))


def prov(df: pd.DataFrame) -> str:
    """n and dataset version of a cell-level frame (one row per cell) or a config-level one (n_cells column)."""
    if df is None or df.empty:
        return "n = 0"
    w = df["n_cells"] if "n_cells" in df else pd.Series(1, index=df.index)
    by = w.groupby(df["rig"]).sum()
    by = by.reindex(sorted(by.index, key=lambda r: (r != "oct2026", r)))
    ver = ", ".join(f"{eda.VERSION.get(r, r)} ({r} rig) {int(n)}" for r, n in by.items())
    cfg = f"{len(df)} configurations / " if "n_cells" in df else ""
    return f"n = {cfg}{int(w.sum())} cells (dataset {ver} cells)"


def cells_prov(cat, qc, task, n_rows) -> str:
    """Provenance of a table whose rows are groups, not cells (cells can feed several rows)."""
    c = deploy._primary(cat, qc)
    return f"{n_rows} groups; " + prov(c[c["task"] == task])


def pivot(df, index, columns, values, fmt="{:.4g}", order=None) -> pd.DataFrame:
    p = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    if order:
        p = p[[c for c in order if c in p.columns] + [c for c in p.columns if c not in order]]
    return p.map(lambda v: "" if pd.isna(v) else fmt.format(v)).reset_index()


def share_pivot(sh: pd.DataFrame, order, extra=False) -> pd.DataFrame:
    if sh.empty:
        return sh
    sh = sh.assign(v=[f"{100 * e:.1f} / {100 * d:.1f}" for e, d in zip(sh["E_share"], sh["dur_share"])])
    p = sh.pivot_table(index=["model_key", "rig", "n_cells"], columns="label", values="v", aggfunc="first")
    p = p[[c for c in order if c in p.columns] + [c for c in p.columns if c not in order]].fillna("-")
    if extra:
        nf = sh[sh["label"] == "forward"].set_index(["model_key", "rig", "n_cells"])["E_share"]
        p["non-forward E %"] = (100 * (1 - nf)).reindex(p.index).map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return p.reset_index()


# ------------------------------------------------------------------ figures

def fig_cv(multi, figdir):
    fig, ax = ps.subplots(size=(6.0, 3.2))
    bins = np.linspace(0, max(float(multi["energy_cv"].max() * 100), 0.5) * 1.02, 20)
    for i, (rig, g) in enumerate(multi.groupby("rig")):
        ax.hist(g["energy_cv"] * 100, bins=bins, alpha=0.75, color=ps.COLORS[i], edgecolor="white",
                label=f"{eda.VERSION.get(rig, rig)} ({rig}), {len(g)} configurations")
    ax.set(xlabel="energy per image CV across repeat cells (%)", ylabel="configurations")
    if len(multi):
        ax.legend()
    return ps.save(fig, "eda_cv_hist", figdir)


def fig_scaling(series, tbl, figdir):
    fig, ax = ps.subplots(size=(6.4, 4.0))
    fam = {"yolov8": 0, "yolo11": 1}
    xs = np.linspace(1.4, 2.5, 50)
    for s in series:
        free = tbl[(tbl["series"] == s["label"]) & (tbl["rig"] == s["rig"]) & (tbl["fit"] == "free")].iloc[0]
        col = ps.INK if s["label"] == "idle" else ps.COLORS[fam.get(s["label"][:6], 2)]
        ls = "--" if s["rig"] == "oct2026" else "-"
        ax.plot(s["f"], s["P"], "o", color=col, ms=4)
        if free["converged"]:
            ax.plot(xs, free["P_static"] + free["k"] * xs ** free["alpha"], ls, color=col, lw=1, alpha=0.8)
    h = [mlines.Line2D([], [], color=c, label=n) for c, n in
         ((ps.COLORS[0], "YOLOv8"), (ps.COLORS[1], "YOLO11"), (ps.COLORS[2], "classifiers"), (ps.INK, "idle"))]
    h += [mlines.Line2D([], [], color=ps.MUTED, ls="-", label="v2 (nov2026)"),
          mlines.Line2D([], [], color=ps.MUTED, ls="--", label="v1 (oct2026)")]
    ax.legend(handles=h, ncols=2)
    ax.set(xlabel="CPU clock (GHz)", ylabel="power (W)")
    return ps.save(fig, "eda_clock_power", figdir)


def fig_threads(grid, figdir):
    models = [m for m in grid["base_model"].unique()]
    fig, axes = ps.subplots(1, max(len(models), 1), size=(3.6 * max(len(models), 1), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        for i, (t, g) in enumerate(grid[grid["base_model"] == m].groupby("threads")):
            ax.plot(g["imgsz"], g["energy_mJ"] / 1000, marker="o", color=ps.COLORS[i], label=f"{t} thread(s)")
        ax.set(title=m, xlabel="image size (px)", ylabel="energy per image (J)")
        ax.legend()
    return ps.save(fig, "eda_threads_imgsz", figdir)


def fig_phase(sh, figdir):
    d = sh[sh["label"] == "measured"].sort_values("E_share")
    fig, ax = ps.subplots(size=(6.4, max(2.5, 0.28 * len(d) + 1.2)))
    y = np.arange(len(d))
    ax.barh(y - 0.2, 100 * d["E_share"], 0.38, color=ps.COLORS[0], label="share of cell energy")
    ax.barh(y + 0.2, 100 * d["dur_share"], 0.38, color=ps.COLORS[1], label="share of cell time")
    ax.set_yticks(y, [f"{m} ({eda.VERSION.get(r, r)})" for m, r in zip(d["model_key"], d["rig"])])
    ax.set(xlabel="measured frames, % of the whole cell (2400 MHz)")
    ax.legend(loc="lower right")
    return ps.save(fig, "eda_phase_shares", figdir)


def fig_energy_map(ea, figdir):
    fig, ax = ps.subplots(size=(6.6, 4.4))
    for r in ea.itertuples():
        c = ps.COLORS[YOLO.index(r.base_model)] if r.base_model in YOLO else ps.MUTED
        ax.scatter(r.energy_mJ / 1000, r.map50_95, color=c, marker=SIZE_MARK.get(r.imgsz, "o"), s=44, zorder=3,
                   edgecolor=ps.INK if r.pareto else "white", linewidth=1.2)
        if r.precision != "fp32":
            ax.annotate(r.precision, (r.energy_mJ / 1000, r.map50_95), textcoords="offset points", xytext=(5, -9),
                        fontsize=7, color=ps.MUTED)
    h = [mlines.Line2D([], [], marker="o", ls="", color=ps.COLORS[i], label=m) for i, m in enumerate(YOLO)]
    h += [mlines.Line2D([], [], marker=mk, ls="", color=ps.MUTED, label=f"{s} px") for s, mk in SIZE_MARK.items()]
    h.append(mlines.Line2D([], [], marker="o", ls="", mfc="white", mec=ps.INK, mew=1.2, color="white", label="Pareto (dark ring)"))
    ax.legend(handles=h, ncols=2, fontsize=7, loc="lower right")
    ax.set(xscale="log", xlabel="energy per image (J, log scale)", ylabel="mAP50-95 (COCO val2017 first 500)")
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    return ps.save(fig, "eda_energy_vs_map", figdir)


# ------------------------------------------------------------------ report

def write_eda_config(cat, der, qc, cfgs, report_dir) -> str:
    """Render sections 1-5, 9, 10 to report_dir/eda.md (+ figures in report_dir/fig); returns the text. The
    trace-part marker sits between sections 5 and 9."""
    report_dir = Path(report_dir)
    figdir = report_dir / "fig"
    figdir.mkdir(parents=True, exist_ok=True)
    cells = eda.mode_a_cells(cat, der, qc, cfgs)
    prim = deploy._primary(cat, qc)
    vers = "; ".join(f"dataset {eda.VERSION.get(r, r)} ({r} rig, {int(n)} cells)" for r, n in prim.groupby("rig").size().items())
    L = ["# Exploration report: Raspberry Pi 5 power datasets, configuration level\n",
         "Descriptive only: nothing on this page fits or evaluates a predictor. All numbers use QC `primary` cells "
         f"only ({len(prim)} of {len(cat)} catalogue cells; {vers}). Energies are mJ per image (per frame / batch), "
         "latencies ms, power W. Every table states n and the dataset version; a configuration is one "
         "(model, precision, image size, batch, threads, clock, rig) cell group.\n",
         f"Mode A configurations: {prov(cfgs)} (Mode A eligible, mode `forward`).\n"]

    # 1 -------------------------------------------------------------- repeatability
    rp = eda.repeatability(cells, cfgs)
    ir, il = rp["icc_raw"], rp["icc_log"]
    L += ["## 1. Repeatability\n",
          f"CV of energy per image across repeat cells, per configuration with at least two cells "
          f"({rp['n_multi']} configurations; {rp['n_single']} single-cell configurations excluded). "
          f"{prov(rp['multi'])}.\n", md_table(rp["cv_summary"].rename(columns={"n": "n configurations", "median": "median CV %", "p90": "p90 CV %",
                                                      "max": "max CV %"}),
                   {c: "{:.2f}" for c in ("median CV %", "p90 CV %", "max CV %")}),
          "Configurations pool cells across the v2 families that share a command line (for example yolo11n, 640 px, "
          "4 threads, 2400 MHz pools families A-input, D and ref), so a configuration's CV includes any family effect.\n",
          f"**Noise floor** (median energy CV over those {rp['n_multi']} configurations, the figure used in every "
          f"error table): **{100 * rp['noise_floor']:.2f} %**.\n",
          "ICC(1,1) from a one-way random-effects ANOVA over configurations with >= 2 cells: "
          "`ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW)` with `n0 = (N - sum n_i^2 / N) / (k - 1)` (unbalanced groups), "
          "k configurations, N cells, MSB and MSW the between- and within-configuration mean squares. "
          f"On energy per image ({ir['k']} configurations, {ir['n_cells']} cells): ICC = {ir['icc']:.4f} "
          f"(between SD {np.sqrt(max(ir['var_between'], 0)):.0f} mJ, within SD {np.sqrt(ir['var_within']):.1f} mJ). "
          f"On log energy: ICC = {il['icc']:.4f} (1 - ICC = {1 - il['icc']:.1e}; on raw energy {1 - ir['icc']:.1e}). The ICC is near 1 because the configurations span a 50-fold energy "
          "range; it says repeats separate configurations, while the CV above is the measure of repeat noise.\n",
          "![CV of energy per image per configuration](fig/eda_cv_hist.png)\n"]
    fig_cv(rp["multi"], figdir)

    # 2 -------------------------------------------------------------- clock scaling and idle
    idle = deploy.idle_table(cat, der, qc)
    series = eda.scaling_series(cfgs, idle)
    st = eda.scaling_table(series)
    L += ["## 2. Clock scaling and idle power\n",
          "`P(f) = P_static + k f^alpha`, f in GHz, fitted with `scipy.optimize.curve_fit` to the configuration "
          "means at 1500, 2000 and 2400 MHz (per base model at 640 px for YOLO / 224 px for classifiers, 4 threads, "
          "batch 1, fp32, fan full; and for idle power per rig). **Three points and three parameters: the free fit is "
          "exact by construction, descriptive only**, no in-sample claim. A negative `P_static` is non-physical; the "
          "second fit restricts `0 <= P_static <= min P` and its residual shows what that costs. "
          "`n_cfg`/`n_cells` count the three clock configurations and their cells.\n"]
    if len(st):
        fails = st[~st["converged"]]
        L.append(md_table(st.drop(columns=["message"]), {"max_abs_resid": "{:.3f}", "P_static": "{:.3f}", "k": "{:.3f}",
                                                        "alpha": "{:.3f}"}))
        L.append("curve_fit did not converge for: " + (", ".join(f"{r.series}/{r.rig}/{r.fit}" for r in fails.itertuples())
                                                       if len(fails) else "none (all fits converged)") + ".\n")
        fw, fr = st[(st["kind"] == "forward power") & (st["fit"] == "free")], st[(st["kind"] == "forward power") & (st["fit"] != "free")]
        L.append(f"Forward power, free fits: P_static < 0 in {int((fw['P_static'] < 0).sum())} of {len(fw)} series, alpha "
                 f"{fw['alpha'].min():.2f}-{fw['alpha'].max():.2f}; restricted fits: P_static = 0 in {int((fr['P_static'] < 1e-6).sum())}, "
                 f"alpha {fr['alpha'].min():.2f}-{fr['alpha'].max():.2f}, worst residual {fr['max_abs_resid'].max():.3f} W. "
                 "Idle power rises more slowly than that (see `alpha` of the idle rows); the free idle fit on the v2 rig "
                 "sits on the alpha bound, so its parameters are not interpretable.\n" if len(fw) else "")
        fig_scaling(series, st, figdir)
        L.append("![Power against clock](fig/eda_clock_power.png)\n")
    else:
        L.append("_No complete three-clock series._\n")
    it = idle.copy().rename(columns={"P_idle_W": "idle W (median)", "n": "n cells", "iqr_W": "IQR W"})
    L += ["Idle power (median `baseline_W` over QC-primary cells of any workload) by rig, clock (or `ondemand`) and fan "
          f"mode; {prov(deploy._primary(cat, qc))} contribute one baseline each.\n",
          md_table(it, {"idle W (median)": "{:.3f}", "IQR W": "{:.3f}"})]

    # 3 -------------------------------------------------------------- rig shift
    rs = eda.rig_shift(cells)
    L += ["## 3. Rig shift: `ref` (v2, nov2026) against its v1 (oct2026) twin\n",
          "Same base model, 640 px, 4 threads, batch 1, fp32, same clock. Means over the cells of each side; "
          "`%` columns are `ref / v1 - 1`; `se %` is the standard error of the energy difference implied by the "
          f"two sides' CVs. {len(rs)} configuration pairs, {int(rs['n_ref'].sum() + rs['n_v1'].sum()) if len(rs) else 0} cells "
          "(ref cells = dataset v2 family `ref`; v1 cells = dataset v1).\n",
          md_table((rs[["base_model", "cpu_mhz", "n_v1", "n_ref", "energy_v1", "energy_ref", "energy_pct", "diff_se_pct",
                        "latency_v1", "latency_ref", "latency_pct", "power_v1", "power_ref", "power_pct"]]
                    .rename(columns={"energy_pct": "energy %", "latency_pct": "latency %", "power_pct": "power %",
                                     "diff_se_pct": "se %"}) if len(rs) else rs),
                   {k: "{:+.2f}" for k in ("energy %", "latency %", "power %")} | {"se %": "{:.2f}"}),
          (f"Energy shift ranges from {rs['energy_pct'].min():+.2f} % to {rs['energy_pct'].max():+.2f} % "
           f"(latency {rs['latency_pct'].min():+.2f} to {rs['latency_pct'].max():+.2f} %, power "
           f"{rs['power_pct'].min():+.2f} to {rs['power_pct'].max():+.2f} %); noise floor {100 * rp['noise_floor']:.2f} %.\n"
           if len(rs) else "")]

    # 4 -------------------------------------------------------------- interactions
    grid, be, pr = eda.thread_grid(cfgs), eda.batch_effect(cfgs), eda.precision_ratios(cfgs)
    L += ["## 4. Interactions\n", "### 4.1 Threads x image size\n",
          f"Energy per image (mJ), 2400 MHz, batch 1, fp32, dataset v2 (nov2026): {prov(cfgs[cfgs['config_key'].isin(grid['config_key'])])}.\n"]
    if len(grid):
        for m, g in grid.groupby("base_model"):
            L += [f"**{m}** energy (mJ):\n", md_table(pivot(g, "threads", "imgsz", "energy_mJ", "{:.0f}")),
                  f"**{m}** % against 4 threads at the same size:\n",
                  md_table(pivot(g, "threads", "imgsz", "pct_vs_4t", "{:+.1f}"))]
        t1 = grid[(grid["threads"] == 1) & (grid["imgsz"] == grid["imgsz"].min())]
        for r in t1.itertuples():
            L.append(f"{r.base_model} at {r.imgsz} px: 1 thread uses {-r.pct_vs_4t:.1f} % less energy per image than 4 "
                     f"threads ({r.energy_mJ:.0f} mJ; latency {r.latency_ms:.0f} ms).\n")
        fig_threads(grid, figdir)
        L.append("![Energy against image size by thread count](fig/eda_threads_imgsz.png)\n")
    L += ["### 4.2 Batch\n", f"Per-image energy, {', '.join(sorted(be['base_model'].unique()))} at 640 px, 4 threads, 2400 MHz, fp32, dataset "
          f"v2: {prov(cfgs[cfgs['config_key'].isin(be['config_key'])])}.\n",
          md_table(be.drop(columns="config_key"), {"pct_vs_batch1": "{:+.2f}"}),
          "### 4.3 INT8 and FP16 against FP32\n",
          "Ratio of the reduced-precision configuration over the matching FP32 configuration (same model, size, "
          f"batch, threads, clock, rig); {len(pr)} pairs, "
          f"{prov(cfgs[cfgs['config_key'].isin(set(pr['config_key']) | set(pr['config_key_fp32']))])}.\n",
          md_table(pr.drop(columns=["config_key", "config_key_fp32"]), {k: "{:.3f}" for k in ("energy_ratio", "latency_ratio", "power_ratio")})]

    # 5 -------------------------------------------------------------- phase and step shares
    ref = eda.reference_cells(cat, qc)
    ph, sp = eda.workload_shares(ref)
    L += ["## 5. Phase and step shares\n",
          f"Per workload (`model_key`), averaged over QC-primary plain cells at 2400 MHz (fan full, 4 threads, fp32; "
          f"batch 1 at 640 px for YOLO / 224 px for classifiers; LM and training at their only configuration): "
          f"{prov(ref)}. Entries are `energy % / time %` of the whole cell. Phase groups: `first` = first:* cold calls, "
          "`sync` = sync:*, `baseline` = baseline_A/B, `bench` = every bench:* microbenchmark phase, `measured` = the "
          "measured mode phases (forward / prefill+decode / train), `other` = idle.\n",
          md_table(share_pivot(ph, eda.GROUPS)),
          "Within the measured frames (iter >= 1), share of frame energy / time by step. A `profiler` step, if a frame "
          "has one, would be its own column; `non-forward E %` is the energy outside the forward step (the pre/post-processing "
          "discount for image workloads).\n",
          md_table(share_pivot(sp, STEP_ORDER, extra=True))]
    if len(ph):
        fig_phase(ph, figdir)
        L.append("![Measured share of cell energy and time](fig/eda_phase_shares.png)\n")
    L += [eda.TRACE_MARKER + "\n"]

    # 9 -------------------------------------------------------------- energy against accuracy
    maps = {}
    for r in cfgs[cfgs["base_model"].str.startswith("yolo") & (cfgs["rig"] == "nov2026")].itertuples():
        mr = data.read_meta(cat.set_index("cell_id").loc[r.cell_ids.split(";")[0], "path"]).get("map_reference") or {}
        for cid in r.cell_ids.split(";"):
            maps[cid] = (mr.get("map50_95", np.nan), mr.get("map50", np.nan))
    ea = eda.energy_accuracy(cfgs, maps)
    L += ["## 9. Energy against accuracy\n",
          "YOLO configurations at 2400 MHz, 4 threads, batch 1, fp32 and INT8/FP16, with `map_reference` (mAP on the "
          "first 500 COCO val2017 images, measured on the Pi, constant per model variant and image size). v1 classifiers "
          "have random weights and v1 cells carry no mAP, so only dataset v2 appears. "
          f"{prov(cfgs.merge(ea[['config_key']], on='config_key'))}. `pareto` = no other configuration is cheaper "
          "and at least as accurate.\n",
          md_table(ea.drop(columns="config_key"), {"map50_95": "{:.4f}", "map50": "{:.4f}", "energy_mJ": "{:.0f}", "latency_ms": "{:.0f}"})]
    if len(ea):
        fig_energy_map(ea, figdir)
        L.append("![Energy per image against mAP50-95](fig/eda_energy_vs_map.png)\n")

    # 10 ------------------------------------------------------------- LM and training
    lm, tr = eda.lm_per_token(cat, der, qc), eda.train_per_step(cat, der, qc)
    L += ["## 10. MiniGPT and training workloads\n",
          "### 10.1 MiniGPT energy per token\n",
          "Prefill: energy per frame divided by `seq_len` (the prompt length in `run_meta`, 16 / 64 / 128 for lm16 / "
          "lm64 / lm128). Decode: a frame is one forward pass on one token without a KV cache, so one token. "
          f"{cells_prov(cat, qc, 'lm', len(lm))}.\n",
          md_table(lm.drop(columns=["rig"]), {"E_per_token_mJ": "{:.3f}", "ms_per_token": "{:.3f}", "cv_pct": "{:.2f}",
                                            "mean_P_W": "{:.2f}"}),
          "### 10.2 Training energy per step\n",
          f"One frame is one optimiser step on `batch` samples (synthetic data; frame 0 excluded). {cells_prov(cat, qc, 'train', len(tr))}.\n",
          md_table(tr.drop(columns=["rig"]), {"E_step_mJ": "{:.1f}", "ms_step": "{:.1f}", "cv_pct": "{:.2f}",
                                            "mean_P_W": "{:.2f}"})]
    md = "\n".join(L)
    (report_dir / "eda.md").write_text(md)
    return md
