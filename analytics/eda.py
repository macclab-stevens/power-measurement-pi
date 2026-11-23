"""Exploration report, configuration level (spec section 5, items 1-5, 9, 10). Descriptive only: no
predictor is fitted here. Every table states n (configurations / cells) and the dataset version.

Computations return DataFrames; write_eda_config() renders them to eda.md plus figures. Section 6-8
(trace level) are appended later by splicing into TRACE_MARKER (see splice_trace)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

import data
import deploy
import plotstyle as ps
import tables

TRACE_MARKER = "<!-- EDA-TRACE-PART: sections 6-8 are spliced in here by the trace-level report -->"
CLOCKS = (1500, 2000, 2400)
VERSION = {"oct2026": "v1", "nov2026": "v2"}
GROUPS = ["load_weights", "first", "warmup", "sync", "baseline", "measured", "gap", "serve", "bench", "other"]
NAN = float("nan")


# ------------------------------------------------------------------ repeatability

def icc_oneway(df: pd.DataFrame, group: str, value: str) -> dict:
    """ICC(1,1) of one-way random-effects ANOVA over groups with >= 2 observations:
    (MSB - MSW) / (MSB + (n0 - 1) MSW), n0 = (N - sum n_i^2 / N) / (k - 1) (unbalanced adjustment)."""
    n = df.groupby(group)[value].transform("size")
    d = df[n >= 2]
    g = d.groupby(group)[value]
    ni, means = g.size().to_numpy(float), g.mean().to_numpy(float)
    k, tot = len(ni), ni.sum()
    out = dict(icc=NAN, k=k, n_cells=int(tot), n0=NAN, msb=NAN, msw=NAN, var_between=NAN, var_within=NAN)
    if k < 2 or tot <= k:
        return out
    grand = d[value].mean()
    msb = float((ni * (means - grand) ** 2).sum() / (k - 1))
    msw = float(((d[value] - g.transform("mean")) ** 2).sum() / (tot - k))
    n0 = float((tot - (ni ** 2).sum() / tot) / (k - 1))
    out.update(icc=(msb - msw) / (msb + (n0 - 1) * msw), n0=n0, msb=msb, msw=msw,
               var_between=(msb - msw) / n0, var_within=msw)
    return out


def mode_a_cells(cat, der, qc, cfgs) -> pd.DataFrame:
    """QC-primary forward cells that make up the Mode A configurations, with their config_key."""
    ct = deploy.cell_table(cat, der, qc)
    key = {c: r.config_key for r in cfgs.itertuples() for c in r.cell_ids.split(";")}
    ct = ct[ct["cell_id"].isin(key)].copy()
    ct["config_key"] = ct["cell_id"].map(key)
    return ct


def repeatability(cells: pd.DataFrame, cfgs: pd.DataFrame) -> dict:
    multi = cfgs[cfgs["n_cells"] >= 2]
    sub = cells[cells["config_key"].isin(multi["config_key"])].assign(
        log_energy=lambda d: np.log(d["energy_mJ_per_image"]))
    cv = multi["energy_cv"] * 100
    mc = multi.assign(cv=cv, rig_=multi["rig"])
    rows = [(r, g["cv"]) for r, g in mc.groupby("rig")] + [("all", mc["cv"])]
    by_rig = pd.DataFrame([{"rig": r, "n": len(x), "median": x.median(), "p90": x.quantile(.9), "max": x.max()}
                           for r, x in rows if len(x)], columns=["rig", "n", "median", "p90", "max"])
    return {"n_multi": len(multi), "n_single": len(cfgs) - len(multi), "noise_floor": tables.noise_floor(cfgs),
            "cv_summary": by_rig, "icc_raw": icc_oneway(sub, "config_key", "energy_mJ_per_image"),
            "icc_log": icc_oneway(sub, "config_key", "log_energy"), "multi": multi}


# ------------------------------------------------------------------ clock scaling and idle

def fit_power_law(f_ghz, p, bounded: bool = False) -> dict:
    """curve_fit P = P_static + k f^alpha (f in GHz). Free: k > 0, 0.05 <= alpha <= 10; bounded adds
    0 <= P_static <= min(P). Several starts, best residual kept; a failure is reported, not hidden."""
    f, y = np.asarray(f_ghz, float), np.asarray(p, float)
    out = dict(P_static=NAN, k=NAN, alpha=NAN, max_abs_resid=NAN, converged=False, at_bound=False, message="")
    lo = [0.0 if bounded else -np.inf, 0.0, 0.05]
    hi = [float(y.min()) if bounded else np.inf, np.inf, 10.0]
    best, last = None, ""
    for a0 in (1.0, 2.0, 0.6, 3.0):
        k0 = max(float(np.ptp(y)), 1e-3) / max(float(np.ptp(f ** a0)), 1e-9)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(lambda x, a, k, al: a + k * x ** al, f, y, p0=[min(float(y.min()) * 0.5, hi[0]), k0, a0],
                                    bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError) as exc:
            last = str(exc)
            continue
        r = float(np.abs(popt[0] + popt[1] * f ** popt[2] - y).max())
        if best is None or r < best[1]:
            best = (popt, r)
    if best is None:
        out["message"] = f"curve_fit did not converge: {last}"
        return out
    (a, k, al), r = best
    out.update(P_static=float(a), k=float(k), alpha=float(al), max_abs_resid=r, converged=True,
               at_bound=bool(np.isclose(a, lo[0]) or np.isclose(a, hi[0]) or np.isclose(al, lo[2]) or np.isclose(al, hi[2])))
    return out


def scaling_series(cfgs: pd.DataFrame, idle: pd.DataFrame) -> list[dict]:
    """(label, rig, f_GHz, P_W, n_configs, n_cells, kind) for each base model per rig at 640 px (YOLO) / 224 px
    (classifiers), 4 threads, batch 1, fp32, and for idle power per rig (fan full); only complete 3-clock sets."""
    out = []
    c = cfgs[(cfgs["threads"] == 4) & (cfgs["batch"] == 1) & (cfgs["precision"] == "fp32")
             & (cfgs["variant"] == cfgs["base_model"]) & cfgs["cpu_mhz"].isin(CLOCKS)]
    c = c[c["imgsz"] == np.where(c["base_model"].str.startswith("yolo"), 640, 224)]
    for (m, rig), g in c.groupby(["base_model", "rig"]):
        g = g.sort_values("cpu_mhz")
        if len(g) == len(CLOCKS):
            out.append(dict(label=m, rig=rig, f=g["cpu_mhz"].to_numpy() / 1000, P=g["mean_P_W"].to_numpy(),
                            n_cfg=len(g), n_cells=int(g["n_cells"].sum()), kind="forward power"))
    i = idle[(idle["fan_mode"] == "full") & idle["cpu_key"].isin([str(x) for x in CLOCKS])]
    for rig, g in i.groupby("rig"):
        g = g.assign(f=g["cpu_key"].astype(int) / 1000).sort_values("f")
        if len(g) == len(CLOCKS):
            out.append(dict(label="idle", rig=rig, f=g["f"].to_numpy(), P=g["P_idle_W"].to_numpy(), n_cfg=len(g),
                            n_cells=int(g["n"].sum()), kind="idle power"))
    return out


def scaling_table(series: list[dict]) -> pd.DataFrame:
    rows = []
    for s in series:
        for bounded in (False, True):
            r = fit_power_law(s["f"], s["P"], bounded)
            rows.append({"series": s["label"], "kind": s["kind"], "rig": s["rig"], "n_cfg": s["n_cfg"],
                         "n_cells": s["n_cells"], "fit": "P_static in [0, min P]" if bounded else "free",
                         **{k: r[k] for k in ("P_static", "k", "alpha", "max_abs_resid", "converged", "at_bound", "message")}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rig shift and interactions

def rig_shift(cells: pd.DataFrame) -> pd.DataFrame:
    """Each `ref` (nov2026) configuration against its v1 (oct2026) twin: same base model, imgsz, clock,
    threads, batch, precision. Means over cells; diff_se_pct is the standard error of the % difference from the
    per-side CVs (a noise yardstick)."""
    key = ["base_model", "variant", "precision", "imgsz", "batch", "threads", "cpu_mhz"]
    m = {"energy": "energy_mJ_per_image", "latency": "latency_ms_per_image", "power": "mean_P_W"}

    def agg(d):
        g = d.groupby(key)
        out = g.agg(n=("cell_id", "size"), **{k: (v, "mean") for k, v in m.items()})
        out["cv"] = g["energy_mJ_per_image"].std(ddof=1) / out["energy"]
        return out

    v1, ref = agg(cells[cells["rig"] == "oct2026"]), agg(cells[(cells["rig"] == "nov2026") & (cells["family"] == "ref")])
    j = ref.add_suffix("_ref").join(v1.add_suffix("_v1"), how="inner").reset_index()
    for k in m:
        j[f"{k}_pct"] = 100 * (j[f"{k}_ref"] / j[f"{k}_v1"] - 1)
    j["diff_se_pct"] = 100 * np.sqrt(j["cv_ref"] ** 2 / j["n_ref"] + j["cv_v1"] ** 2 / j["n_v1"])
    return j.sort_values(["base_model", "cpu_mhz"]).reset_index(drop=True)


def _plain(cfgs, **eq) -> pd.DataFrame:
    m = (cfgs["rig"] == "nov2026") & (cfgs["variant"] == cfgs["base_model"])
    for k, v in eq.items():
        m &= cfgs[k] == v
    return cfgs[m]


def thread_grid(cfgs: pd.DataFrame, models=("yolo11n", "yolo11s")) -> pd.DataFrame:
    """Energy per image over threads x imgsz (2400 MHz, batch 1, fp32, nov2026), % against 4 threads."""
    g = _plain(cfgs, cpu_mhz=2400, batch=1, precision="fp32")
    g = g[g["base_model"].isin(models)][["config_key", "base_model", "imgsz", "threads", "energy_mJ", "latency_ms", "mean_P_W", "n_cells"]]
    t4 = g[g["threads"] == 4].set_index(["base_model", "imgsz"])["energy_mJ"]
    g["pct_vs_4t"] = [100 * (r.energy_mJ / t4.get((r.base_model, r.imgsz), NAN) - 1) for r in g.itertuples()]
    return g.sort_values(["base_model", "imgsz", "threads"]).reset_index(drop=True)


def batch_effect(cfgs: pd.DataFrame) -> pd.DataFrame:
    g = _plain(cfgs, cpu_mhz=2400, threads=4, precision="fp32", imgsz=640)
    g = g[g.groupby("base_model")["batch"].transform("nunique") > 1]
    g = g[["config_key", "base_model", "batch", "energy_mJ", "latency_ms", "mean_P_W", "n_cells"]].sort_values(["base_model", "batch"])
    b1 = g[g["batch"] == 1].set_index("base_model")["energy_mJ"]
    g["pct_vs_batch1"] = [100 * (r.energy_mJ / b1.get(r.base_model, NAN) - 1) for r in g.itertuples()]
    return g.reset_index(drop=True)


def precision_ratios(cfgs: pd.DataFrame) -> pd.DataFrame:
    """INT8 / FP16 configurations over the FP32 configuration with the same model, size, batch, threads, clock, rig."""
    key = ["base_model", "imgsz", "batch", "threads", "cpu_mhz", "rig"]
    fp = cfgs[(cfgs["precision"] == "fp32") & (cfgs["variant"] == cfgs["base_model"])].set_index(key)
    q = cfgs[cfgs["precision"] != "fp32"].join(
        fp[["config_key", "energy_mJ", "latency_ms", "mean_P_W", "n_cells"]].add_suffix("_fp32"), on=key)
    q = q.dropna(subset=["energy_mJ_fp32"])
    q = q.assign(energy_ratio=q["energy_mJ"] / q["energy_mJ_fp32"], latency_ratio=q["latency_ms"] / q["latency_ms_fp32"],
                 power_ratio=q["mean_P_W"] / q["mean_P_W_fp32"])
    return q[["config_key", "config_key_fp32", "variant", "precision", "base_model", "imgsz", "cpu_mhz", "n_cells",
              "n_cells_fp32", "energy_ratio", "latency_ratio", "power_ratio"]].sort_values(["variant", "cpu_mhz"]).reset_index(drop=True)


# ------------------------------------------------------------------ phase and step shares

def phase_group(phase: str, modes) -> str:
    if phase in modes:
        return "measured"
    for pre, name in (("first:", "first"), ("sync:", "sync"), ("baseline_", "baseline"), ("bench:", "bench")):
        if str(phase).startswith(pre):
            return name
    return phase if phase in ("load_weights", "warmup", "gap", "serve") else "other"


def _shares(rows: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    g = rows.assign(label=label).groupby("label")[["E_mJ", "dur_ms"]].sum()
    return pd.DataFrame({"E_share": g["E_mJ"] / g["E_mJ"].sum(), "dur_share": g["dur_ms"] / g["dur_ms"].sum()})


def phase_shares(pe: pd.DataFrame, modes) -> pd.DataFrame:
    """Share of the cell's energy and time per phase group (all rows of the cell)."""
    return _shares(pe, pe["phase"].map(lambda p: phase_group(p, modes)))


def step_shares(pe: pd.DataFrame, modes) -> pd.DataFrame:
    """Share of the measured frames' (iter >= 1) energy and time per step; rows without a step are `(none)`."""
    rows = pe[pe["phase"].isin(list(modes)) & (pd.to_numeric(pe["iter"], errors="coerce") >= 1)]
    return _shares(rows, rows["step"].fillna("(none)")) if len(rows) else pd.DataFrame(columns=["E_share", "dur_share"])


def reference_cells(cat: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """QC-primary plain cells at 2400 MHz pinned, fan full, fp32, 4 threads: batch 1 at 640 px (YOLO) / 224 px
    (classifiers); LM and training cells at their only configuration."""
    c = deploy._primary(cat, qc)
    c = c[c["serve_fps"].isna() & c["soak_minutes"].isna() & c["concurrent_mode"].isna() & (c["input_source"] == "repeat")
          & (c["cpu_mhz"] == 2400) & (c["governor"] == "pinned") & (c["fan_mode"] == "full") & (c["threads"] == 4)
          & (c["precision"] == "fp32") & (c["variant"] == c["base_model"])]
    img = (c["task"] == "image") & (c["batch"] == 1) & (c["imgsz"] == np.where(c["model_key"].str.startswith("yolo"), 640, 224))
    return c[img | (c["task"] != "image")]


def workload_shares(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(phase shares, step shares) averaged over cells per (model_key, rig); long: model_key, rig, label, E_share,
    dur_share, n_cells."""
    ph, st = [], []
    for r in ref.itertuples():
        pe = data.read_csv_opt(r.path, "phase_energy.csv")
        if pe is None:
            continue
        modes = data.read_meta(r.path).get("modes") or []
        for out, fn in ((ph, phase_shares), (st, step_shares)):
            s = fn(pe, modes).reset_index()
            out.append(s.assign(model_key=r.model_key, rig=r.rig, cell_id=r.cell_id))

    def avg(parts):
        if not parts:
            return pd.DataFrame(columns=["model_key", "rig", "label", "E_share", "dur_share", "n_cells"])
        d = pd.concat(parts)
        g = d.groupby(["model_key", "rig", "label"])[["E_share", "dur_share"]].sum().reset_index()
        n = d.groupby(["model_key", "rig"])["cell_id"].nunique()
        g["n_cells"] = [n[(m, r)] for m, r in zip(g["model_key"], g["rig"])]
        g[["E_share", "dur_share"]] = g[["E_share", "dur_share"]].div(g["n_cells"], axis=0)   # absent in a cell = 0
        return g
    return avg(ph), avg(st)


# ------------------------------------------------------------------ energy vs accuracy, LM, training

def energy_accuracy(cfgs: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """YOLO configurations at 2400 MHz, 4 threads, batch 1, nov2026 (v2 carries map_reference); `maps` is
    {cell_id: (map50_95, map50)}. pareto = no other configuration is both cheaper and at least as accurate."""
    g = cfgs[(cfgs["base_model"].str.startswith("yolo")) & (cfgs["cpu_mhz"] == 2400) & (cfgs["threads"] == 4)
             & (cfgs["batch"] == 1) & (cfgs["rig"] == "nov2026")].copy()
    mm = [np.nanmean([maps.get(c, (NAN, NAN))[i] for c in ids.split(";")]) for ids in g["cell_ids"] for i in (0, 1)]
    g["map50_95"], g["map50"] = mm[0::2], mm[1::2]
    g = g.dropna(subset=["map50_95"])
    g["pareto"] = [not ((g["energy_mJ"] < r.energy_mJ) & (g["map50_95"] >= r.map50_95)).any() for r in g.itertuples()]
    return g[["config_key", "base_model", "variant", "precision", "imgsz", "n_cells", "map50_95", "map50", "energy_mJ",
              "latency_ms", "pareto"]].sort_values("energy_mJ").reset_index(drop=True)


def _modes(cat, der, qc, modes) -> pd.DataFrame:
    return deploy._primary(cat, qc).merge(der[der["mode"].isin(modes)].drop(columns="batch"), on="cell_id")


def lm_per_token(cat, der, qc) -> pd.DataFrame:
    """Energy per token per (model, mode, clock). A prefill frame processes `seq_len` tokens (run_meta); a decode
    frame is one forward on a single token (no KV cache), so one token."""
    c = _modes(cat, der, qc, ("prefill", "decode"))
    c = c[c["task"] == "lm"].copy()
    seq = {p: data.read_meta(p).get("seq_len") for p in c["path"].unique()}
    c["tokens"] = np.where(c["mode"] == "prefill", c["path"].map(seq), 1).astype(float)
    c["E_per_token_mJ"] = c["energy_mJ_per_image"] / c["tokens"]
    c["ms_per_token"] = c["latency_ms_per_image"] / c["tokens"]
    g = c.groupby(["model_key", "mode", "cpu_mhz", "rig"])
    out = g.agg(n_cells=("cell_id", "size"), tokens_per_frame=("tokens", "first"), E_per_token_mJ=("E_per_token_mJ", "mean"),
                ms_per_token=("ms_per_token", "mean"), mean_P_W=("mean_P_W", "mean"))
    out["cv_pct"] = 100 * g["E_per_token_mJ"].std(ddof=1) / out["E_per_token_mJ"]
    return out.reset_index()


def train_per_step(cat, der, qc) -> pd.DataFrame:
    """Energy per optimiser step (frame) per (workload, optimiser, clock); batch is the number of samples."""
    c = _modes(cat, der, qc, ("train",)).copy()
    c["workload"] = c["model_key"].str.replace("^train_", "", regex=True).str.rsplit("_", n=1).str[0]
    c["optimiser"] = c["model_key"].str.rsplit("_", n=1).str[1]
    c["E_step_mJ"] = c["energy_mJ_per_image"] * c["batch"]
    c["ms_step"] = c["latency_ms_per_image"] * c["batch"]
    g = c.groupby(["workload", "optimiser", "cpu_mhz", "rig"])
    out = g.agg(n_cells=("cell_id", "size"), batch=("batch", "first"), E_step_mJ=("E_step_mJ", "mean"),
                ms_step=("ms_step", "mean"), mean_P_W=("mean_P_W", "mean"))
    out["cv_pct"] = 100 * g["E_step_mJ"].std(ddof=1) / out["E_step_mJ"]
    return out.reset_index()


def write_eda_config(*args, **kwargs) -> str:
    """Render sections 1-5, 9, 10 to eda.md plus figures (see eda_write.write_eda_config)."""
    import eda_write
    return eda_write.write_eda_config(*args, **kwargs)


def splice_trace(md: str, trace_md: str) -> str:
    """Replace the trace-part marker of an eda.md text with the trace-level sections."""
    return md.replace(TRACE_MARKER, trace_md.rstrip() + "\n")
