#!/usr/bin/env python3
"""Honest trace plots: actual 10 Hz grid vs stitched prediction + residuals by phase.

Consumes: actual 10 Hz grid (targets.build_grid_targets) + stitched prediction
  (stitch.stitch with LOMO train-prior phase means, planned-duration bands).
Produces: <stem>_actual_vs_pred.png + <stem>_residual_by_phase.png, each with
  VDD_CORE + schema + threads/governor/PSU footnote. Never hides FAIL:
  titles/footnotes report coverage + base-vs-full MAPE with PASS/FAIL, and
  both baseline (latency-only) and full (phase-aware) arms are shown.

Constraints: Agg backend, Okabe-Ito palette, no twin axes.
Usage: uv run trace_forecast/plot_trace.py --run runs/v3_yolo11n_640 --out /tmp/trace.png
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from targets import build_grid_targets, build_phase_targets
except ImportError:  # pragma: no cover - fallback when run as module
    from trace_forecast.targets import build_grid_targets, build_phase_targets
try:
    from stitch import stitch
except ImportError:  # pragma: no cover
    from trace_forecast.stitch import stitch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
FORWARD = {"forward", "prefill", "decode"}
DUR_REL_STD = 0.1
RMSE_FLOOR = 0.15
RMSE_BASE = 0.3


def load_meta(run_dir):
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        return json.load(f)


def footnote_for(meta, run_dir, hz, extra=""):
    schema = meta.get("schema_version", "?")
    threads = meta.get("threads", 4)
    governor = meta.get("governor", "ondemand? (not logged)")
    psu = meta.get("psu", "5V/5A? (not logged)")
    base = (
        f"VDD_CORE rail only | schema={schema} | threads={threads} "
        f"governor={governor} PSU={psu} | {os.path.basename(run_dir.rstrip('/'))} hz={hz}"
    )
    return base + (f" | {extra}" if extra else "")


def _finalize_fig(fig, footnote):
    if footnote:
        fig.text(0.01, 0.02, footnote, fontsize=6, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=[0, 0.14, 1, 0.90])


def build_actual_with_phase(run_dir, hz=10):
    """10 Hz bins with majority phase (linear scan, mirrors targets.build_grid_targets)."""
    with open(os.path.join(run_dir, "samples.csv")) as f:
        rows = list(csv.DictReader(f))
    step = 1000.0 / hz
    t0 = float(rows[0]["t_ms"])
    t1 = float(rows[-1]["t_ms"])
    out = []
    j = 0
    n = len(rows)
    ts = [float(r["t_ms"]) for r in rows]
    b = t0
    while b < t1:
        while j < n and ts[j] < b:
            j += 1
        k = j
        cnt = collections.Counter()
        psum = 0.0
        m = 0
        while k < n and ts[k] < b + step:
            cnt[rows[k]["phase"].split(":")[0]] += 1
            psum += float(rows[k]["P_W"])
            m += 1
            k += 1
        if m:
            ph = cnt.most_common(1)[0][0]
            out.append({"t_s": round((b - t0) / 1000.0, 1), "phase": ph, "P_mean": psum / m, "n": m})
        b += step
    return out


def load_train_prior(run_dir):
    """LOMO train prior from sibling runs/v3_* (schema 3), else in-sample fallback.

    Returns dict with phase_mean, phase_rmse (log-space std, floored), mean_p,
    idle (baseline mean), t_fwd_prior_s, fallback (bool), train_n.
    """
    cur_abs = os.path.abspath(run_dir)
    cur_norm = os.path.normpath(cur_abs)
    repo = os.getcwd()
    # run_dir like runs/v3_x -> repo root is parent of runs/
    if os.path.basename(os.path.dirname(cur_norm)) == "runs":
        repo = os.path.dirname(os.path.dirname(cur_norm))
    cands = sorted(glob.glob(os.path.join(repo, "runs", "v3_*")))
    cands = [d for d in cands if os.path.normpath(os.path.abspath(d)) != cur_norm and os.path.isdir(d)]
    train_dirs = []
    for d in cands:
        try:
            m = json.load(open(os.path.join(d, "run_meta.json")))
        except Exception:
            continue
        if m.get("schema_version") == 3 and "_TAINTED_V2" not in d:
            train_dirs.append(d)
    if train_dirs:
        trows = []
        for d in train_dirs:
            try:
                trows.extend(build_phase_targets(d))
            except Exception:
                continue
        if trows:
            by_ph = collections.defaultdict(list)
            for r in trows:
                by_ph[r["phase"]].append(r["P_mean"])
            phase_mean = {k: sum(v) / len(v) for k, v in by_ph.items()}
            phase_rmse = {}
            for k, v in by_ph.items():
                if len(v) > 1:
                    logs = [math.log(max(x, 1e-6)) for x in v]
                    mu = sum(logs) / len(logs)
                    sd = math.sqrt(sum((x - mu) ** 2 for x in logs) / len(logs))
                else:
                    sd = RMSE_BASE
                phase_rmse[k] = max(float(sd), RMSE_FLOOR)
            mean_p = sum(r["P_mean"] for r in trows) / len(trows)
            base_rows = [r["P_mean"] for r in trows if r["phase"] in ("baseline_A", "baseline_B")]
            idle = sum(base_rows) / len(base_rows) if base_rows else mean_p
            fwd = [r["dur_s"] for r in trows if r["phase"] in FORWARD]
            t_prior = sum(fwd) / len(fwd) if fwd else 0.4
            return {
                "phase_mean": phase_mean,
                "phase_rmse": phase_rmse,
                "mean_p": float(mean_p),
                "idle": float(idle),
                "t_fwd_prior_s": float(t_prior),
                "fallback": False,
                "train_n": len(trows),
                "train_dirs": train_dirs,
            }
    # fallback: in-sample phase means (demo only; forecast uses train prior R5)
    rows = build_phase_targets(run_dir)
    by_ph = collections.defaultdict(list)
    for r in rows:
        by_ph[r["phase"]].append(r["P_mean"])
    phase_mean = {k: sum(v) / len(v) for k, v in by_ph.items()}
    phase_rmse = {k: RMSE_BASE for k in by_ph}
    mean_p = sum(r["P_mean"] for r in rows) / len(rows)
    base_rows = [r["P_mean"] for r in rows if r["phase"] in ("baseline_A", "baseline_B")]
    idle = sum(base_rows) / len(base_rows) if base_rows else mean_p
    fwd = [r["dur_s"] for r in rows if r["phase"] in FORWARD]
    t_prior = sum(fwd) / len(fwd) if fwd else 0.4
    return {
        "phase_mean": phase_mean,
        "phase_rmse": phase_rmse,
        "mean_p": float(mean_p),
        "idle": float(idle),
        "t_fwd_prior_s": float(t_prior),
        "fallback": True,
        "train_n": 0,
        "train_dirs": [],
    }


def build_stitched_traces(run_dir, prior):
    """Stitched base (latency-only) + full (phase-aware) traces.

    Uses stitch() for the forecast trace (compliance + E_cum bands), then maps
    to time-matched bins for honest overlay: stitch rounds each of the ~200
    segments to 0.1 s, accumulating ~3 s drift that misaligns phases by index.
    Time-matched pred uses identical centers/bands (quadrature log scale) at
    the actual bin times, so coverage/residuals reflect model error, not
    rounding drift. Both are returned; plots/stats use time-matched.
    """
    phase_rows = build_phase_targets(run_dir)
    # R5 note: demo uses measured dur_s so time axes align; forecast uses
    # planned frames x train prior (see footnote). Order preserved.
    plan = [{"phase": r["phase"], "frames": 0, "dur_s": r["dur_s"]} for r in phase_rows]
    idle = prior["idle"]
    mean_p = prior["mean_p"]
    phase_mean = prior["phase_mean"]
    phase_rmse = prior["phase_rmse"]

    def base_fn(t):
        return idle

    def delta_full(ph):
        return (phase_mean.get(ph, mean_p) - idle, phase_rmse.get(ph, RMSE_BASE))

    def delta_base(ph):
        return (mean_p - idle, RMSE_BASE)

    # stitch() integration (consumes stitched prediction; E_cum bands per R5)
    _full_stitched = stitch(plan, prior["t_fwd_prior_s"], base_fn, delta_full,
                            dur_rel_std=DUR_REL_STD, step=0.1)
    _base_stitched = stitch(plan, prior["t_fwd_prior_s"], base_fn, delta_base,
                            dur_rel_std=DUR_REL_STD, step=0.1)
    # time-matched overlay (same t_s as actual grid; avoids rounding drift)
    actual = build_actual_with_phase(run_dir, hz=10)
    base, full = [], []
    for a in actual:
        ph = a["phase"]
        pf = phase_mean.get(ph, mean_p)
        sig_f = math.sqrt(phase_rmse.get(ph, RMSE_BASE) ** 2 + DUR_REL_STD ** 2)
        lp = math.log(max(pf, 1e-6))
        full.append({"t_s": a["t_s"], "phase": ph, "P_mean": pf,
                     "P_lo": math.exp(lp - 1.96 * sig_f), "P_hi": math.exp(lp + 1.96 * sig_f),
                     "E_cum": 0.0, "E_lo": 0.0, "E_hi": 0.0})
        sig_b = math.sqrt(RMSE_BASE ** 2 + DUR_REL_STD ** 2)
        lb = math.log(max(mean_p, 1e-6))
        base.append({"t_s": a["t_s"], "phase": ph, "P_mean": mean_p,
                     "P_lo": math.exp(lb - 1.96 * sig_b), "P_hi": math.exp(lb + 1.96 * sig_b),
                     "E_cum": 0.0, "E_lo": 0.0, "E_hi": 0.0})
    # fill E_cum from stitched (truncated to matched length for reference)
    n = min(len(actual), len(_full_stitched), len(_base_stitched))
    for i in range(n):
        full[i]["E_cum"] = _full_stitched[i]["E_cum"]
        full[i]["E_lo"] = _full_stitched[i]["E_lo"]
        full[i]["E_hi"] = _full_stitched[i]["E_hi"]
        base[i]["E_cum"] = _base_stitched[i]["E_cum"]
        base[i]["E_lo"] = _base_stitched[i]["E_lo"]
        base[i]["E_hi"] = _base_stitched[i]["E_hi"]
    return base, full


def _bin_mape(actual, pred_by_idx):
    s, n = 0.0, 0
    for a, p in zip(actual, pred_by_idx):
        s += abs(a["P_mean"] - p) / max(abs(a["P_mean"]), 1e-12)
        n += 1
    return s / max(1, n)


def coverage_stats(actual, pred):
    """Time-aligned by index (both 10 Hz from t=0); reports honest coverage."""
    n = min(len(actual), len(pred))
    cov = sum(1 for i in range(n) if pred[i]["P_lo"] <= actual[i]["P_mean"] <= pred[i]["P_hi"])
    mape = _bin_mape(actual[:n], [pred[i]["P_mean"] for i in range(n)])
    return {"cover": cov, "total": n, "frac": cov / max(1, n), "mape": mape}


def plot_actual_vs_pred(actual, pred, out, footnote="", title_extra=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([r["t_s"] for r in actual], [r["P_mean"] for r in actual], lw=1, color=PAL[0], label="actual 10Hz")
    ax.plot([r["t_s"] for r in pred], [r["P_mean"] for r in pred], lw=1, color=PAL[1], label="pred")
    ax.fill_between(
        [r["t_s"] for r in pred],
        [r["P_lo"] for r in pred],
        [r["P_hi"] for r in pred],
        color=PAL[1],
        alpha=0.2,
        label="95%",
    )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("VDD_CORE P (W)")
    ax.legend(fontsize=8)
    if title_extra:
        ax.set_title(title_extra, fontsize=9)
    _finalize_fig(fig, footnote)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_actual_vs_pred_with_base(actual, base, full, out, footnote="", title=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([r["t_s"] for r in actual], [r["P_mean"] for r in actual], lw=1, color=PAL[0], label="actual 10Hz")
    ax.plot(
        [r["t_s"] for r in base], [r["P_mean"] for r in base], lw=1, color=PAL[2], ls="--", label="base (latency-only)"
    )
    ax.plot([r["t_s"] for r in full], [r["P_mean"] for r in full], lw=1, color=PAL[1], label="full (phase-aware)")
    ax.fill_between(
        [r["t_s"] for r in full],
        [r["P_lo"] for r in full],
        [r["P_hi"] for r in full],
        color=PAL[1],
        alpha=0.2,
        label="full 95%",
    )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("VDD_CORE P (W)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    _finalize_fig(fig, footnote)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_residual_by_phase(actual, base, full, out, footnote="", title=""):
    # per-bin residuals grouped by true phase; shows base vs full (never hides FAIL)
    phases = sorted({r["phase"] for r in actual})
    n = min(len(actual), len(base), len(full))
    res_base = collections.defaultdict(list)
    res_full = collections.defaultdict(list)
    for i in range(n):
        ph = actual[i]["phase"]
        res_base[ph].append(actual[i]["P_mean"] - base[i]["P_mean"])
        res_full[ph].append(actual[i]["P_mean"] - full[i]["P_mean"])
    x = list(range(len(phases)))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, ph in enumerate(phases):
        rb = res_base.get(ph, [0.0])
        rf = res_full.get(ph, [0.0])
        mb = sum(rb) / len(rb)
        mf = sum(rf) / len(rf)
        sb = math.sqrt(sum((v - mb) ** 2 for v in rb) / len(rb)) if len(rb) > 1 else 0.0
        sf = math.sqrt(sum((v - mf) ** 2 for v in rf) / len(rf)) if len(rf) > 1 else 0.0
        ax.bar(j - w / 2, mb, w, color=PAL[2], label="base resid" if j == 0 else "")
        ax.bar(j + w / 2, mf, w, color=PAL[1], label="full resid" if j == 0 else "")
        ax.errorbar([j - w / 2], [mb], yerr=[sb], color="black", capsize=3, lw=1)
        ax.errorbar([j + w / 2], [mf], yerr=[sf], color="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("phase")
    ax.set_ylabel("VDD_CORE residual actual-pred (W)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    _finalize_fig(fig, footnote)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def derive_outputs(out):
    """Map --out base to the two binding deliverables (plus literal for compat)."""
    outs = set()
    outs.add(out)
    if out.endswith("_actual_vs_pred.png") or out.endswith("_residual_by_phase.png"):
        # explicit single output requested; still derive sibling from stem
        if out.endswith("_actual_vs_pred.png"):
            stem = out[: -len("_actual_vs_pred.png")]
        else:
            stem = out[: -len("_residual_by_phase.png")]
        outs.add(stem + "_actual_vs_pred.png")
        outs.add(stem + "_residual_by_phase.png")
    elif out.endswith(".png"):
        stem = out[:-4]
        outs.add(stem + "_actual_vs_pred.png")
        outs.add(stem + "_residual_by_phase.png")
    else:
        outs.add(out + "_actual_vs_pred.png")
        outs.add(out + "_residual_by_phase.png")
    # canonical pair for stem
    if out.endswith(".png") and not out.endswith(("_actual_vs_pred.png", "_residual_by_phase.png")):
        stem = out[:-4]
        return stem + "_actual_vs_pred.png", stem + "_residual_by_phase.png", sorted(outs)
    # fall back: pick the two canonical names
    cands = sorted(outs)
    p1 = [p for p in cands if p.endswith("_actual_vs_pred.png")]
    p2 = [p for p in cands if p.endswith("_residual_by_phase.png")]
    return (p1[0] if p1 else out), (p2[0] if p2 else out), cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=int, default=10)
    a = ap.parse_args()
    run_dir = a.run
    meta = load_meta(run_dir)
    actual = build_actual_with_phase(run_dir, hz=a.hz)
    prior = load_train_prior(run_dir)
    base, full = build_stitched_traces(run_dir, prior)
    st_full = coverage_stats(actual, full)
    st_base = coverage_stats(actual, base)
    delta = st_base["mape"] - st_full["mape"]
    win = "WIN" if delta > 0.03 else "FAIL"
    cov_tag = "PASS" if st_full["frac"] >= 0.9 else "FAIL"
    mape_tag = "PASS" if st_full["mape"] <= 0.05 else "FAIL"
    dur_note = "durations=measured(demo); forecast uses planned frames x prior (R5)"
    fb_note = "fallback:in-sample means" if prior.get("fallback") else f"LOMO train n={prior.get('train_n')}"
    foot = footnote_for(meta, run_dir, a.hz, f"{fb_note} | {dur_note}")
    title1 = (
        f"Actual vs pred — {os.path.basename(run_dir.rstrip('/'))} | "
        f"cover {st_full['cover']}/{st_full['total']} ({st_full['frac']:.1%} {cov_tag}; want >=90%) | "
        f"full bin MAPE {st_full['mape']:.1%} ({mape_tag}; want <=5%) | "
        f"base {st_base['mape']:.1%} vs full {st_full['mape']:.1%} ({win}; want >3pp)"
    )
    title2 = (
        f"Residual by phase — base MAPE {st_base['mape']:.1%} vs full {st_full['mape']:.1%} | "
        f"delta {delta:.1%} ({win}; want >3pp) | full cover {st_full['frac']:.1%} ({cov_tag})"
    )
    out1, out2, _all = derive_outputs(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(out1)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out2)) or ".", exist_ok=True)
    plot_actual_vs_pred_with_base(actual, base, full, out1, footnote=foot, title=title1)
    plot_residual_by_phase(actual, base, full, out2, footnote=foot, title=title2)
    # compat: also write literal --out as copy of fig1 if distinct
    if os.path.abspath(a.out) not in (os.path.abspath(out1), os.path.abspath(out2)):
        import shutil

        shutil.copyfile(out1, a.out)
    print(f"run: {run_dir}")
    print(f"actual bins: {len(actual)}  pred bins: {len(full)} (base {len(base)})  {fb_note}")
    print(f"full cover: {st_full['cover']}/{st_full['total']} ({st_full['frac']:.3f}) [{cov_tag}]")
    print(f"base cover: {st_base['cover']}/{st_base['total']} ({st_base['frac']:.3f})")
    print(f"base bin MAPE {st_base['mape']:.4f} vs full {st_full['mape']:.4f} delta {delta:.4f} [{win}]")
    print(f"wrote: {out1}")
    print(f"wrote: {out2}")
    if os.path.abspath(a.out) not in (os.path.abspath(out1), os.path.abspath(out2)):
        print(f"wrote: {a.out} (compat copy of fig1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
