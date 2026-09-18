"""v2 explainability plots (Task 7).

Fig1 roofline-power: x=arith intensity (macs/bytes), y=E_ABAB (mJ) log-log,
  color=class (Okabe-Ito colorblind-safe), size~bytes. Memory/compute knees.
Fig2 per-class median+IQR: barh median, xerr=[med-Q1,Q3-med], sorted, n=.
  Two panels: raw P (W) vs log-residuals baseline-vs-full (honest T6 FAIL).
Fig3 waterfall-with-bounds: measured total vs stacked predicted per-class
  + hatched correction gap + bounds from per-class rmse. VDD_CORE/threads/
  governor/PSU footnote. No dual-axis, no mean+-std, barh zero baseline.

Usage: uv run --with matplotlib analysis/plots_v2.py \
  --run runs/v3_yolo11n_640 --out /tmp/v2_waterfall.png  (3 PNGs)
"""
import argparse
import csv
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
       "#D55E00", "#F0E442", "#999999"]
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

RAIL = "VDD_CORE"
SCHEMA = 3


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows(run_dir):
    if os.path.exists(os.path.join(run_dir, "_TAINTED_V2")):
        return []
    try:
        if json.load(open(os.path.join(run_dir, "run_meta.json"))).get("schema_version") != SCHEMA:
            return []
    except (OSError, ValueError):
        return []
    rows = []
    with open(os.path.join(run_dir, "perop_power.csv")) as f:
        for r in csv.DictReader(f):
            if str(r.get("SKIPPED")).strip() == "True":
                continue
            pa = _f(r.get("P_delta_ABAB_W")) or _f(r.get("P_delta_W"))
            tp = _f(r.get("t_per_call_ms"))
            e = (pa * tp) if (pa and tp and pa * tp > 0) else _f(r.get("E_per_call_mJ"))
            if not (pa and tp and e and e > 0):
                continue
            rows.append({"class": r.get("class") or "?", "macs": _f(r.get("macs")),
                         "bytes": _f(r.get("bytes_moved")), "t_eff": _f(r.get("t_eff")),
                         "P": pa, "t": tp, "E": e})
    return rows


def load_meta(run_dir):
    try:
        m = json.load(open(os.path.join(run_dir, "run_meta.json")))
    except (OSError, ValueError):
        m = {}
    threads = m.get("threads", 4)
    tb = m.get("throttle_bits", "?")
    tbh = ("0x%X" % tb) if isinstance(tb, int) else str(tb)
    return threads, tbh, m.get("schema_version", SCHEMA)


def load_scaling(repo_root, path):
    p = path if os.path.isabs(path) else os.path.join(repo_root, path)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def epred(coef, t, b):
    return math.exp(coef["a"] + coef["b_T"] * math.log(max(1, t))
                    + coef.get("c_B", coef.get("c_bytes")) * math.log(max(1, b)))


def arms(rows, scaling):
    pool_rmse = float(np.mean([v.get("rmse", 0.5) for v in scaling.values()])) if scaling else 0.5
    pool = None
    if scaling:
        a = np.mean([v["a"] for v in scaling.values()])
        bt = np.mean([v["b_T"] for v in scaling.values()])
        cb = np.mean([v.get("c_B", v.get("c_bytes")) for v in scaling.values()])
        pool = {"a": a, "b_T": bt, "c_B": cb, "rmse": pool_rmse}
    mean_p = float(np.mean([r["E"] / r["t"] for r in rows]))
    for r in rows:
        r["E_base"] = mean_p * r["t"]
        c = scaling.get(r["class"], pool) if scaling else None
        ok = c and r["t_eff"] and r["bytes"] and r["t_eff"] > 0 and r["bytes"] > 0
        r["E_full"] = epred(c, r["t_eff"], r["bytes"]) if ok else None
        r["rmse"] = c.get("rmse", pool_rmse) if ok else pool_rmse
    return mean_p, pool_rmse


def mape(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    return float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-12)))


def footnote(run, threads, tbh, schema):
    return ("src=%s | %s rail only | schema_version=%s | threads=%s | "
            "governor=ondemand (default; perf pin needs sudo) | PSU stock "
            "5V/5A throttle_bits=%s" % (run, RAIL, schema, threads, tbh))


def build_cmap(rows):
    return {c: PAL[i % len(PAL)] for i, c in enumerate(sorted({r["class"] for r in rows}))}


def fig_roofline(rows, path, note, cmap):
    pts = [r for r in rows if r["macs"] and r["bytes"] and r["macs"] > 0 and r["bytes"] > 0]
    cls = sorted({r["class"] for r in pts})
    fig, ax = plt.subplots(figsize=(8, 6))
    bmax = max(r["bytes"] for r in pts)
    for r in pts:
        ax.scatter(r["macs"] / r["bytes"], r["E"],
                   c=cmap[r["class"]], s=12 + 90 * math.log10(1 + 9 * r["bytes"] / bmax),
                   alpha=0.7, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (MACs/byte)")
    ax.set_ylabel("energy per call E_ABAB (mJ)")
    ax.set_title("Roofline-power: energy vs intensity (VDD_CORE, schema 3)")
    knee = float(np.median([r["macs"] / r["bytes"] for r in pts]))
    ax.axvline(knee, color="k", ls="--", lw=1)
    ax.text(knee, ax.get_ylim()[0] * 1.1, " median AI=%.1f" % knee, fontsize=7)
    ax.text(0.02, 0.94, "memory-bound <-", transform=ax.transAxes, fontsize=8)
    ax.text(0.78, 0.94, "-> compute-bound", transform=ax.transAxes, fontsize=8)
    ax.legend(handles=[mpatches.Patch(color=cmap[c], label=c) for c in cls],
              fontsize=7, ncol=2, loc="best")
    fig.text(0.01, 0.01, note, fontsize=6)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_iqr(rows, path, note, verdict, cmap):
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    order = sorted(by, key=lambda c: np.median([r["P"] for r in by[c]]))
    y = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, max(4, 0.6 * len(order) + 2)),
                                 sharey=True)
    med, lo, hi, ns = [], [], [], []
    for c in order:
        v = np.array([r["P"] for r in by[c]])
        m, q1, q3 = np.median(v), *np.percentile(v, [25, 75])
        med.append(m); lo.append(m - q1); hi.append(q3 - m); ns.append(len(v))
    a1.barh(y, med, xerr=[lo, hi], capsize=3, color=[cmap[c] for c in order], ecolor="k")
    a1.set_yticks(y); a1.set_yticklabels(["%s (n=%d)" % (c, n) for c, n in zip(order, ns)])
    a1.set_xlabel("P_delta_ABAB (W)"); a1.set_title("raw power: median+IQR")
    a1.set_xlim(left=0)
    bm, bl, bh, fm, fl, fh = [], [], [], [], [], []
    for c in order:
        rb = np.array([math.log(r["E_base"] / r["E"]) for r in by[c]])
        rf = np.array([math.log(r["E_full"] / r["E"]) for r in by[c] if r["E_full"]])
        for arr, out in ((rb, (bm, bl, bh)), (rf, (fm, fl, fh))):
            m, q1, q3 = np.median(arr), *np.percentile(arr, [25, 75])
            out[0].append(m); out[1].append(m - q1); out[2].append(q3 - m)
    h = 0.38
    a2.barh(y - h / 2, bm, xerr=[bl, bh], height=h, capsize=2, color=PAL[1],
            ecolor="k", label="baseline (meanP*t)")
    a2.barh(y + h / 2, fm, xerr=[fl, fh], height=h, capsize=2, color=PAL[2],
            ecolor="k", label="full (scaling-law)")
    a2.axvline(0, color="k", lw=1)
    a2.set_xlabel("log-residual log(E_pred/E_meas)"); a2.set_title("residuals: median+IQR")
    a2.legend(fontsize=7)
    fig.suptitle("Per-class power overlaps; %s" % verdict, fontsize=10)
    fig.text(0.01, 0.01, note, fontsize=6)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_waterfall(rows, path, note, mean_p, cmap):
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    cls = sorted(by, key=lambda c: -sum(r["E_full"] or 0 for r in by[c]))
    pred_c = [sum(r["E_full"] or 0 for r in by[c]) for c in cls]
    lo_c = [sum((r["E_full"] or 0) * math.exp(-r["rmse"]) for r in by[c]) for c in cls]
    hi_c = [sum((r["E_full"] or 0) * math.exp(r["rmse"]) for r in by[c]) for c in cls]
    meas, base = sum(r["E"] for r in rows), sum(r["E_base"] for r in rows)
    pred, lo, hi = sum(pred_c), sum(lo_c), sum(hi_c)
    gap = meas - pred
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["predicted (full)", "measured", "baseline (meanP*t)", "gap |meas-pred|"]
    ax.barh([labels[1]], [meas], color="#333333")
    ax.barh([labels[2]], [base], color=PAL[1])
    left = 0
    for c, v in zip(cls, pred_c):
        ax.barh([labels[0]], [v], left=left,
                color=cmap[c], label="%s %.1f mJ" % (c, v))
        left += v
    ax.errorbar([pred], [labels[0]], xerr=[[pred - lo], [hi - pred]],
                fmt="none", ecolor="k", capsize=4)
    ax.barh([labels[3]], [abs(gap)], color="none", edgecolor="#D55E00", hatch="///")
    ax.set_xlabel("forward energy (mJ, VDD_CORE)"); ax.set_xlim(left=0)
    ax.set_title("Waterfall-with-bounds: predicted stack vs measured (hatched gap)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.text(0.01, 0.01, note + " | totals mJ: meas=%.1f pred=%.1f [%.1f,%.1f] "
             "base=%.1f gap=%+.1f" % (meas, pred, lo, hi, base, gap), fontsize=6)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="waterfall PNG path; siblings derived")
    ap.add_argument("--scaling", default="modeling/scaling.json")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = a.run if os.path.isabs(a.run) else os.path.join(repo, a.run)
    rows = load_rows(run)
    if not rows:
        raise SystemExit("no valid rows in %s" % run)
    threads, tbh, schema = load_meta(run)
    if schema != SCHEMA:
        raise SystemExit("schema_version=%r want %d (run=%s)" % (schema, SCHEMA, run))
    scaling = load_scaling(repo, a.scaling)
    mean_p, pool_rmse = arms(rows, scaling)
    ok = [r for r in rows if r["E_full"]]
    mb, mf = (mape([r["E"] for r in ok], [r["E_base"] for r in ok]) if ok else float("nan"),
              mape([r["E"] for r in ok], [r["E_full"] for r in ok]) if ok else float("nan"))
    verdict = ("residuals NOT shrunk: MAPE base=%.3f (n=%d) full=%.3f (n=%d) "
               "(T6 0/7 FAIL honest)" % (mb, len(ok), mf, len(ok)))
    note = footnote(a.run, threads, tbh, schema) + " | bounds pooled fallback rmse=%.3f" % pool_rmse
    cmap = build_cmap(rows)
    d, b = os.path.split(os.path.abspath(a.out))
    if "waterfall" in b:
        pr, pi = b.replace("waterfall", "roofline"), b.replace("waterfall", "iqr")
    else:
        pr, pi = "roofline_" + b, "iqr_" + b
    os.makedirs(d or ".", exist_ok=True)
    pr, pi = os.path.join(d, pr), os.path.join(d, pi)
    fig_roofline(rows, pr, note, cmap)
    fig_iqr(rows, pi, note + " | " + verdict, verdict, cmap)
    fig_waterfall(rows, os.path.abspath(a.out), note, mean_p, cmap)
    print("wrote:\n%s\n%s\n%s" % (pr, pi, os.path.abspath(a.out)))


if __name__ == "__main__":
    main()
