"""Lever: size of the sync-fit shear caused by mid-run sample-loss slips.
For each cell with slips: corrected time t' = t + corr(t), corr from the step model
(true offset steps by +step at each slip; the fitted line is exact at the start/end pulse trains).
Reports max |corr| in the forward segment, the forward energy-per-image change and 100 ms window P change."""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

def corr_fn(al, ev):
    pul = ev[ev.kind.isin(["pulse_on", "pulse_off"])].sort_values("t_real_ns")["t_real_ns"].to_numpy(float)
    if len(pul) < 12:
        return None
    ta, tb = pul[:6].mean(), pul[-6:].mean()
    c0 = al["c0_ns"]
    steps = [(s["t_ns"] - c0, s["step_ms"] * 1e6) for s in al["slips"]]
    H = lambda x: (x > 0).astype(float) if isinstance(x, np.ndarray) else float(x > 0)
    def corr(t):
        out = np.zeros_like(t)
        for ts, d in steps:
            ha, hb = H(ta - ts), H(tb - ts)
            out += d * (H(t - ts) - (ha + (hb - ha) * (t - ta) / (tb - ta)))
        return out
    return corr

rows = []
for al_f in sorted(glob.glob(str(Path.home() / "pi-power-data/*/cells/*/alignment.json"))):
    al = json.load(open(al_f)); cell = Path(al_f).parent
    if not al.get("slips"):
        continue
    ev = pd.read_csv(cell / "events.csv.gz", usecols=["t_real_ns", "kind"])
    corr = corr_fn(al, ev)
    if corr is None:
        continue
    tr = pq.read_table(cell / "power_trace.parquet", columns=["t_pi_ns", "P_W", "phase", "iter"]).to_pandas()
    t = tr["t_pi_ns"].to_numpy(float); P = tr["P_W"].to_numpy()
    c = corr(t); t2 = t + c
    # label at corrected time = label of the sample nearest t2 on the original time axis
    j = np.clip(np.searchsorted(t, t2), 0, len(t) - 1)
    lab_ph, lab_it = tr["phase"].to_numpy(), tr["iter"].to_numpy()
    it_num = pd.to_numeric(pd.Series(lab_it), errors="coerce").to_numpy()
    fwd0 = (lab_ph == "forward") & (it_num >= 1)
    fwd1 = fwd0[j]
    rec = {"cell": cell.parent.parent.name + "/" + cell.name, "n_slips": len(al["slips"]),
           "slip_ms": sum(s["step_ms"] for s in al["slips"]), "drift_ppm": al["drift_ppm"],
           "max_abs_corr_ms": float(np.abs(c).max() / 1e6)}
    if fwd0.any():
        n0 = len(set(lab_it[fwd0])); n1 = len(set(lab_it[j][fwd1]))
        e0 = P[fwd0].sum() / n0; e1 = P[fwd1].sum() / max(n1, 1)
        rec.update(fwd_max_corr_ms=float(np.abs(c[fwd0]).max() / 1e6), dE_img_pct=100 * (e1 - e0) / e0,
                   dP_fwd_pct=100 * (P[fwd1].mean() - P[fwd0].mean()) / P[fwd0].mean())
    # 100 ms windows on the original vs corrected axis (same grid)
    g0 = np.floor((t - t[0]) / 1e8).astype(int); g1 = np.floor((t2 - t[0]) / 1e8).astype(int)
    w0 = pd.Series(P).groupby(g0).mean(); w1 = pd.Series(P).groupby(g1).mean()
    common = w0.index.intersection(w1.index)
    dw = (w1[common] - w0[common]).abs()
    rec.update(win_dP_med_W=float(dw.median()), win_dP_p99_W=float(dw.quantile(0.99)), win_dP_max_W=float(dw.max()))
    rows.append(rec)
    print(rec["cell"], round(rec["max_abs_corr_ms"], 2), round(rec.get("dE_img_pct", np.nan), 4), flush=True, file=sys.stderr)
out = pd.DataFrame(rows)
out.to_csv(sys.argv[1], index=False)
print(out.describe(percentiles=[.5, .9, .99]).T.round(4).to_string())
