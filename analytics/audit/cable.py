"""Lever: harness resistance from Monsoon V/I vs Pi PMIC EXT5V during each vcgencmd read window.
Per read: Monsoon V, I averaged over the ctx_read segment that starts nearest after the read's t_ms.
Per rig: OLS  V_mon - V_ext5v = b + R * I  (b absorbs a fixed ADC offset; R is the round-trip harness R)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, ".")
import data
rows = []
for cell in data.cell_dirs():
    meta = data.read_meta(cell)
    cx = data.read_csv_opt(cell, "context.csv")
    if cx is None or "rail_EXT5V_V volt(24)" not in cx.columns or meta.get("t0_real_ns") is None:
        continue
    tr = pq.read_table(cell / "power_trace.parquet", columns=["t_pi_ns", "V_V", "I_A", "ctx_read", "phase"]).to_pandas()
    f = tr["ctx_read"].to_numpy()
    if not f.any():
        continue
    d = np.diff(f.astype(int)); st = np.flatnonzero(d == 1) + 1; en = np.flatnonzero(d == -1) + 1
    if f[0]: st = np.r_[0, st]
    if f[-1]: en = np.r_[en, len(f)]
    t = tr["t_pi_ns"].to_numpy(); V = tr["V_V"].to_numpy(); I = tr["I_A"].to_numpy(); ph = tr["phase"].to_numpy()
    seg_t = t[st]
    tread = int(meta["t0_real_ns"]) + np.round(cx["t_ms"].to_numpy(float) * 1e6)
    for k, tr_ns in enumerate(tread):
        j = np.searchsorted(seg_t, tr_ns - 5e6)
        if j >= len(seg_t) or seg_t[j] - tr_ns > 50e6:
            continue
        a, b = st[j], en[j]
        rows.append({"cell_id": cell.name, "src": cell.parent.parent.name, "rig": meta.get("rig", ""),
                     "V_mon": V[a:b].mean(), "I": I[a:b].mean(), "I_sd": I[a:b].std(), "phase": ph[a],
                     "V_ext": cx["rail_EXT5V_V volt(24)"].iloc[k]})
x = pd.DataFrame(rows)
x.to_parquet(sys.argv[1])
x["rig2"] = np.where(x["src"] == "v1", "v1(5.25V)", np.where(x["src"] == "lockbox", "lockbox", "v2"))
for rig, g in x.groupby("rig2"):
    g = g[g["I_sd"] < 0.15]       # steady reads only
    A = np.c_[np.ones(len(g)), g["I"]]; coef, *_ = np.linalg.lstsq(A, g["V_mon"] - g["V_ext"], rcond=None)
    act = g[g["I"] > 1.6]
    print(f"{rig}: n={len(g)} R={coef[1]:.4f} ohm offset={coef[0]*1000:.1f} mV; active I={act.I.median():.3f} A "
          f"V_mon={act.V_mon.median():.3f} V_ext={act.V_ext.median():.3f}; I^2R={act.I.median()**2*coef[1]:.3f} W "
          f"({100*act.I.median()*coef[1]/act.V_mon.median():.2f}% of P); min V_ext={g.V_ext.min():.3f}")
