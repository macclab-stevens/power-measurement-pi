"""Lever: does removing harness I^2R close the oct->nov rig gap on identical configs?
Identical configs: yolo11n / yolov8n, 640 px, 4 threads, fp32, 1500/2000/2400 MHz on v1 (oct) and v2 v1ref (nov)."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, ".")
import data
R = {"oct2026": float(sys.argv[1]), "nov2026": float(sys.argv[2])}
cat = data.catalogue()
sel = cat[cat.base_model.isin(["yolo11n", "yolov8n"]) & (cat.imgsz == 640) & (cat.threads == 4) & (cat.precision == "fp32")
          & (cat.batch == 1) & cat.cpu_mhz.isin([1500, 2000, 2400]) & (cat.governor == "pinned") & (cat.fan_mode == "full")
          & (cat.source.isin(["v1", "v2"]))]
sel = sel[sel.cell_id.str.startswith("v1ref") | (sel.source == "v1")]
sel = sel[sel.mode_a_eligible.astype(bool)]
rows = []
for r in sel.itertuples():
    tr = pq.read_table(f"{r.path}/power_trace.parquet", columns=["V_V", "I_A", "phase", "iter"]).to_pandas()
    for ph in ("forward", "baseline_A"):
        m = tr["phase"] == ph
        if ph == "forward":
            m &= pd.to_numeric(tr["iter"], errors="coerce") >= 1
        V, I = tr.loc[m, "V_V"].to_numpy(), tr.loc[m, "I_A"].to_numpy()
        rows.append({"model": r.base_model, "mhz": r.cpu_mhz, "rig": r.rig, "phase": ph,
                     "P_raw": float((V * I).mean()), "P_board": float((V * I).mean() - R[r.rig] * (I * I).mean())})
x = pd.DataFrame(rows).groupby(["phase", "model", "mhz", "rig"])[["P_raw", "P_board"]].mean().unstack("rig")
for c in ("P_raw", "P_board"):
    x[(c, "nov-oct")] = x[(c, "nov2026")] - x[(c, "oct2026")]
print(x.round(3).to_string())
for c in ("P_raw", "P_board"):
    d = x[(c, "nov-oct")]
    print(c, "gap: mean", round(d.mean(), 3), "W, mean |gap|", round(d.abs().mean(), 3), "W")
