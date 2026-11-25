"""Lever: (1) effect of the vcgencmd reads (ctx_read) on forward energy per image if they were masked
(mean power over non-ctx samples x full duration); (2) VDD_CORE_V per pinned frequency from context.csv."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, ".")
import data
cat = data.catalogue()
cat = cat[cat.mode_a_eligible.astype(bool)]
rows, vc = [], []
for r in cat.itertuples():
    tr = pq.read_table(f"{r.path}/power_trace.parquet", columns=["P_W", "phase", "iter", "ctx_read"]).to_pandas()
    m = (tr["phase"] == "forward") & (pd.to_numeric(tr["iter"], errors="coerce") >= 1)
    P, c = tr.loc[m, "P_W"].to_numpy(), tr.loc[m, "ctx_read"].to_numpy()
    if len(P) and (~c).any():
        rows.append({"cell_id": r.cell_id, "ctx_frac": c.mean(), "dE_pct": 100 * (P[~c].mean() - P.mean()) / P.mean(),
                     "dP_ctx_W": (P[c].mean() - P[~c].mean()) if c.any() else np.nan})
    cx = data.read_csv_opt(r.path, "context.csv")
    if cx is not None and "rail_VDD_CORE_V volt(15)" in cx.columns and r.governor == "pinned":
        vc.append({"mhz": r.cpu_mhz, "rig": r.rig, "v": cx["rail_VDD_CORE_V volt(15)"].median()})
x = pd.DataFrame(rows)
print("forward ctx_read: frac median", round(x.ctx_frac.median(), 4), "| masking changes mean power by median",
      round(x.dE_pct.median(), 4), "%, p1/p99", x.dE_pct.quantile([.01, .99]).round(4).tolist(), "% | P(ctx)-P(other) median",
      round(x.dP_ctx_W.median(), 3), "W")
v = pd.DataFrame(vc)
print(v.groupby(["rig", "mhz"])["v"].describe()[["count", "50%", "min", "max"]].round(4))
