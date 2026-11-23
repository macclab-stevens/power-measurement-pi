"""Mode A configuration table: one row per measured configuration (mean energy/latency per image
and the static model features), the training and evaluation data of the a priori predictor."""
import numpy as np
import pandas as pd

CONFIG_COLS = ["base_model", "variant", "precision", "imgsz", "batch", "threads", "cpu_mhz", "rig"]
MEASURES = {"energy_mJ": "energy_mJ_per_image", "latency_ms": "latency_ms_per_image",
            "mean_P_W": "mean_P_W", "dyn_energy_mJ": "dyn_energy_mJ_per_image"}
MODE = "forward"


def config_key(row) -> str:
    return "|".join(str(row[c]) for c in CONFIG_COLS)


def _cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean()) if len(x) >= 2 else float("nan")


def mode_a_table(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration over Mode A eligible, QC-primary cells (mode `forward`, primary
    slip-excluded values; cells with no finite energy are dropped). Raises ValueError naming a
    configuration whose (base_model, imgsz) has no static features in `st`."""
    ok = qc.loc[qc["primary"].astype(bool), "cell_id"]
    c = cat[cat["mode_a_eligible"].astype(bool) & cat["cell_id"].isin(ok)]
    d = der[der["mode"] == MODE][["cell_id", *MEASURES.values()]]
    c = c.merge(d, on="cell_id", how="inner")
    c = c[np.isfinite(c["energy_mJ_per_image"].astype(float))].copy()
    c["config_key"] = [config_key(r) for _, r in c.iterrows()]
    feat_cols = [k for k in st.columns if k != "cell_id"]
    rows = []
    for key, g in c.sort_values(["config_key", "cell_id"]).groupby("config_key", sort=True):
        first = g.iloc[0]
        sk = (first["base_model"], int(first["imgsz"]))
        if sk not in st.index:
            raise ValueError(f"no static features for configuration {key}: no FP32 batch-1 layer table "
                             f"for (base_model, imgsz) = {sk}")
        r = {"config_key": key, **{k: first[k] for k in CONFIG_COLS},
             "quant": first["quant"], "n_quantized_modules": first["n_quantized_modules"],
             "n_cells": len(g), "cell_ids": ";".join(g["cell_id"])}
        for out, col in MEASURES.items():
            r[out] = float(g[col].mean())
        r["energy_cv"] = _cv(g["energy_mJ_per_image"])
        r["latency_cv"] = _cv(g["latency_ms_per_image"])
        r["family_set"] = ";".join(sorted(set(g["family"].astype(str))))
        srow = st.loc[sk]
        r.update({k: srow[k] for k in feat_cols})
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    for k in ("macs", "params", "weight_bytes", "act_bytes"):
        t[f"log_{k}"] = np.log(t[k].astype(float))
    t["cpu_ghz"] = t["cpu_mhz"].astype(float) / 1000.0
    t["is_int8"] = (t["precision"] == "int8").astype(int)
    t["is_fp16"] = (t["precision"] == "fp16").astype(int)
    return t.reset_index(drop=True)


def noise_floor(t: pd.DataFrame) -> float:
    """Median energy_cv over configurations with at least two cells (NaN if there are none)."""
    sub = t.loc[t["n_cells"] >= 2, "energy_cv"]
    return float(sub.median()) if len(sub) else float("nan")
