"""Mode A configuration table: one row per measured configuration (mean energy/latency per image
and the static model features), the training and evaluation data of the a priori predictor."""
import numpy as np
import pandas as pd

import data
import settings

CONFIG_COLS = ["base_model", "variant", "precision", "imgsz", "batch", "threads", "cpu_mhz", "rig"]
MEASURES = {"energy_mJ": "energy_mJ_per_image", "latency_ms": "latency_ms_per_image",
            "mean_P_W": "mean_P_W", "dyn_energy_mJ": "dyn_energy_mJ_per_image"}
MODE = "forward"


def config_key(row) -> str:
    return "|".join(str(row[c]) for c in CONFIG_COLS)


def _cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean()) if len(x) >= 2 else float("nan")


def dvfs_table() -> tuple[np.ndarray, np.ndarray]:
    """(clocks in MHz ascending, VDD_CORE in V) from the `[dvfs]` section of config.toml."""
    d = settings.cfg()["dvfs"]
    mhz = sorted(d, key=float)
    return np.array([float(k) for k in mhz]), np.array([float(d[k]) for k in mhz])


def vcore_v(cpu_mhz) -> np.ndarray:
    """Core voltage at `cpu_mhz` by linear interpolation of the DVFS table, clamped outside it."""
    mhz, volt = dvfs_table()
    return np.interp(np.asarray(cpu_mhz, dtype=float), mhz, volt)


def add_derived_columns(t: pd.DataFrame) -> pd.DataFrame:
    """Copy of `t` plus the derived Mode A columns: log_{macs,params,weight_bytes,act_bytes}, cpu_ghz,
    is_int8, is_fp16, and the R2 columns vcore_V (DVFS table), v2f = vcore_V**2 * cpu_ghz and
    log_n_calls = log(n_module_calls). Shared by `mode_a_table` and the lockbox feature rows so both are
    built identically."""
    t = t.copy()
    for k in ("macs", "params", "weight_bytes", "act_bytes"):
        t[f"log_{k}"] = np.log(t[k].astype(float))
    t["cpu_ghz"] = t["cpu_mhz"].astype(float) / 1000.0
    t["vcore_V"] = vcore_v(t["cpu_mhz"].astype(float))
    t["v2f"] = t["vcore_V"] ** 2 * t["cpu_ghz"]
    t["log_n_calls"] = np.log(t["n_module_calls"].astype(float))
    t["is_int8"] = (t["precision"] == "int8").astype(int)
    t["is_fp16"] = (t["precision"] == "fp16").astype(int)
    return t


def mode_a_table(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration over Mode A eligible, QC-primary cells (mode `forward`, primary
    slip-excluded values; cells with no finite energy are dropped). Raises ValueError naming a
    configuration whose (base_model, imgsz) has no static features in `st`."""
    ok = qc.loc[qc["primary"].astype(bool), "cell_id"]
    c = data.training_cells(cat)
    c = c[c["mode_a_eligible"].astype(bool) & c["cell_id"].isin(ok)]
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
    t = add_derived_columns(t)
    return t.reset_index(drop=True)


def noise_floor(t: pd.DataFrame) -> float:
    """Median energy_cv over configurations with at least two cells (NaN if there are none)."""
    sub = t.loc[t["n_cells"] >= 2, "energy_cv"]
    return float(sub.median()) if len(sub) else float("nan")
