"""Mode A configuration table: one row per measured configuration (mean energy/latency per image
and the static model features), the training and evaluation data of the a priori predictor."""
import numpy as np
import pandas as pd

import data
import settings

CONFIG_COLS = ["base_model", "variant", "precision", "imgsz", "batch", "threads", "cpu_mhz", "rig"]
MEASURES = {"energy_mJ": "energy_mJ_per_image", "latency_ms": "latency_ms_per_image",
            "mean_P_W": "mean_P_W", "dyn_energy_mJ": "dyn_energy_mJ_per_image",
            "energy_board_mJ": "energy_board_mJ_per_image", "mean_P_board_W": "mean_P_board_W"}   # board = minus harness I^2 R
BOARD_COLS = ["energy_board_mJ_per_image", "mean_P_board_W"]
MODE = "forward"


def config_key(row) -> str:
    """CONFIG_COLS joined by "|"; a hooks-off cell (v3, `layer_hooks` "off") gets "|hooks=off" appended, so every
    key from before v3 (all hooks on) is unchanged."""
    k = "|".join(str(row[c]) for c in CONFIG_COLS)
    hooks = row["layer_hooks"] if "layer_hooks" in row else "on"
    return k + "|hooks=off" if str(hooks) == "off" else k


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
    calls = t["n_module_calls"].astype(float)
    if len(calls) and not (np.isfinite(calls) & (calls > 0)).all():
        raise ValueError("n_module_calls must be finite and positive (log_n_calls); got "
                         f"{calls[~(np.isfinite(calls) & (calls > 0))].unique()[:3].tolist()}")
    t["log_n_calls"] = np.log(calls)
    # R3: the measured cells paid the layer-hook cost only when the hooks were on (every cell before v3)
    hooks = t["layer_hooks"] if "layer_hooks" in t.columns else pd.Series("on", index=t.index)
    t["hooks_on"] = (hooks.astype(str) != "off").astype(int)
    t["hooks_log_calls"] = t["hooks_on"] * t["log_n_calls"]
    t["is_int8"] = (t["precision"] == "int8").astype(int)
    t["is_fp16"] = (t["precision"] == "fp16").astype(int)
    return t


def mode_a_table(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame, st: pd.DataFrame,
                 holdouts: tuple | None = None) -> pd.DataFrame:
    """One row per configuration over Mode A eligible, QC-primary cells (mode `forward`, primary
    slip-excluded values; cells with no finite energy are dropped). Raises ValueError naming a
    configuration whose (base_model, imgsz) has no static features in `st`. `holdouts` (R4) replaces
    data.HOLDOUT_SOURCES as the sources left out; the caller then sets `mode_a_eligible` itself."""
    ok = qc.loc[qc["primary"].astype(bool), "cell_id"]
    c = data.training_cells(cat) if holdouts is None else cat[~cat["source"].isin(holdouts)]
    c = c[c["mode_a_eligible"].astype(bool) & c["cell_id"].isin(ok)]
    d = der[der["mode"] == MODE]
    for col in BOARD_COLS:                   # a derivation table from before the board columns gives NaN
        if col not in d.columns:
            d = d.assign(**{col: np.nan})
    d = d[["cell_id", *MEASURES.values()]]
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
             "layer_hooks": first["layer_hooks"] if "layer_hooks" in first else "on",
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


def board_view(t: pd.DataFrame) -> pd.DataFrame:
    """The configuration table with the board-level labels in place of the Monsoon ones: `energy_mJ` is
    `energy_board_mJ`, `mean_P_W` is `mean_P_board_W` (latency unchanged), and the configurations where either
    is NaN are dropped. Raises ValueError if the table has no board columns (it predates them)."""
    need = ["energy_board_mJ", "mean_P_board_W"]
    if any(c not in t.columns for c in need):
        raise ValueError(f"the configuration table has no {need} columns; re-run `uv run python cli.py tables`")
    b = t.assign(energy_mJ=t["energy_board_mJ"], mean_P_W=t["mean_P_board_W"])
    return b.dropna(subset=["energy_mJ", "mean_P_W"]).reset_index(drop=True)


def noise_floor(t: pd.DataFrame) -> float:
    """Median energy_cv over configurations with at least two cells (NaN if there are none)."""
    sub = t.loc[t["n_cells"] >= 2, "energy_cv"]
    return float(sub.median()) if len(sub) else float("nan")
