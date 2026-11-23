"""Static (a priori) model features from a layer table. No torch: the same function reads a Pi
`layer_table.csv` and the one `tools/static_features.py` writes for a not-yet-measured model."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import data

GROUPS = ["conv", "linear", "norm", "act", "pool", "other"]
CLASS_GROUPS = {
    **dict.fromkeys(["Conv2d", "ConvTranspose2d"], "conv"),
    "Linear": "linear",
    **dict.fromkeys(["BatchNorm2d", "BatchNorm1d", "LayerNorm", "GroupNorm"], "norm"),
    **dict.fromkeys(["SiLU", "ReLU", "ReLU6", "Hardswish", "Hardsigmoid", "Sigmoid", "GELU", "Tanh"],
                    "act"),
    **dict.fromkeys(["MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"], "pool"),
}
MACS_AGREE_TOL = 1e-3   # candidate cells of one (model, imgsz) must agree on macs within 0.1 %


def _is_leaf(s: pd.Series) -> pd.Series:
    """`leaf` is a bool in memory but the string 'True'/'False' once read back from CSV."""
    return s.map(lambda v: v is True or str(v).strip().lower() in ("true", "1"))


def features_from_layer_table(lt: pd.DataFrame, batch: int = 1) -> dict:
    """Per-image static features over the leaf rows of a layer table (see Task 4 brief).

    A row's per-inference multiplicity is round(count / min_count), min_count being the smallest
    positive leaf count; MACs and activation bytes are divided by `batch`, params are not."""
    leaf = lt[_is_leaf(lt["leaf"])].copy()
    count = pd.to_numeric(leaf["count"], errors="coerce").fillna(0)
    pos = count[count > 0]
    mult = (count / pos.min()).round() if len(pos) else count * 0
    macs_row = pd.to_numeric(leaf["macs"], errors="coerce").fillna(0) * mult
    params_row = pd.to_numeric(leaf["params"], errors="coerce").fillna(0)
    bm = pd.to_numeric(leaf["bytes_moved"], errors="coerce").fillna(0)
    act_row = (bm - 4 * params_row).clip(lower=0) * mult
    group = leaf["class"].map(lambda c: CLASS_GROUPS.get(c, "other"))
    uniq = leaf.drop_duplicates("path")
    params = float(pd.to_numeric(uniq["params"], errors="coerce").fillna(0).sum())
    total = float(macs_row.sum())
    out = {
        "macs": total / batch,
        "params": params,
        "weight_bytes": 4 * params,
        "act_bytes": float(act_row.sum()) / batch,
        "depth": int(leaf["path"].astype(str).str.count(r"\.").max()) + 1 if len(leaf) else 0,
        "n_leaf": int(len(uniq)),
    }
    upath_group = uniq["class"].map(lambda c: CLASS_GROUPS.get(c, "other"))
    for g in GROUPS:
        out[f"n_{g}"] = int((upath_group == g).sum())
    for g in GROUPS:
        out[f"frac_macs_{g}"] = float(macs_row[group == g].sum()) / total if total > 0 else math.nan
    return out


def static_table(cat: pd.DataFrame) -> pd.DataFrame:
    """Static features per (base_model, imgsz) from FP32 batch-1 Mode A cells that have a
    layer_table.csv. The first cell by cell_id supplies the row; every candidate must agree on
    `macs` within 0.1 % (else ValueError). No QC input: selection is by configuration only."""
    sel = data.training_cells(cat)
    sel = sel[sel["mode_a_eligible"].astype(bool) & (sel["precision"] == "fp32")
              & (sel["batch"] == 1)].sort_values("cell_id")
    feats: dict = {}
    for r in sel.itertuples():
        lt = data.read_csv_opt(Path(r.path), "layer_table.csv")
        if lt is None:
            continue
        f = features_from_layer_table(lt, batch=1)
        key = (r.base_model, int(r.imgsz))
        if key not in feats:
            feats[key] = {"cell_id": r.cell_id, **f}
            continue
        ref = feats[key]
        if abs(f["macs"] - ref["macs"]) > MACS_AGREE_TOL * max(abs(ref["macs"]), 1.0):
            raise ValueError(f"layer tables of {key} disagree on macs: {ref['cell_id']}="
                             f"{ref['macs']} vs {r.cell_id}={f['macs']}")
    st = pd.DataFrame.from_dict(feats, orient="index")
    st.index = pd.MultiIndex.from_tuples(st.index, names=["base_model", "imgsz"])
    return st.sort_index()


def model_descriptor(st: pd.DataFrame) -> pd.DataFrame:
    """Per base model: log_params, log_macs_per_px, depth, frac_macs_* at imgsz 640 if present,
    else the model's only imgsz (ValueError if several and no 640)."""
    rows = {}
    for model, sub in st.groupby(level="base_model"):
        sizes = sub.index.get_level_values("imgsz")
        if 640 in sizes:
            size = 640
        elif len(sizes) == 1:
            size = int(sizes[0])
        else:
            raise ValueError(f"{model}: several imgsz {sorted(sizes)} and none is 640")
        r = sub.loc[(model, size)]
        rows[model] = {
            "log_params": float(np.log(r["params"])),
            "log_macs_per_px": float(np.log(r["macs"] / size ** 2)),
            "depth": r["depth"],
            **{f"frac_macs_{g}": r[f"frac_macs_{g}"] for g in GROUPS},
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "base_model"
    return out.sort_index()
