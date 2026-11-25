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


def n_module_calls(lt: pd.DataFrame, batch: int = 1) -> float:
    """Per-image count of module (forward-hook) calls: the sum over ALL rows, containers and leaves, of
    the multiplicity round(count / min_positive_leaf_count), divided by `batch`. This is the number of
    hook invocations a measured (instrumented) cell paid for per image. The multiplicity unit is the
    smallest positive leaf count, as for the MAC features (the smallest positive count over all rows if
    there are no leaf rows). 0.0 when no row has a positive count."""
    count = pd.to_numeric(lt["count"], errors="coerce").fillna(0)
    ref = count[_is_leaf(lt["leaf"]) & (count > 0)]
    if not len(ref):
        ref = count[count > 0]
    if not len(ref):
        return 0.0
    return float((count / ref.min()).round().sum()) / batch


def features_from_layer_table(lt: pd.DataFrame, batch: int = 1) -> dict:
    """Per-image static features over the leaf rows of a layer table (see Task 4 brief).

    A row's per-inference multiplicity is round(count / min_count), min_count being the smallest
    positive leaf count; MACs and activation bytes are divided by `batch`, params are not."""
    is_leaf = _is_leaf(lt["leaf"])
    leaf = lt[is_leaf].copy()
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
        "n_module_calls": n_module_calls(lt, batch),
    }
    upath_group = uniq["class"].map(lambda c: CLASS_GROUPS.get(c, "other"))
    for g in GROUPS:
        out[f"n_{g}"] = int((upath_group == g).sum())
    for g in GROUPS:
        out[f"frac_macs_{g}"] = float(macs_row[group == g].sum()) / total if total > 0 else math.nan
    return out


def static_layer_tables(static_dir=None) -> dict:
    """{(base_model, imgsz): layer table} from `static_dir`/<base_model>_<imgsz>.csv (default analytics/static_v3/):
    layer tables built on the NUC by tools/static_features.py for configurations whose cells ran with the hooks off and
    so wrote none. A missing folder gives {}."""
    d = Path(static_dir) if static_dir else Path(__file__).parent / "static_v3"
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*_*.csv")):
            m, s = p.stem.rsplit("_", 1)
            out[(m, int(s))] = pd.read_csv(p)
    return out


def static_table(cat: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    """Static features per (base_model, imgsz) from FP32 batch-1 Mode A cells that have a
    layer_table.csv. The first cell by cell_id supplies the row; every candidate must agree on
    `macs` within 0.1 % (else ValueError). No QC input: selection is by configuration only.
    `extra` ({(base_model, imgsz): layer table}, see `static_layer_tables`) fills configurations no cell covers;
    a cell's own layer table always wins."""
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
    for key, lt in (extra or {}).items():          # configurations whose cells have no layer table (hooks off)
        if key not in feats:
            feats[key] = {"cell_id": f"static:{key[0]}_{key[1]}", **features_from_layer_table(lt, batch=1)}
    st = pd.DataFrame.from_dict(feats, orient="index")
    st.index = pd.MultiIndex.from_tuples(st.index, names=["base_model", "imgsz"])
    return add_flops(st.sort_index())


FLOPS_CSV = Path(__file__).parent / "static_flops.csv"


def add_flops(st: pd.DataFrame, path=None) -> pd.DataFrame:
    """`st` (indexed by base_model, imgsz, with `macs`) plus macs_total (whole-model count, tools/flop_count.py),
    macs_attn = max(0, macs_total - macs) (matrix products the layer table misses: attention) and
    frac_attn = macs_attn / macs_total. With no static_flops.csv the three columns are absent (R1 / R2 tables);
    a configuration missing from an existing file raises ValueError."""
    p = Path(path) if path else FLOPS_CSV
    if not p.is_file():
        return st
    fl = pd.read_csv(p).set_index(["base_model", "imgsz"])["macs_total"]
    missing = [k for k in st.index if k not in fl.index]
    if missing:
        raise ValueError(f"no whole-model MAC count in {p.name} for {missing[:3]}; run tools/flop_count.py")
    st = st.copy()
    st["macs_total"] = [float(fl.loc[k]) for k in st.index]
    st["macs_attn"] = np.maximum(0.0, st["macs_total"] - st["macs"])
    st["frac_attn"] = st["macs_attn"] / st["macs_total"]
    return st


def model_descriptor(st: pd.DataFrame) -> pd.DataFrame:
    """Per base model: log_params, log_macs_per_px, depth, frac_macs_* at imgsz 640 if present,
    else 224, else the model's only imgsz (ValueError if several and neither)."""
    rows = {}
    for model, sub in st.groupby(level="base_model"):
        sizes = sub.index.get_level_values("imgsz")
        if 640 in sizes:
            size = 640
        elif 224 in sizes:                       # R4: the torchvision models are also measured at 320 px
            size = 224
        elif len(sizes) == 1:
            size = int(sizes[0])
        else:
            raise ValueError(f"{model}: several imgsz {sorted(sizes)} and none is 640 or 224")
        r = sub.loc[(model, size)]
        rows[model] = {
            "log_params": float(np.log(r["params"])),
            "log_macs_per_px": float(np.log(r["macs"] / size ** 2)),
            "depth": r["depth"],
            **{f"frac_macs_{g}": r[f"frac_macs_{g}"] for g in GROUPS},
            **({"frac_attn": float(r["frac_attn"])} if "frac_attn" in r.index else {}),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "base_model"
    if "frac_attn" in out.columns and out["frac_attn"].max() <= ATTN_VARY:
        out = out.drop(columns="frac_attn")       # no attention model: the R1 / R2 descriptor, unchanged
    return out.sort_index()


ATTN_VARY = 0.02    # frac_attn above this in some model: attention is present and enters descriptors and features
