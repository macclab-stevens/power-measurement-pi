"""Cell discovery and readers. Optional files return None; run_meta.json is the truth."""
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import settings

SOURCES = ("v1", "v2", "lockbox")
VISION_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m",
                 "effb0", "mbv3", "resnet18"]  # Mode A groups; splits.VISION re-exports this


def cell_dirs(root=None) -> list[Path]:
    """root/{v1,v2,lockbox}/cells/* directories (missing sources skipped), sorted."""
    root = Path(root) if root else settings.data_root()
    out: list[Path] = []
    for src in SOURCES:
        d = root / src / "cells"
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.is_dir())
    return out


def read_meta(cell) -> dict:
    return json.loads((Path(cell) / "run_meta.json").read_text())


def read_json_opt(cell, name) -> dict | None:
    p = Path(cell) / name
    return json.loads(p.read_text()) if p.is_file() else None


def read_csv_opt(cell, name) -> pd.DataFrame | None:
    """Read cell/name or cell/name.gz (compression inferred); None if neither exists."""
    for cand in (name, name + ".gz"):
        p = Path(cell) / cand
        if p.is_file():
            return pd.read_csv(p)
    return None


def read_trace(cell, columns) -> pd.DataFrame:
    """Column-projected read of power_trace.parquet; string columns come back as plain str."""
    tbl = pq.read_table(Path(cell) / "power_trace.parquet", columns=list(columns))
    for i, f in enumerate(tbl.schema):
        if pa.types.is_dictionary(f.type):
            tbl = tbl.set_column(i, f.name, tbl.column(i).cast(f.type.value_type))
    return tbl.to_pandas()


def precision_of(quant, executed_dtype) -> str:
    quant = quant or "fp32"
    if str(quant).startswith("int8"):
        return "int8"
    if quant == "fp16" or executed_dtype == "float16":
        return "fp16"
    return "fp32"


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


def _n_modules(qm) -> int:
    if qm is None:
        return 0
    return len(qm) if isinstance(qm, (list, tuple, dict)) else int(qm)


def _load_indexes(root: Path) -> dict:
    idx = {}
    for src in SOURCES:
        p = root / src / "dataset_index.csv"
        if p.is_file():
            df = pd.read_csv(p, dtype={"cell_id": str})
            idx[src] = df.drop_duplicates("cell_id", keep="last").set_index("cell_id")
    return idx


def _row(cell: Path, src: str, ix: pd.Series, m: dict) -> dict:
    quant = m.get("quant") or "fp32"
    base = ix["model_key"]
    pinned = m.get("cpu_mhz_pinned")
    pc = m.get("perf_counters") or {}
    conc = m.get("concurrent")
    serve, soak = _nan(m.get("serve_fps")), _nan(m.get("soak_minutes"))
    conc_mode = conc.get("mode") if isinstance(conc, dict) else None
    governor, fan = m.get("governor") or "pinned", m.get("fan_mode") or "full"
    has_counters = (cell / "counters.csv").is_file() and bool(pc.get("enabled"))
    task = m["task"]
    return {
        "cell_id": cell.name, "source": src, "path": str(cell), "task": task,
        "model_key": base, "base_model": base, "variant": m.get("variant") or base,
        "precision": precision_of(quant, m.get("executed_dtype")), "quant": quant,
        "n_quantized_modules": _n_modules(m.get("quantized_modules")),
        "imgsz": m.get("imgsz"), "batch": m.get("batch") or 1, "threads": m.get("threads") or 4,
        "cpu_mhz": int(pinned if pinned is not None else ix["cpu_mhz"]),
        "governor": governor, "fan_mode": fan, "serve_fps": serve, "soak_minutes": soak,
        "concurrent_mode": conc_mode,
        "input_source": (m.get("inputs") or {}).get("source") or "repeat",
        "family": "v1" if src == "v1" else ix["family"],
        "repeat": int(ix["repeat"]),
        "rig": "oct2026" if str(m.get("kernel", "")).startswith("6.18.50") else "nov2026",
        "dataset_version": int(m.get("dataset_version") or 1),
        "holdout": m.get("holdout"),
        "has_counters": has_counters,
        "mode_a_eligible": bool(task == "image" and base in VISION_MODELS and math.isnan(serve)
                                and math.isnan(soak) and conc is None and governor == "pinned"
                                and fan == "full"),
        "mode_l_eligible": has_counters,
    }


CATALOGUE_COLUMNS = [
    "cell_id", "source", "path", "task", "model_key", "base_model", "variant", "precision", "quant",
    "n_quantized_modules", "imgsz", "batch", "threads", "cpu_mhz", "governor", "fan_mode",
    "serve_fps", "soak_minutes", "concurrent_mode", "input_source", "family", "repeat", "rig",
    "dataset_version", "holdout", "has_counters", "mode_a_eligible", "mode_l_eligible"]


def catalogue(root=None) -> pd.DataFrame:
    """One row per cell (reads run_meta.json and the dataset_index.csv files only)."""
    root = Path(root) if root else settings.data_root()
    idx = _load_indexes(root)
    rows = []
    for cell in cell_dirs(root):
        src = cell.parent.parent.name
        table = idx.get(src)
        if table is None or cell.name not in table.index:
            raise KeyError(f"cell {cell.name} ({src}) is missing from {src}/dataset_index.csv")
        rows.append(_row(cell, src, table.loc[cell.name], read_meta(cell)))
    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)
