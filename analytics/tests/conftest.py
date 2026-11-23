"""Shared fixtures: synthetic cells (make_cell) and the real_data skip guard."""
import csv
import gzip
import json
from pathlib import Path

import pytest

import settings

V1_META = {
    "task": "image", "model": "yolo11n.pt", "modes": ["forward"], "imgsz": 640, "batch": 1,
    "threads": 4, "cpu_mhz_pinned": 2400, "kernel": "6.18.50+rpt-rpi-2712",
    "t0_real_ns": 1_000_000_000, "serve_fps": None, "soak_minutes": None, "concurrent": None,
}
V2_META = {
    **V1_META, "kernel": "6.18.60+rpt-rpi-2712", "variant": "yolo11n", "quant": "fp32",
    "executed_dtype": "float32", "quantized_modules": None, "governor": "pinned",
    "fan_mode": "full", "holdout": None, "dataset_version": 2,
    "perf_counters": {"enabled": True, "interval_ms": 100, "offset_ns": 0, "events": []},
}
INDEX_COLS = ["cell_id", "model_key", "task", "cpu_mhz", "repeat", "family"]


def make_cell(tmp_path, source, cell_id, meta=None, phase_rows=None, alignment=None,
              derived=None, files=None, index=None):
    """Write a minimal synthetic cell under tmp_path/<source>/cells/<cell_id>.

    meta: overrides merged on the v1 or v2 defaults (a key set to the sentinel
    "__drop__" is removed). phase_rows: list of dicts -> phase_energy.csv.
    alignment / derived: dicts -> alignment.json / derived.json (derived None -> no file).
    files: {name: str|bytes|dict}; names ending .gz are gzipped, dicts become JSON.
    index: overrides for the dataset_index.csv row (a row is always appended).
    Returns the cell directory.
    """
    cell = Path(tmp_path) / source / "cells" / cell_id
    cell.mkdir(parents=True, exist_ok=True)
    m = dict(V1_META if source == "v1" else V2_META)
    for k, v in (meta or {}).items():
        if v == "__drop__":
            m.pop(k, None)
        else:
            m[k] = v
    (cell / "run_meta.json").write_text(json.dumps(m))
    if phase_rows:
        cols = list(phase_rows[0])
        with open(cell / "phase_energy.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(phase_rows)
    if alignment is not None:
        (cell / "alignment.json").write_text(json.dumps(alignment))
    if derived is not None:
        (cell / "derived.json").write_text(json.dumps(derived))
    for name, content in (files or {}).items():
        if isinstance(content, dict):
            content = json.dumps(content)
        data = content.encode() if isinstance(content, str) else content
        if name.endswith(".gz"):
            data = gzip.compress(data)
        (cell / name).write_bytes(data)
    row = {"cell_id": cell_id, "model_key": m.get("variant", "yolo11n").split("_")[0],
           "task": m["task"], "cpu_mhz": m["cpu_mhz_pinned"], "repeat": 0, "family": "A"}
    if source == "v1":
        row.pop("family")
    row.update(index or {})
    idx = Path(tmp_path) / source / "dataset_index.csv"
    new = not idx.exists()
    cols = list(row) if new else idx.read_text().splitlines()[0].split(",")
    with open(idx, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
    return cell


@pytest.fixture
def real_data():
    root = settings.data_root()
    if not (root / "v2" / "cells").is_dir():
        pytest.skip("real dataset not present")
    return root
