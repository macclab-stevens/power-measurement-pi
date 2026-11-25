#!/usr/bin/env python3
"""Phase 4: package the finished dataset-v2 cells into a publishable folder (EliteBook, host env).

    uv run python package_dataset.py --runs ../runs/v2 --manifest ../runs/v2/manifest.jsonl \\
        --out ../dataset-stage/data/monsoon_v2_2026-11 --name monsoon_v2_2026-11
    ... --dry-run     # index / coverage / excluded / readiness / README only; no trace is copied

What it does
  * keeps the cells whose LAST results.jsonl row is `status == "ok"`; every other manifest cell
    (failed, or never run) goes to excluded.csv with its last stage, last error line and attempts;
  * per kept cell, one at a time (the EliteBook has 7 GB and runs the recorder): copies the small
    files, repacks power_trace.parquet (zstd level 9, streamed in row slices, never the whole
    table in memory) and events.csv -> events.csv.gz, then VERIFIES both against the originals
    (parquet: same schema, row count and values; gz: decompresses to the identical bytes) and
    stops with an error on any mismatch. A cell appears in the output only after it verified;
  * writes dataset_index.csv, coverage.csv, excluded.csv, readiness.md, README.md, and a copy of
    the manifest and results log.

Needs pyarrow and numpy only (no pandas). Never touches the Pi, the recorder or src/run.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import json
import re
import shutil
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from matrix_v2 import holdout_tag

ZSTD_LEVEL = 9
MAX_GROUP_ROWS = 2_097_152          # source row groups above this are read with iter_batches
ROW_SLICE = 262_144                 # rows per streamed slice / output row group (~80 MB at 38 columns)
COUNTER_COVERAGE_MIN = 0.95         # verify_cell.COUNTER_COVERAGE_MIN
ALIGN_LIMIT_MS = 1.0                # alignment's fatal residual limit
THERMAL_ABORT_C = 85.0
NO_QUANT = (None, "", "fp32")

REQUIRED_FILES = ("run_meta.json", "alignment.json", "phase_energy.csv", "derived.json",
                  "layer_table.csv", "op_table.csv", "perop_power.csv", "context.csv",
                  "sysfs.csv", "power_trace.parquet", "events.csv")
COPIED_FILES = ("run_meta.json", "alignment.json", "phase_energy.csv", "derived.json",
                "layer_table.csv", "op_table.csv", "perop_power.csv", "context.csv", "sysfs.csv")
OPTIONAL_FILES = ("counters.csv", "thermal_abort.json")
# the orchestrator keeps the Pi's console / kernel log next to the cell folder (v1 packaged them)
SIDE_LOGS = (("{cid}.log", "pi_console.log"), ("{cid}.dmesg.log", "pi_dmesg.log"))

V1_COLUMNS = ["cell_id", "model_key", "task", "model", "cpu_mhz", "repeat", "round", "modes",
              "frames_per_mode", "imgsz", "batch", "started", "git_commit", "n_samples", "trace_s",
              "baseline_W", "mean_P_W_forward", "energy_total_J", "step_coverage",
              "max_residual_ms", "drift_ppm", "slip_warning", "slip_samples_total", "n_slips",
              "drift_warning", "throttled_start", "throttled_end"]
V2_COLUMNS = ["family", "dataset_version", "variant", "quant", "holdout", "threads", "governor",
              "fan_mode", "input_source", "input_image", "serve_fps", "soak_minutes",
              "concurrent_mode", "energy_per_frame_mJ", "dyn_energy_per_frame_mJ",
              "latency_ms_per_frame", "latency_ms_per_image", "map50_95", "map50",
              "map_measured_on_machine", "counter_coverage", "max_temp_C",
              "baseline_drift_exempt", "bench"]
INDEX_COLUMNS = V1_COLUMNS + V2_COLUMNS

GLOSSARY = {
    "cell_id": "Run id and folder name under `cells/`: `{model}[_{variant}][_{condition}]_{imgsz}_t{threads}[_b{batch}]_{clock}_r{repeat}`; `clock` is the pinned MHz or `dvfs`; `v1ref_` cells are the v1 cross-check.",
    "model_key": "Model family key of the configuration (`yolo11s`, `yolov8m`, ...).",
    "task": "`image` (all v2 cells).",
    "model": "Weights file the run loaded (`yolo11n.pt`; the INT8 variant is in `variant`).",
    "cpu_mhz": "Pinned CPU clock in MHz. For a DVFS cell (`governor` = `ondemand`) it is 2400 and only the clock around the sync pulses; the clock then moved freely inside 1500-2400 MHz.",
    "repeat": "Repeat number of this configuration (0-based).",
    "round": "Round in which the cell ran (round r holds repeat r of every configuration).",
    "modes": "Measured modes (`forward`).",
    "frames_per_mode": "Nominal measured iterations per mode (a soak runs until its time is up; see `derived.json` `n_frames`).",
    "imgsz": "Image size in px (letterboxed square).",
    "batch": "Images per forward call.",
    "started": "When the orchestrator started this (last) attempt, EliteBook local time.",
    "git_commit": "Repo commit on the Pi that ran the cell.",
    "n_samples": "Rows in `power_trace.parquet`.",
    "trace_s": "Trace length in seconds (`n_samples` / the fitted sample rate).",
    "baseline_W": "Idle power: mean of the baseline_A and baseline_B windows around the measured phases, W.",
    "mean_P_W_forward": "Mean power over the whole `forward` phase (all iterations, W).",
    "energy_total_J": "Energy of the whole trace, J.",
    "step_coverage": "Share of samples inside the measured phase that carry a `step` label.",
    "max_residual_ms": "Worst clock-alignment residual over the sync edges, ms (limit 1.0; every kept cell passed).",
    "drift_ppm": "Fitted clock drift between the Monsoon and the Pi, ppm.",
    "slip_warning": "True if samples were lost during the run (see Quality flags).",
    "slip_samples_total": "Samples lost in the run (about 0.2 ms each).",
    "n_slips": "Number of loss events (times in `alignment.json` `slips[].t_ns`).",
    "drift_warning": "True if the fitted clock drift is above 50 ppm.",
    "throttled_start": "`vcgencmd get_throttled` before the run (0 = nothing).",
    "throttled_end": "`vcgencmd get_throttled` after the run (0 = nothing; an under-voltage fails the cell).",
    "family": "Matrix family from the manifest: `A`, `A-input`, `B`, `C`, `D`, `soak`, `ref`.",
    "dataset_version": "2 for every v2 cell (counters guaranteed); 1 for the `ref` cells (v1 command line, counters off).",
    "variant": "Model variant from `run_meta.json` (`yolo11n`, `yolo11n_int8s`, ...).",
    "quant": "Precision: `fp32` or `int8-static` (FX graph mode, qnnpack, 200 calibration pictures).",
    "holdout": "Holdout tag: `primary-s` (yolo11s, yolov8s), `secondary-m` (yolo11m, yolov8m), `none` (everything else). Empty for the `ref` cells, which carry no tag.",
    "threads": "PyTorch threads for the main workload.",
    "governor": "`pinned` (performance governor at `cpu_mhz`) or `ondemand` (free-running clock, DVFS).",
    "fan_mode": "`full`, `auto` or `off`.",
    "input_source": "`repeat` (one picture repeated) or `stream` (50 different pictures).",
    "input_image": "The single picture used (`bus.jpg`, `empty.jpg`, `crowded.jpg`), or `N files` for a stream.",
    "serve_fps": "Requests per second of the serve phase (1, 5, 15); empty for the default 10 requests with 200 ms gaps.",
    "soak_minutes": "Length of a soak run in minutes; empty otherwise.",
    "concurrent_mode": "`model` (a second yolo11n at the same time), `spinner` (a matrix-multiply spinner), or empty.",
    "energy_per_frame_mJ": "Energy of one measured iteration (frame 0 excluded), mJ. For a concurrent cell: the whole board, both workloads.",
    "dyn_energy_per_frame_mJ": "The same minus `baseline_W` times the iteration's duration, mJ.",
    "latency_ms_per_frame": "Mean duration of one measured iteration, ms.",
    "latency_ms_per_image": "The same per image (divided by `batch`), ms.",
    "map50_95": "COCO mAP50-95 of this model / variant / image size on 500 validation pictures (one number per model, not per cell).",
    "map50": "COCO mAP50 of the same measurement.",
    "map_measured_on_machine": "Machine the mAP was measured on (`aarch64` = the Pi; anything else is a provisional NUC number).",
    "counter_coverage": "Share of samples in the measured phase that have `perf` counter data (limit 0.95; empty if the cell has no counters).",
    "max_temp_C": "Highest SoC temperature read during the run, C (the run aborts at 85).",
    "baseline_drift_exempt": "Reason the idle-baseline drift check did not apply (soak, fan auto); empty otherwise.",
    "bench": "True if the per-layer-type power benchmarks ran (`perop_power.csv` has rows); only repeat 0 of A / A-input / C.",
}

FAMILY_DESC = {
    "A": "YOLO models yolov8s, yolov8m, yolo11s, yolo11m at 2400 MHz (5 repeats), 2000 and 1500 MHz (3 each); per-layer benchmarks on repeat 0",
    "A-input": "yolo11n on a picture the detector finds nothing in and on a crowded one, 3 repeats each",
    "B": "INT8 static yolo11n (2400, 2000, 1500 MHz) and yolo11s (2400 MHz), 3 repeats each",
    "C": "image size 320 / 480 / 960, batch 2 / 4 / 8, threads 1 / 2, 3 repeats each",
    "D": "free-running clock (DVFS), fan `auto` and `off`, serving at 1 / 5 / 15 requests per second, a stream of 50 different pictures, two workloads at once (`conc2`, `concspin`)",
    "soak": "30 minutes of continuous inference, fan `full`, yolo11n and yolo11m",
    "ref": "the v1 cross-check: yolov8n and yolo11n with the v1 command lines at 1500 / 2000 / 2400 MHz, counters off, no holdout (`dataset_version` 1)",
    "lockbox": "holdout configurations measured after their predictions were frozen (never training data)",
    "v3cal": "v3 calibration pairs: yolo11n, yolov8n, yolo11s, yolo12n, mbv3, effb0, resnet18 with the layer hooks on and off (`run_meta.layer_hooks`), 2400 MHz, 3 repeats each",
    "v3add": "v3 training additions, hooks off: yolo12s, Swin-T, ViT-B/16, ConvNeXt-T, MaxViT-T at 2400 MHz and yolo12s, Swin-T at 1500 MHz, 3 repeats each",
    "lockbox2": "second holdout (R3), hooks off: resnet34, mobilenet_v3_large, efficientnet_b1, swin_s, yolo12m at 2400 MHz, 3 repeats each, measured after the R3 predictions were frozen",
}

COVERAGE_DIMENSIONS = ("family", "dataset_version", "model", "imgsz", "batch", "threads", "variant",
                       "quant", "clock_mhz_or_dvfs", "governor", "fan_mode", "serve", "soak_minutes",
                       "input_source", "input_image", "concurrent", "holdout", "bench")


class PackageError(RuntimeError):
    """A packaging step failed or a verification did not match. Never swallowed."""


# ---- reading inputs --------------------------------------------------------------------------
def load_jsonl(path) -> list[dict]:
    """One JSON object per line. A truncated LAST line (the file is being appended to by a running
    matrix) is ignored; a bad line anywhere else is an error."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    out = []
    for i, ln in enumerate(lines):
        try:
            out.append(json.loads(ln))
        except ValueError:
            if i == len(lines) - 1:
                continue
            raise PackageError(f"{path}: line {i + 1} is not valid JSON") from None
    return out


def read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise PackageError(f"cannot read {path}: {e}") from None


@dataclass
class Source:
    runs: Path
    manifest: list[dict]
    results: list[dict]


@dataclass
class Cell:
    cell_id: str
    family: str
    manifest: dict
    last: dict
    attempts: int
    runs: Path

    @property
    def dir(self) -> Path:
        return self.runs / self.cell_id


def error_line(text, limit: int = 300) -> str:
    """The most telling single line of a failure text: the last ERROR / ...Error line if there is
    one, else the last non-empty line."""
    lines = [" ".join(ln.split()) for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in reversed(lines):
        if re.search(r"\bERROR\b|\b[A-Za-z]+Error\b|\bUndervoltage\b", ln):
            return ln[:limit]
    return lines[-1][:limit]


def select(sources: list[Source]) -> tuple[list[Cell], list[dict], list[str]]:
    """(kept cells in manifest order, excluded rows, results cell ids that are not in a manifest).

    A cell is kept iff the LAST row of its results is `status == "ok"` (the cell folder holds the
    last attempt's files, so an earlier ok followed by a later failure is excluded)."""
    kept: list[Cell] = []
    excluded: list[dict] = []
    orphans: list[str] = []
    seen: set[str] = set()
    for src in sources:
        by_cell: dict[str, list[dict]] = {}
        for r in src.results:
            if r.get("cell_id"):
                by_cell.setdefault(r["cell_id"], []).append(r)
        ids = {m["cell_id"] for m in src.manifest}
        orphans += sorted(set(by_cell) - ids)
        for m in src.manifest:
            cid = m["cell_id"]
            if cid in seen:
                raise PackageError(f"cell {cid} appears twice (in more than one manifest)")
            seen.add(cid)
            rows = by_cell.get(cid, [])
            last = rows[-1] if rows else None
            family = m.get("family") or ""
            if last is not None and last.get("status") == "ok":
                kept.append(Cell(cid, family, m, last, len(rows), src.runs))
            else:
                excluded.append({"cell_id": cid, "family": family,
                                 "last_stage": (last.get("stage") or "unknown") if last else "not_run",
                                 "last_error": error_line(last.get("error")) if last else "",
                                 "attempts": len(rows)})
    return kept, excluded, orphans


# ---- trace streaming, repack, verify ---------------------------------------------------------
def _decode(col):
    return col.dictionary_decode() if pa.types.is_dictionary(col.type) else col


def trace_stats(path, modes) -> dict:
    """Row count and counter coverage of a trace, reading only the `phase` and `ctr_instructions`
    columns in batches. Coverage = share of samples inside the mode phases that have counter data
    (verify_cell's definition); None if the trace has no counter column or no mode samples."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in ("phase", "ctr_instructions") if c in names]
    n_mode = n_ctr = 0
    if "phase" in cols and modes:
        for b in pf.iter_batches(batch_size=ROW_SLICE, columns=cols):
            phase = _decode(b.column("phase"))
            m = pc.is_in(phase, value_set=pa.array(list(modes)).cast(phase.type))
            n_mode += pc.sum(m.cast(pa.int64())).as_py() or 0
            if "ctr_instructions" in cols:
                both = pc.and_(m, pc.is_valid(b.column("ctr_instructions")))
                n_ctr += pc.sum(both.cast(pa.int64())).as_py() or 0
    cov = (n_ctr / n_mode) if ("ctr_instructions" in names and n_mode) else None
    return {"num_rows": pf.metadata.num_rows, "mode_rows": n_mode, "counter_coverage": cov}


def _batches(path, n: int):
    """The file as record batches of at most `n` rows. Row group by row group (peak memory is about
    three times one decoded row group; the writer's default is 1 Mi rows, ~0.3 GB at 38 columns);
    `iter_batches` reads ahead several row groups and used 4 GB on a 9 M-row trace, so it is only the
    fallback for a file with a very large row group."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    biggest = max((md.row_group(i).num_rows for i in range(md.num_row_groups)), default=0)
    if biggest > MAX_GROUP_ROWS:
        yield from pf.iter_batches(batch_size=n, use_threads=False)
        return
    for i in range(md.num_row_groups):
        yield from pf.read_row_group(i, use_threads=False).to_batches(max_chunksize=n)


def _slices(path, n: int = ROW_SLICE):
    """The file as tables of exactly `n` rows (the last one shorter), whatever its row groups."""
    buf: list[pa.RecordBatch] = []
    have = 0
    for b in _batches(path, n):
        if not b.num_rows:
            continue
        buf.append(b)
        have += b.num_rows
        while have >= n:
            t = pa.Table.from_batches(buf)
            yield t.slice(0, n)
            rest = t.slice(n)
            buf, have = rest.to_batches(), rest.num_rows
    if have:
        yield pa.Table.from_batches(buf)


def repack_parquet(src, dst, rows: int = ROW_SLICE) -> None:
    """Rewrite `src` as zstd level 9, one slice (= one row group) at a time. The schema, including
    the pandas metadata, is kept."""
    schema = pq.ParquetFile(src).schema_arrow
    with pq.ParquetWriter(dst, schema, compression="zstd", compression_level=ZSTD_LEVEL) as w:
        for t in _slices(src, rows):
            w.write_table(t, row_group_size=rows)


def _col_equal(a: pa.ChunkedArray, b: pa.ChunkedArray) -> bool:
    a, b = _decode(a), _decode(b)
    if a.type != b.type or len(a) != len(b):
        return False
    if a.equals(b):
        return True
    if pa.types.is_floating(a.type):        # Arrow's equals treats NaN != NaN; the trace has NaNs
        if not np.array_equal(a.is_null().to_numpy(zero_copy_only=False),
                              b.is_null().to_numpy(zero_copy_only=False)):
            return False
        x = a.fill_null(0).to_numpy()
        y = b.fill_null(0).to_numpy()
        return bool(np.array_equal(x, y, equal_nan=True))
    return False


def tables_equal(a: pa.Table, b: pa.Table) -> bool:
    if a.schema.names != b.schema.names or a.num_rows != b.num_rows:
        return False
    return a.equals(b) or all(_col_equal(a[n], b[n]) for n in a.schema.names)


def verify_parquet(src, dst, rows: int = ROW_SLICE) -> int:
    """Raise PackageError unless `dst` has the same schema, row count and values as `src`.
    Returns the row count."""
    ps, pd_ = pq.ParquetFile(src), pq.ParquetFile(dst)
    if not ps.schema_arrow.equals(pd_.schema_arrow, check_metadata=True):
        raise PackageError(f"{dst}: schema differs from {src}")
    if ps.metadata.num_rows != pd_.metadata.num_rows:
        raise PackageError(f"{dst}: {pd_.metadata.num_rows} rows, original has {ps.metadata.num_rows}")
    for rg in range(pd_.metadata.num_row_groups):
        for c in range(pd_.metadata.num_columns):
            codec = pd_.metadata.row_group(rg).column(c).compression
            if codec != "ZSTD":
                raise PackageError(f"{dst}: row group {rg} column {c} is {codec}, not ZSTD")
    n = 0
    sa, sb = _slices(src, rows), _slices(dst, rows)
    for i, (ta, tb) in enumerate(zip(sa, sb)):
        if not tables_equal(ta, tb):
            raise PackageError(f"{dst}: values differ from {src} in rows {i * rows}-{i * rows + rows}")
        n += ta.num_rows
    if n != ps.metadata.num_rows:
        raise PackageError(f"{dst}: compared {n} of {ps.metadata.num_rows} rows")
    return n


def gzip_events(src, dst) -> None:
    """events.csv -> events.csv.gz (level 9, no name or mtime in the header, so it is reproducible)."""
    with open(src, "rb") as f, open(dst, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g, 1 << 20)


def _digest(fileobj) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    while chunk := fileobj.read(1 << 20):
        h.update(chunk)
        n += len(chunk)
    return h.hexdigest(), n


def verify_gzip(src, dst) -> int:
    """Raise PackageError unless `dst` decompresses to exactly the bytes of `src`."""
    with open(src, "rb") as f:
        want = _digest(f)
    try:
        with gzip.open(dst, "rb") as g:
            got = _digest(g)
    except (OSError, EOFError) as e:
        raise PackageError(f"{dst}: not a readable gzip file: {e}") from None
    if want != got:
        raise PackageError(f"{dst}: decompressed content differs from {src} "
                           f"({got[1]} bytes against {want[1]})")
    return want[1]


def copy_verified(src, dst) -> None:
    shutil.copy2(src, dst)
    with open(src, "rb") as a, open(dst, "rb") as b:
        if _digest(a) != _digest(b):
            raise PackageError(f"copy of {src} to {dst} differs")


HOOKS_OFF_ABSENT = ("layer_table.csv",)   # run.py --layer-hooks off writes no layer table


def hooks_off(cell_dir: Path) -> bool:
    """True when the cell's run_meta.json records `layer_hooks: "off"` (older cells have no key: hooks on)."""
    try:
        return read_json(Path(cell_dir) / "run_meta.json").get("layer_hooks") == "off"
    except (OSError, ValueError, PackageError):   # unreadable meta: check_files / later steps report it
        return False


def required_files(cell_dir: Path) -> tuple:
    """REQUIRED_FILES without the layer table for a hooks-off cell (static features then come from
    tools/static_features.py)."""
    return tuple(f for f in REQUIRED_FILES if not (hooks_off(cell_dir) and f in HOOKS_OFF_ABSENT))


def check_files(cell: Cell) -> None:
    missing = [f for f in required_files(cell.dir) if not (cell.dir / f).is_file()]
    if missing:
        raise PackageError(f"{cell.cell_id}: ok in results but files missing in {cell.dir}: {missing}")


def package_cell(cell: Cell, out: Path) -> dict:
    """Copy and repack one cell into <out>/cells/<cell_id>/ (built in a temp folder, moved into
    place only after everything verified). Returns sizes for the report."""
    final = out / "cells" / cell.cell_id
    tmp = out / "cells" / f".{cell.cell_id}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = cell.dir
    try:
        copied = tuple(f for f in COPIED_FILES if not (hooks_off(src) and f in HOOKS_OFF_ABSENT))
        for f in copied + tuple(f for f in OPTIONAL_FILES if (src / f).is_file()):
            copy_verified(src / f, tmp / f)
        for pat, name in SIDE_LOGS:
            p = src.parent / pat.format(cid=cell.cell_id)
            if p.is_file():
                copy_verified(p, tmp / name)
        repack_parquet(src / "power_trace.parquet", tmp / "power_trace.parquet")
        rows = verify_parquet(src / "power_trace.parquet", tmp / "power_trace.parquet")
        gzip_events(src / "events.csv", tmp / "events.csv.gz")
        verify_gzip(src / "events.csv", tmp / "events.csv.gz")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"rows": rows, "parquet_in": (src / "power_trace.parquet").stat().st_size,
            "parquet_out": (final / "power_trace.parquet").stat().st_size}


# ---- index row ---------------------------------------------------------------------------------
def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _round(x, nd):
    x = _num(x)
    return round(x, nd) if x is not None else None


def _phase_energy_fallback(cell: Cell, modes) -> tuple[float | None, float | None]:
    """(total energy J, mean forward power W) from phase_energy.csv, for results rows without a
    verify summary."""
    e = n = s = 0.0
    mode = modes[0] if modes else None
    with open(cell.dir / "phase_energy.csv", newline="") as f:
        for r in csv.DictReader(f):
            try:
                e += float(r["E_mJ"])
                if r["phase"] == mode:
                    k = float(r["n_finite"])
                    n += k
                    s += k * float(r["mean_P_W"])
            except (KeyError, ValueError):
                continue
    return (e / 1000.0 or None), (s / n if n else None)


def index_row(cell: Cell, stats: dict) -> tuple[dict, dict]:
    """(dataset_index.csv row, extras for the README / readiness checks)."""
    meta = read_json(cell.dir / "run_meta.json")
    ali = read_json(cell.dir / "alignment.json")
    der = read_json(cell.dir / "derived.json")
    last, man = cell.last, cell.manifest
    ver = last.get("verify") or {}
    modes = list(meta.get("modes") or [])
    dv = meta.get("dataset_version")
    ref = der.get("map_reference") or meta.get("map_reference") or {}
    inputs = meta.get("inputs") or {}
    files = inputs.get("files") or []
    image = Path(files[0].get("path", "")).name if len(files) == 1 else (
        f"{len(files)} files" if files else "")
    batch = meta.get("batch")
    lat = der.get("latency_ms_per_frame")
    lat_img = (meta.get("latency_ms_per_image") or {}).get("mean")
    if lat_img is None and _num(lat) is not None and batch:
        lat_img = lat / batch
    energy_j = ver.get("energy_total_J")
    mean_fwd = (ver.get("mean_P_W") or {}).get("forward")
    if energy_j is None or mean_fwd is None:
        fe, fp = _phase_energy_fallback(cell, modes)
        energy_j = fe if energy_j is None else energy_j
        mean_fwd = fp if mean_fwd is None else mean_fwd
    rate = ali.get("slip_rate_hz") or 5000.0
    row = {
        "cell_id": cell.cell_id, "model_key": man.get("model_key") or last.get("model_key"),
        "task": meta.get("task"), "model": meta.get("model"),
        "cpu_mhz": man.get("cpu_mhz", last.get("cpu_mhz")), "repeat": man.get("repeat"),
        "round": man.get("round"), "modes": ",".join(modes),
        "frames_per_mode": meta.get("frames_per_mode"), "imgsz": meta.get("imgsz"), "batch": batch,
        "started": last.get("started"), "git_commit": last.get("git_pi"),
        "n_samples": stats["num_rows"], "trace_s": round(stats["num_rows"] / rate, 2),
        "baseline_W": _round(meta.get("baseline_W"), 6), "mean_P_W_forward": _round(mean_fwd, 6),
        "energy_total_J": _round(energy_j, 3), "step_coverage": _round(ver.get("step_coverage"), 4),
        "max_residual_ms": _round(ali.get("max_residual_ms"), 4),
        "drift_ppm": _round(ali.get("drift_ppm"), 3),
        "slip_warning": bool(ali.get("slip_warning")),
        "slip_samples_total": ali.get("slip_samples_total", 0),
        "n_slips": len(ali.get("slips") or []), "drift_warning": bool(ali.get("drift_warning")),
        "throttled_start": meta.get("throttled_start"), "throttled_end": meta.get("throttled_end"),
        "family": cell.family, "dataset_version": dv, "variant": meta.get("variant"),
        "quant": meta.get("quant"),
        "holdout": (meta.get("holdout") or "none") if dv == 2 else "",
        "threads": meta.get("threads"), "governor": meta.get("governor"),
        "fan_mode": meta.get("fan_mode"), "input_source": inputs.get("source"),
        "input_image": image, "serve_fps": meta.get("serve_fps"),
        "soak_minutes": meta.get("soak_minutes"),
        "concurrent_mode": (meta.get("concurrent") or {}).get("mode") or "",
        "energy_per_frame_mJ": _round(der.get("energy_per_frame_mJ"), 3),
        "dyn_energy_per_frame_mJ": _round(der.get("dyn_energy_per_frame_mJ"), 3),
        "latency_ms_per_frame": _round(lat, 3), "latency_ms_per_image": _round(lat_img, 3),
        "map50_95": ref.get("map50_95"), "map50": ref.get("map50"),
        "map_measured_on_machine": (ref.get("measured_on") or {}).get("machine"),
        "counter_coverage": _round(stats["counter_coverage"], 4),
        "max_temp_C": meta.get("max_temp_C"),
        "baseline_drift_exempt": meta.get("baseline_drift_exempt"),
        "bench": not (meta.get("bench_skipped") or meta.get("bench_forced_off")
                      or meta.get("bench_forced_off_by_soak")),
    }
    pc_ = meta.get("perf_counters") or {}
    serve = (f"{meta['serve_fps']} requests/s" if meta.get("serve_fps") else
             f"{meta.get('serve_requests')} requests, {meta.get('serve_gap_ms'):g} ms gaps"
             if meta.get("serve_gap_ms") is not None else "none")
    extra = {
        "meta_holdout": meta.get("holdout"), "has_holdout_key": "holdout" in meta,
        "align_pass": bool(ali.get("pass")), "warnings": list(ver.get("warnings") or []),
        "vout_V": (meta.get("recorder_status_start") or {}).get("vout_V"),
        "kernel": meta.get("kernel"), "torch": meta.get("torch_version"),
        "paranoid": pc_.get("paranoid"), "scope": pc_.get("scope"),
        "interval_ms": pc_.get("interval_ms"), "events": tuple(pc_.get("events") or ()),
        "min_pct_running": pc_.get("min_pct_running"), "perf_enabled": bool(pc_.get("enabled")),
        "serve": serve, "yolo": str(meta.get("model") or "").lower().startswith("yolo"),
    }
    return row, extra


# ---- coverage ----------------------------------------------------------------------------------
def feature_values(row: dict, extra: dict) -> dict[str, str]:
    def s(x, empty="none"):
        return empty if x in (None, "") else str(x)
    clock = "dvfs" if row["governor"] == "ondemand" else s(row["cpu_mhz"], "?")
    return {
        "family": s(row["family"]), "dataset_version": s(row["dataset_version"]),
        "model": s(row["model_key"]), "imgsz": s(row["imgsz"]), "batch": s(row["batch"]),
        "threads": s(row["threads"]), "variant": s(row["variant"]), "quant": s(row["quant"]),
        "clock_mhz_or_dvfs": clock, "governor": s(row["governor"]), "fan_mode": s(row["fan_mode"]),
        "serve": extra["serve"], "soak_minutes": s(row["soak_minutes"]),
        "input_source": s(row["input_source"]), "input_image": s(row["input_image"]),
        "concurrent": s(row["concurrent_mode"]), "holdout": s(row["holdout"], "n/a (no tag)"),
        "bench": s(row["bench"]),
    }


def _sort_key(v: str):
    m = re.match(r"-?\d+(\.\d+)?", v)
    return (0, float(m.group()), v) if m else (1, 0.0, v)


def coverage_rows(feats: list[dict[str, str]]) -> list[dict]:
    out = []
    for dim in COVERAGE_DIMENSIONS:
        c = Counter(f[dim] for f in feats)
        for v in sorted(c, key=_sort_key):
            out.append({"dimension": dim, "value": v, "n_cells": c[v]})
    return out


# ---- readiness ---------------------------------------------------------------------------------
def _probe_status(doc: Path | None) -> tuple[bool, str]:
    """Phase 0 probe table of docs/dataset-v2.md: a probe is open if its `Known so far` cell still
    says it was not run, needs repeating or is still to be confirmed."""
    if doc is None or not doc.is_file():
        return False, f"probe table not found ({doc})"
    rows = []
    for ln in doc.read_text().splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0].isdigit() and ln.lstrip().startswith("|"):
            rows.append((int(cells[0]), cells[1], cells[-1]))
    if not rows:
        return False, f"no probe rows found in {doc.name}"
    open_re = re.compile(r"^not run|no Monsoon cell has been run|repeat it|Still to confirm|"
                         r"^not run on the Pi", re.I)
    open_ = [(n, name) for n, name, known in rows if open_re.search(known)]
    ev = f"{len(rows) - len(open_)} of {len(rows)} probes have an outcome in {doc.name} section 6"
    if open_:
        ev += "; still open: " + ", ".join(f"{n} ({name})" for n, name in open_)
    return not open_, ev


def readiness_checks(rows, extras, excluded, n_manifest, *, dry_run, verified, out: Path,
                     docs: Path | None) -> list[tuple[bool, str, str]]:
    """(ok, title, evidence) per item of the predictor-readiness checklist."""
    checks: list[tuple[bool, str, str]] = []
    n = len(rows)
    by_stage = Counter(e["last_stage"] for e in excluded)
    no_reason = [e["cell_id"] for e in excluded if e["last_stage"] == "not_run"]
    ok = not no_reason
    ev = f"{n} kept + {len(excluded)} excluded = {n + len(excluded)} of {n_manifest} manifest cells"
    if excluded:
        ev += "; excluded by last stage: " + ", ".join(f"{k} {v}" for k, v in sorted(by_stage.items()))
    if no_reason:
        ev += f"; {len(no_reason)} never ran (no reason yet), e.g. {no_reason[0]}"
    elif excluded:
        ev += "; every excluded cell has a recorded stage and error line (excluded.csv)"
    checks.append((ok and n > 0, "All cells ok, or explicitly excluded with a reason", ev))

    res = [(float(r["max_residual_ms"]), r["cell_id"]) for r in rows if r["max_residual_ms"] is not None]
    bad = [c for v, c in res if v >= ALIGN_LIMIT_MS]
    dvfs = [(float(r["max_residual_ms"]), r["cell_id"]) for r in rows
            if r["governor"] == "ondemand" and r["max_residual_ms"] is not None]
    not_passed = [r["cell_id"] for r, x in zip(rows, extras) if not x["align_pass"]]
    ev = (f"worst {max(res)[0]:.4f} ms ({max(res)[1]}) over {len(res)} cells" if res else "no cells")
    ev += (f"; DVFS cells {len(dvfs)}, worst {max(dvfs)[0]:.4f} ms" if dvfs
           else "; NO DVFS cell in this selection")
    if bad:
        ev += f"; {len(bad)} cells at or above {ALIGN_LIMIT_MS} ms: {bad[:3]}"
    if not_passed:
        ev += f"; alignment.json pass=false: {not_passed[:3]}"
    checks.append((bool(res) and not bad and not not_passed and bool(dvfs),
                   f"Alignment residual < {ALIGN_LIMIT_MS:g} ms for every kept cell, DVFS included", ev))

    v2 = [r for r in rows if r["dataset_version"] == 2]
    cov = [(r["counter_coverage"], r["cell_id"]) for r in v2]
    missing = [c for v, c in cov if v is None]
    low = [(v, c) for v, c in cov if v is not None and v < COUNTER_COVERAGE_MIN]
    have = [x for x in cov if x[0] is not None]
    ev = f"{len(v2)} v2 cells"
    if have:
        ev += f", worst coverage {min(have)[0]:.4f} ({min(have)[1]})"
    if missing:
        ev += f"; {len(missing)} without counter data, e.g. {missing[0]}"
    if low:
        ev += f"; {len(low)} below {COUNTER_COVERAGE_MIN:g}, e.g. {low[0][1]}"
    ref = n - len(v2)
    if ref:
        ev += f" ({ref} dataset_version 1 cells have no counters by design)"
    checks.append((bool(v2) and not missing and not low,
                   f"Counter coverage >= {COUNTER_COVERAGE_MIN * 100:.0f} % for every v2 cell", ev))

    bad_h = []
    for r, x in zip(rows, extras):
        want = holdout_tag(r["model_key"] or "") if r["dataset_version"] == 2 else None
        if not x["has_holdout_key"] or x["meta_holdout"] != want:
            bad_h.append(r["cell_id"])
    tags = Counter(r["holdout"] or "n/a" for r in rows)
    ev = "run_meta.holdout present and equal to the model rule for all cells" if not bad_h else \
        f"{len(bad_h)} cells with a missing or wrong holdout, e.g. {bad_h[0]}"
    ev += "; cells per tag: " + ", ".join(f"{k} {v}" for k, v in sorted(tags.items()))
    checks.append((n > 0 and not bad_h, "Holdout recorded for every cell", ev))

    quant = [r for r in rows if r["quant"] not in NO_QUANT]
    miss = [r["cell_id"] for r in quant if any(r[k] is None for k in
            ("map50_95", "energy_per_frame_mJ", "latency_ms_per_frame"))]
    variants = sorted({r["variant"] for r in quant})
    ev = (f"{len(quant)} quantized cells, variants {variants}" if quant else
          "no quantized cell in this selection")
    if miss:
        ev += f"; {len(miss)} lack mAP, energy/frame or latency/frame, e.g. {miss[0]}"
    checks.append((bool(quant) and not miss,
                   "Every quantized variant has mAP, energy/frame and latency/frame", ev))

    yolo = [r for r, x in zip(rows, extras) if x["yolo"]]
    pi = [r for r in yolo if r["map_measured_on_machine"] == "aarch64"]
    checks.append((bool(yolo) and len(pi) == len(yolo),
                   "mAP of every YOLO cell measured on the Pi (plan Task 16)",
                   f"{len(pi)} of {len(yolo)} YOLO cells carry a Pi-measured mAP reference"))

    have_cov = (out / "coverage.csv").is_file()
    checks.append((have_cov, "Coverage table published",
                   "coverage.csv written (dimension, value, n_cells)" if have_cov else "coverage.csv missing"))

    readme = out / "README.md"
    has_split = readme.is_file() and "v1 versus v2" in readme.read_text()
    has_manifest = (out / "manifest.jsonl").is_file() and (out / "results.jsonl").is_file()
    checks.append((has_split and has_manifest, "README and manifest included, with the v1-vs-v2 counter split",
                   f"README.md {'with' if has_split else 'WITHOUT'} the counter-split section; "
                   f"manifest.jsonl + results.jsonl {'copied' if has_manifest else 'MISSING'}; "
                   "committing them to the dataset branch is the controller's step"))

    ok, ev = _probe_status(docs)
    checks.append((ok, "Probe outcomes documented", ev))

    if dry_run:
        checks.append((False, "Trace repack verified (parquet identical, events gz identical)",
                       "dry run: no trace was copied or verified"))
    else:
        checks.append((verified == n and n > 0, "Trace repack verified (parquet identical, events gz identical)",
                       f"{verified} of {n} cells repacked and verified (schema, row count, values; gz bytes)"))
    return checks


def readiness_md(checks, name, dry_run) -> str:
    done = sum(1 for c in checks if c[0])
    lines = [f"# Predictor readiness: {name}", "",
             f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by `host/package_dataset.py`"
             + (" (**dry run**: no trace was copied)" if dry_run else "") + ".",
             f"{done} of {len(checks)} items satisfied. Each line is computed from the data in this folder "
             "where it can be.", ""]
    for ok, title, ev in checks:
        lines.append(f"- [{'x' if ok else ' '}] **{title}.** {ev}")
    return "\n".join(lines) + "\n"


# ---- README ------------------------------------------------------------------------------------
def _fmt_list(vals, fmt=str):
    vals = sorted({v for v in vals if v is not None})
    return ", ".join(fmt(v) for v in vals) if vals else "unknown"


def _human(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def readme_md(rows, extras, excluded, sources, name, *, dry_run, out: Path, probe_doc: str,
              branch_name: str | None = None) -> tuple[str, list[str]]:
    """The dataset README (v1 style). Returns (text, TODO lines for the controller)."""
    todos: list[str] = []
    n = len(rows)
    total = sum(r["n_samples"] for r in rows)
    hours = sum(float(r["trace_s"]) for r in rows) / 3600.0
    hours_txt = f"{hours:.1f}" if hours >= 1 else f"{hours:.2f}"
    started = sorted(r["started"] for r in rows if r["started"])
    first, last = (started[0], started[-1]) if started else ("?", "?")
    commits = _fmt_list(r["git_commit"] for r in rows)
    kernels = _fmt_list(x["kernel"] for x in extras)
    vouts = _fmt_list((x["vout_V"] for x in extras), lambda v: f"{v:.2f}")
    fans = Counter(r["fan_mode"] for r in rows)
    paranoid = _fmt_list(x["paranoid"] for x in extras)
    scopes = _fmt_list(x["scope"] for x in extras)
    events = max((x["events"] for x in extras), key=len, default=())
    interval = _fmt_list(x["interval_ms"] for x in extras)
    minpct = [x["min_pct_running"] for x in extras if x["perf_enabled"] and x["min_pct_running"] is not None]
    temps = [r["max_temp_C"] for r in rows if r["max_temp_C"] is not None]
    n_manifest = sum(len(s.manifest) for s in sources)
    lfs = 2 * n
    if dry_run:
        size = "TODO(controller): size on disk (dry run, nothing was copied)"
        todos.append("size on disk (rerun without --dry-run)")
    else:
        size = f"{_human(_dir_bytes(out / 'cells'))} (Git LFS: {lfs} objects, `power_trace.parquet` and `events.csv.gz` of every cell)"
    fam_n = Counter(r["family"] for r in rows)
    fam_plan = Counter(m.get("family") or "" for s in sources for m in s.manifest)
    fam_rows = []
    for fam in sorted(fam_plan, key=lambda f: (list(FAMILY_DESC).index(f) if f in FAMILY_DESC else 99, f)):
        d = FAMILY_DESC.get(fam, "TODO(controller): describe this family")
        if fam not in FAMILY_DESC:
            todos.append(f"describe family {fam}")
        fam_rows.append(f"| `{fam}` | {d} | {fam_n.get(fam, 0)} / {fam_plan[fam]} |")
    bench_n = Counter(r["family"] for r in rows if r["bench"])
    v2n = sum(1 for r in rows if r["dataset_version"] == 2)
    quality = quality_lines(rows, extras)
    pi_map = sum(1 for r, x in zip(rows, extras) if x["yolo"] and r["map_measured_on_machine"] == "aarch64")
    n_yolo = sum(1 for x in extras if x["yolo"])
    if pi_map < n_yolo:
        todos.append(f"mAP measured on the Pi for {n_yolo - pi_map} of {n_yolo} YOLO cells (plan Task 16)")
    ex_line = (f"{len(excluded)} did not reach `ok`: see `excluded.csv`" if excluded
               else "every manifest cell reached `ok`")
    if excluded:
        todos.append(f"decide for the {len(excluded)} excluded cells: rerun or accept (see excluded.csv)")
    if branch_name:
        branch = branch_name
    else:
        branch = "<BRANCH>"
        todos.append("branch name for the download commands (rerun with --branch-name)")
    gloss = "\n".join(f"| `{c}` | {GLOSSARY[c]} |" for c in INDEX_COLUMNS)
    text = f"""# Monsoon power dataset v2: Raspberry Pi 5 YOLO workloads ({name})

Whole-board input power of a Raspberry Pi 5, measured by a Monsoon HVPM at about 5,000 samples
per second, while the Pi runs YOLO object detection under many conditions. Every power sample is
labelled with what the Pi was doing at that instant (the **phase**, the **iteration**, the
**step** preprocess / forward / postprocess, the **network layer**), and every v2 sample also
carries the Pi's hardware performance counters (10 Hz) and its CPU clock, temperature and fan
state (1 Hz). It extends dataset v1 (branch `dataset-2026-10-matrix`, 224 cells).

| | |
|---|---|
| Runs ("cells") | **{n}** of {n_manifest} planned ({ex_line}) |
| Labelled samples | {total / 1e6:.1f} million, about {hours_txt} hours of trace |
| Collected | {first} to {last} (the EliteBook's local time) |
| Code | commit(s) `{commits}` of this repo, on the Pi |
| Size | {size} |

The rig, the method and the code are described in the repo's `README.md` (sections 1, 6, 10 and
11) and `docs/dataset-v2.md`.

## What was measured

Every cell is a YOLO detection run (`--task image`, `bus.jpg` as the detection sanity check,
Ultralytics weights, PyTorch eager on the CPU, 10 warm-up passes), repeated as the table says.
The last column is cells in this dataset / cells planned.

| Family | What varies | Cells |
|---|---|---|
{chr(10).join(fam_rows)}

Cell counts per value of every feature are in `coverage.csv`.

## Conditions

- **Supply:** Monsoon output {vouts} V (v1 used 5.25 V; it was raised after under-voltage with the
  s / m models). A cell that logged under-voltage (kernel message, the live `sys_uv_alarm` or the
  `get_throttled` bits) failed and is not in the dataset.
- **Kernel:** {kernels} (v1 ran on 6.18.50; the `ref` family measures the difference).
- **CPU clock:** pinned with the performance governor except `governor` = `ondemand` cells (DVFS:
  the clock is pinned only around the sync pulses).
- **Fan:** {", ".join(f"`{k}` {v}" for k, v in sorted(fans.items()))} cells. The soaks use `full`.
- **Temperature:** each cell started below 50 C (the orchestrator waits); `run.py` aborts at 85 C and
  stops at 90 C, and a cell with `thermal_abort.json` or `max_temp_C` >= 85 fails.
  Highest temperature in this dataset: {max(temps) if temps else "unknown"} C.
- **Hardware counters:** `perf stat` every {interval} ms with `perf_event_paranoid` = {paranoid}, so
  `scope` = {scopes}: only the run's **user-space** code is counted (events print as `cycles:u`);
  kernel time is not included. Events: {", ".join(events) if events else "none"}.
  Lowest share of an interval in which any hardware event was counted: {min(minpct) if minpct else "n/a"} %.
  Counters are on in {v2n} of the {n} cells (all `dataset_version` 2 cells).
- **Per-layer benchmarks** (`bench:*` phases, `perop_power.csv`) run only on **repeat 0** of each
  A / A-input / C configuration (Ruling R31): per-op steady-state power is a per-configuration
  property, while energy and latency per frame come from every repeat's trace. Cells with
  `bench` = True: {", ".join(f"{k} {v}" for k, v in sorted(bench_n.items())) or "none"}. INT8 and soak cells have none.
- **Wi-Fi** on the Pi was off while measuring.
- **Serve phase:** 10 single requests with 200 ms gaps (or the family's request rate).

## Layout

```
data/{name}/
  README.md                  this file
  dataset_index.csv          one row per run: settings, derived numbers, quality flags
  coverage.csv               every feature dimension, its values and the number of cells
  excluded.csv               manifest cells that did not reach ok, with the reason
  readiness.md               the predictor-readiness checklist, computed from this data
  manifest.jsonl             the planned cells and their exact run.py arguments
  results.jsonl              the orchestrator's log, one line per attempt (incl. failed ones)
  cells/<cell_id>/           one folder per run
    power_trace.parquet      the labelled 5 kHz trace (main data), zstd
    events.csv.gz            the Pi's raw event log (input to the alignment)
    run_meta.json            settings, software versions, temperatures, baseline, perf-counter setup, mAP reference
    alignment.json           clock alignment quality and detected sample losses
    derived.json             energy and latency per frame (measured iterations only)
    phase_energy.csv         energy and mean power per phase / iteration / step
    layer_table.csv          per-layer timing, shapes (and MACs / bytes for FP32 cells)
    perop_power.csv          isolated per-layer-type power (repeat 0 of A / A-input / C; otherwise empty)
    op_table.csv             PyTorch operator timings (first iteration)
    context.csv              temperature, CPU clock and throttle flags every 5 s
    sysfs.csv                per-core clock, temperature, fan state and under-voltage alarm every second
    counters.csv             hardware counters, one row per 100 ms (v2 cells)
    thermal_abort.json       only if run.py stopped the run for heat (such cells are not kept)
    pi_console.log, pi_dmesg.log   run.py's console output and the Pi's kernel messages, when present
```

## `dataset_index.csv` columns

| Column | Meaning |
|---|---|
{gloss}

Empty means "does not apply or was not recorded". The v1 index has the same first columns
(minus `seq_len`, `optimizer` and the train / prefill / decode power columns, which do not apply to
YOLO inference), so v1 and v2 indexes can be concatenated on the shared columns.

## `power_trace.parquet` columns

| Column | Meaning |
|---|---|
| `t_s` | seconds since the run's first event |
| `t_pi_ns` | sample time on the Pi's clock (GPS/PTP-disciplined wall clock), ns |
| `I_A`, `V_V`, `P_W` | input current, voltage and power at the Monsoon output |
| `phase` | `load_weights`, `first:<mode>`, `warmup`, `sync:start`, `baseline_A`, `forward`, `concurrent_drain`, `serve`, `baseline_B`, `gap`, `bench:<config_id>`, `sync:end`, `idle` |
| `mode`, `iter` | the mode and iteration number inside a measured phase (empty elsewhere) |
| `step` | `preprocess` / `forward` / `postprocess` (`idle_between_requests` in `serve`), or empty. In a concurrent cell: the main workload's step |
| `layer_path`, `depth` | the innermost network module running and its nesting depth (empty in soak and concurrent cells: no per-layer events) |
| `ctx_read` | `True` while the Pi's background sensor read (`vcgencmd`) ran; drop these rows for clean power |
| `ctr_<event>` | hardware counter for the 100 ms interval containing the sample: a **count per interval**, identical for every sample in it (use ratios, or one row per interval). Empty outside every interval |
| `pct_<event>` | share (0-100) of that interval in which the event was really counted; 100 = no counter sharing |
| `sys_cpu0_kHz` ... `sys_cpu3_kHz`, `sys_temp_C`, `sys_fan_state`, `sys_uv_alarm` | the latest 1 Hz reading (at most 2 s old): per-core clock, SoC temperature, fan state (0-4), live under-voltage alarm |
| `tag`, `step_m0`, `step_m1` / `step_spin` | only in `--concurrent` cells: which workloads were inside a step, and each one's own step |

## Loading

```python
import json, pandas as pd
idx = pd.read_csv("data/{name}/dataset_index.csv")
cell = "data/{name}/cells/" + idx.cell_id[0]
df = pd.read_parquet(cell + "/power_trace.parquet")
df = df[~df.ctx_read]
print(df[df.phase == "forward"].groupby("step").P_W.mean())
print(json.load(open(cell + "/derived.json"))["energy_per_frame_mJ"])
```

## The holdout

`holdout` marks cells to keep out of predictor training so that it can be tested on models it has
never seen. The tag follows the model, not the family, so every cell of an s-model (image sizes,
threads, INT8, ...) carries it:

| Model | `holdout` |
|---|---|
| yolo11s, yolov8s | `primary-s` (primary holdout: interpolation between the nano and medium models) |
| yolo11m, yolov8m | `secondary-m` (a second, harder holdout) |
| everything else | `none` |

The `ref` cells carry no tag (empty): they are v1 command lines for comparison, not training data.
Training rows: `idx[idx.holdout == "none"]` (this drops the `ref` cells, whose tag is empty); when
`secondary-m` is the second test set it is already excluded.

## v1 versus v2: the counter split

**The 224 v1 cells have no hardware counters and cannot get them afterwards.** Every
`dataset_version` 2 cell has `counters.csv`, the `ctr_*` / `pct_*` / `sys_*` columns in
`power_trace.parquet` and `perf_counters.enabled: true` in `run_meta.json`.

- A predictor that uses **counter features** trains on **v2 cells only**.
- A predictor that does **not** use them can use v1 and v2 together.

Both are legitimate; nobody should train on counter features with rows that silently have none.
The `ref` family ({fam_n.get("ref", 0)} cells here) re-runs the v1 command lines (yolov8n, yolo11n;
1500 / 2000 / 2400 MHz) with counters off on the new rig. Between v1 and v2 two things changed
together, and the `ref` cells measure them together: the supply voltage (5.25 V to {vouts} V) and the
Pi kernel (6.18.50 to {kernels}). They stay `dataset_version` 1. Use them only to compare against the
224 v1 cells.

## Quality flags (read before training on it)

{quality}
- **Not included:** failed or retried attempts (their last stage and error line are in
  `excluded.csv`, the full log in `results.jsonl`), cells cut from the matrix (below), the raw
  continuous Monsoon recording, `counters.raw.txt` (`counters.csv` is the same data), and v1 data.
- **Layer table for INT8 cells:** `bytes_moved`, `macs` and `arith_intensity` in `layer_table.csv`
  assume 4-byte floats, so they are wrong for quantized cells. Do not use them there.
- **INT8 static is not a fully INT8 network:** 80 convolutions are quantized; the C2PSA attention
  blocks, the Detect decode and every SiLU stay FP32.
- **Concurrent cells:** `energy_per_frame_mJ` covers both workloads together, not only the main one.
- **Soaks** (the 30-minute ones have about 9 million samples) have no per-layer begin/end events (`layer_events: false`).

## What was cut and why

- **INT8 dynamic.** Dynamic quantization only replaces Linear-type layers and YOLO is almost all
  convolutions, so it quantizes 0 modules; `run.py` fails such a run rather than label float32
  data INT8. The 9 cells are buildable behind `--include-int8d` but were not run.
- **Pruning (30 % and 50 % of yolo11n).** Structured pruning without fine-tuning gives mAP50-95
  of about 0 even after re-estimating the BatchNorm statistics, and the detection check on
  `bus.jpg` finds nothing. The tool and numbers are kept in `tools/`; the 6 cells were not run
  (`--include-prune`).
- **FP16** is optional (`--include-fp16`): it executes in float16 on the Pi CPU but whole yolo11n
  at 640 was slower than FP32 in an informal check (about 553 ms against about 330 ms per frame).
- **Cost of the perf counters.** Sampling ten counters at 10 Hz costs little, but the overhead was
  not measured by a strict A/B, so treat it as indicative: the perf-off `ref` cell yolo11n at 2400 MHz
  (v1 arguments) drew 9.6 W at 291 ms per frame, the perf-on yolo11n cells at 640 px 9.3 W at
  295.5 ms (different recipes). A controlled single-cell A/B on 2026-11-20 gave -0.15 W and +6 %
  latency, taken under under-voltage, so also indicative.
- Probe outcomes: `{probe_doc}` section 6.

## Downloading a subset

The traces are in Git LFS (a full download is a large share of GitHub's LFS bandwidth quota, so
take a subset). Everything except `power_trace.parquet` and `events.csv.gz` is plain git and
arrives with a normal clone:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch {branch} --single-branch https://github.com/macclab-stevens/power-measurement-pi.git
cd power-measurement-pi
# the index, README, coverage and every small file are now present; the traces are LFS pointers
git lfs pull --include="data/{name}/cells/yolo11s_640_t4_2400_r00/*"       # one cell
git lfs pull --include="data/{name}/cells/yolo11s_*"                      # one model
```

To select by feature, filter `dataset_index.csv` and pass the cells to `git lfs pull`:

```python
import pandas as pd, subprocess
idx = pd.read_csv("data/{name}/dataset_index.csv")
pick = idx[(idx.family == "C") & (idx.imgsz == 320)].cell_id
subprocess.run(["git", "lfs", "pull", "--include", ",".join(f"data/{name}/cells/{{c}}/*" for c in pick)], check=True)
```

## Open items (for the dataset owner)

{chr(10).join("- TODO(controller): " + t for t in todos)}
"""
    return text, todos


def quality_lines(rows, extras) -> str:
    n = len(rows)
    res = [float(r["max_residual_ms"]) for r in rows if r["max_residual_ms"] is not None]
    slips = [r for r in rows if r["slip_warning"]]
    sl = [int(r["slip_samples_total"]) for r in slips]
    drift = sum(1 for r in rows if r["drift_warning"])
    thr = [r["cell_id"] for r in rows if (r["throttled_start"] or 0) or (r["throttled_end"] or 0)]
    kinds = Counter(re.sub(r"\d+(\.\d+)?", "N", w) for x in extras for w in x["warnings"])
    ex = Counter(r["baseline_drift_exempt"] for r in rows if r["baseline_drift_exempt"])
    out = [f"- **Alignment:** every one of the {n} cells passed. Maximum sync-edge residual "
           + (f"{max(res):.3f} ms (limit 1.0; median {statistics.median(res):.3f})." if res else "unknown."),
           f"- **Sample losses (`slip_warning`, `slip_samples_total`, `n_slips`).** The Monsoon occasionally "
           f"loses a few milliseconds of samples without reporting it. {len(slips)} of {n} cells have at least "
           + (f"one detected loss (median {statistics.median(sl):.0f} samples, maximum {max(sl)}). " if sl else
              "one detected loss. ")
           + "After a loss that run's labels are shifted by roughly the loss size; averages over phases and "
           "steps are unaffected. For sample-exact work use the cells with `slip_warning == False` or cut "
           "the trace at `alignment.json` `slips[].t_ns`.",
           f"- **`drift_warning`:** {drift} of {n} cells have a fitted clock drift above 50 ppm (usually a side effect "
           "of a sample loss between the two sync trains). Their alignment still passed.",
           f"- **Under-voltage and throttling:** {len(thr)} of {n} cells have a non-zero `throttled_start` / `throttled_end`"
           + (f" (e.g. {thr[0]})" if thr else "") + "; a new under-voltage fails the cell, so none is kept.",
           "- **Baseline drift check:** cells with `baseline_drift_exempt` (" + (", ".join(f"{k}: {v}" for k, v in ex.items())
           or "none") + ") are exempt from the 0.5 W idle-baseline drift limit by design.",
           "- **Verify warnings** (the cell still passed): "
           + ("; ".join(f"{k} ({v} cells)" for k, v in kinds.most_common()) if kinds else "none") + "."]
    return "\n".join(out)


# ---- writing -----------------------------------------------------------------------------------
def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r[c]) for c in columns})


def package(sources: list[Source], out: Path, name: str, *, dry_run=False, docs: Path | None = None,
            branch_name: str | None = None, log=lambda m: None) -> dict:
    kept, excluded, orphans = select(sources)
    for cid in orphans:
        log(f"warning: results.jsonl has rows for {cid}, which is not in a manifest (ignored)")
    out.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        (out / "cells").mkdir(exist_ok=True)
    rows, extras, feats, verified = [], [], [], 0
    sizes = {"parquet_in": 0, "parquet_out": 0}
    for i, cell in enumerate(kept, 1):
        check_files(cell)
        modes = (read_json(cell.dir / "run_meta.json").get("modes")) or []
        stats = trace_stats(cell.dir / "power_trace.parquet", modes)
        if not dry_run:
            res = package_cell(cell, out)
            if res["rows"] != stats["num_rows"]:
                raise PackageError(f"{cell.cell_id}: repacked {res['rows']} rows, trace has {stats['num_rows']}")
            verified += 1
            for k in sizes:
                sizes[k] += res[k]
        row, extra = index_row(cell, stats)
        rows.append(row)
        extras.append(extra)
        feats.append(feature_values(row, extra))
        log(f"[{i}/{len(kept)}] {cell.cell_id}" + ("" if dry_run else " repacked + verified"))
    write_csv(out / "dataset_index.csv", INDEX_COLUMNS, rows)
    write_csv(out / "excluded.csv", ["cell_id", "family", "last_stage", "last_error", "attempts"], excluded)
    write_csv(out / "coverage.csv", ["dimension", "value", "n_cells"], coverage_rows(feats))
    _write_logs(sources, out)
    probe_doc = "docs/dataset-v2.md"
    text, todos = readme_md(rows, extras, excluded, sources, name, dry_run=dry_run, out=out, probe_doc=probe_doc,
                          branch_name=branch_name)
    (out / "README.md").write_text(text)
    n_manifest = sum(len(s.manifest) for s in sources)
    checks = readiness_checks(rows, extras, excluded, n_manifest, dry_run=dry_run, verified=verified,
                              out=out, docs=docs)
    (out / "readiness.md").write_text(readiness_md(checks, name, dry_run))
    return {"kept": len(kept), "excluded": len(excluded), "rows": rows, "checks": checks,
            "todos": todos, "sizes": sizes}


def _write_logs(sources: list[Source], out: Path) -> None:
    """manifest.jsonl and results.jsonl of all sources, concatenated in order."""
    for fname, attr in (("manifest.jsonl", "manifest"), ("results.jsonl", "results")):
        with open(out / fname, "w") as f:
            for s in sources:
                for r in getattr(s, attr):
                    f.write(json.dumps(r) + "\n")


def build_sources(runs, manifest, results, extra_runs) -> list[Source]:
    def one(r, m, res):
        r = Path(r)
        m = Path(m) if m else r / "manifest.jsonl"
        res = Path(res) if res else m.with_name("results.jsonl")
        if not m.is_file():
            raise PackageError(f"manifest not found: {m}")
        return Source(r, load_jsonl(m), load_jsonl(res) if res.is_file() else [])
    srcs = [one(runs, manifest, results)]
    srcs += [one(r, None, None) for r in (extra_runs or [])]
    return srcs


def main(argv=None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="runs folder holding the cell folders")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", default=None, help="default: results.jsonl next to the manifest")
    ap.add_argument("--extra-runs", action="append", default=[],
                    help="a second runs folder with its own manifest.jsonl / results.jsonl (repeatable)")
    ap.add_argument("--out", required=True, help="dataset folder (becomes data/<name>/ in git)")
    ap.add_argument("--name", required=True, help="dataset name, e.g. monsoon_v2_2026-11")
    ap.add_argument("--branch-name", default=None,
                    help="git branch the dataset will be pushed to (goes into the README download commands)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build index, coverage, excluded, readiness and README only; copy no trace")
    ap.add_argument("--docs", default=str(Path(__file__).resolve().parent.parent / "docs" / "dataset-v2.md"),
                    help="dataset-v2.md, whose Phase 0 probe table the readiness check reads")
    a = ap.parse_args(argv)
    try:
        srcs = build_sources(a.runs, a.manifest, a.results, a.extra_runs)
        r = package(srcs, Path(a.out), a.name, dry_run=a.dry_run, docs=Path(a.docs),
                    branch_name=a.branch_name, log=lambda m: print(m, file=sys.stderr))
    except PackageError as e:
        print(f"PACKAGING FAILED: {e}", file=sys.stderr)
        return 1
    print(f"{r['kept']} cells kept, {r['excluded']} excluded -> {a.out}"
          + (" (dry run)" if a.dry_run else ""))
    for ok, title, ev in r["checks"]:
        print(f"  [{'x' if ok else ' '}] {title}: {ev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
