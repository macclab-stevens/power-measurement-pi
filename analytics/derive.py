"""One derivation of energy and latency per image for every cell of v1, v2 and lockbox.

Reproduces host/verify_cell.py (_frame_table, derived_metrics) from phase_energy.csv: the `*_raw`
columns equal derived.json. The primary columns additionally drop every frame that overlaps a
power-meter time slip (alignment.json), so one resynchronisation glitch cannot bend a cell's mean.
Durations and energies are read from dur_ms / E_mJ, never recomputed from the trace.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import data
import runlog
import settings

CACHE_NAME = "derive.parquet"
BASELINE_PHASES = ("baseline_A", "baseline_B")
IDLE_STEP = "idle_between_requests"
COLUMNS = [
    "cell_id", "mode", "batch", "n_frames", "n_frames_slip_excluded", "n_images",
    "energy_mJ_per_image", "latency_ms_per_image", "mean_P_W", "dyn_energy_mJ_per_image",
    "energy_mJ_per_image_raw", "latency_ms_per_image_raw", "baseline_W", "baseline_A_W",
    "baseline_B_W", "serve_n_requests", "serve_dur_ms", "serve_E_mJ", "serve_idle_dur_ms",
    "serve_idle_E_mJ", "trace_E_J", "trace_dur_s", "slips_total", "slips_in_scored"]
NAN = float("nan")


def frame_table(pe: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per measured iteration (frame >= 1) of `mode`, indexed by frame (int).

    E_mJ and dur_ms are summed over all the frame's rows (every step, including an empty step and
    v2's profiler step); t_start_s is the minimum and t_end_s the maximum over those rows. Frame 0 is
    the cold first call; rows whose iter is not numeric are not frames."""
    rows = pe[pe["phase"] == mode]
    frame = pd.to_numeric(rows["iter"], errors="coerce")
    keep = frame >= 1
    rows = rows[keep].assign(frame=frame[keep].astype(int))
    out = rows.groupby("frame").agg(E_mJ=("E_mJ", "sum"), dur_ms=("dur_ms", "sum"),
                                    t_start_s=("t_start_s", "min"), t_end_s=("t_end_s", "max"))
    out.index = out.index.astype(int)
    return out


def slip_times_s(alignment: dict | None, meta: dict) -> list[tuple[float, float]]:
    """Each slip as (t_s, step_ms), t_s seconds since trace start; empty if t0_real_ns or slips missing."""
    t0 = (meta or {}).get("t0_real_ns")
    slips = (alignment or {}).get("slips")
    if t0 is None or not slips:
        return []
    return [((int(s["t_ns"]) - int(t0)) / 1e9, float(s["step_ms"])) for s in slips]


def slip_overlaps(frames: pd.DataFrame, slips: list[tuple[float, float]]) -> np.ndarray:
    """Bool mask over frame rows: [t_start_s, t_end_s] overlaps any [t_s, t_s + step_ms / 1000]."""
    n = len(frames)
    if n == 0 or not slips:
        return np.zeros(n, dtype=bool)
    s0 = np.array([t for t, _ in slips], dtype=float)
    s1 = s0 + np.array([ms for _, ms in slips], dtype=float) / 1000.0
    a = frames["t_start_s"].to_numpy(dtype=float)[:, None]
    b = frames["t_end_s"].to_numpy(dtype=float)[:, None]
    return ((a <= s1[None, :]) & (b >= s0[None, :])).any(axis=1)


def _baselines(pe: pd.DataFrame) -> tuple[float, float, float]:
    """(baseline_W, baseline_A_W, baseline_B_W): mean_P_W of each baseline phase that exists."""
    vals = []
    for ph in BASELINE_PHASES:
        r = pe.loc[pe["phase"] == ph, "mean_P_W"]
        vals.append(float(r.mean()) if len(r) and r.notna().any() else NAN)
    have = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(have)) if have else NAN, vals[0], vals[1])


def _serve(pe: pd.DataFrame) -> dict:
    s = pe[pe["phase"] == "serve"]
    if s.empty:
        return {"serve_n_requests": 0, "serve_dur_ms": NAN, "serve_E_mJ": NAN,
                "serve_idle_dur_ms": NAN, "serve_idle_E_mJ": NAN}
    idle = s[s["step"] == IDLE_STEP]
    # A request is a numeric iter with at least one named step. The recorder also leaves a single
    # sample with no step carrying the last measured frame's iter at the start of the serve phase;
    # it is part of the serve totals but is not a request.
    it = pd.to_numeric(s.loc[s["step"].notna(), "iter"], errors="coerce").dropna()
    return {"serve_n_requests": int(it.nunique()),
            "serve_dur_ms": float(s["dur_ms"].sum()), "serve_E_mJ": float(s["E_mJ"].sum()),
            "serve_idle_dur_ms": float(idle["dur_ms"].sum()), "serve_idle_E_mJ": float(idle["E_mJ"].sum())}


def derive_cell(cell) -> list[dict]:
    """One dict per measured mode of the cell (see COLUMNS); [] if the cell has no phase_energy."""
    cell = Path(cell)
    pe = data.read_csv_opt(cell, "phase_energy.csv")
    if pe is None:
        return []
    meta = data.read_meta(cell)
    batch = int(meta.get("batch") or 1)
    slips = slip_times_s(data.read_json_opt(cell, "alignment.json"), meta)
    base_w, base_a, base_b = _baselines(pe)
    common = {"cell_id": cell.name, "batch": batch, "baseline_W": base_w, "baseline_A_W": base_a,
              "baseline_B_W": base_b, **_serve(pe),
              "trace_E_J": float(pe["E_mJ"].sum()) / 1000.0,
              "trace_dur_s": float(pe["dur_ms"].sum()) / 1000.0, "slips_total": len(slips)}
    tables = {m: frame_table(pe, m) for m in meta.get("modes") or []}
    all_frames = pd.concat(tables.values()) if tables else pd.DataFrame(columns=["t_start_s", "t_end_s"])
    # slips overlapping any measured frame of any mode
    in_scored = sum(bool(slip_overlaps(all_frames, [s]).any()) for s in slips)
    out = []
    for mode, t in tables.items():
        row = {**common, "mode": mode, "n_frames": len(t), "slips_in_scored": in_scored}
        drop = slip_overlaps(t, slips)
        kept = t[~drop]
        row["n_frames_slip_excluded"] = int(drop.sum())
        row["n_images"] = len(kept) * batch
        if t.empty:
            row.update(energy_mJ_per_image_raw=NAN, latency_ms_per_image_raw=NAN)
        else:
            row.update(energy_mJ_per_image_raw=float(t["E_mJ"].mean()) / batch,
                       latency_ms_per_image_raw=float(t["dur_ms"].mean()) / batch)
        if kept.empty:
            row.update(energy_mJ_per_image=NAN, latency_ms_per_image=NAN, mean_P_W=NAN,
                       dyn_energy_mJ_per_image=NAN)
        else:
            row.update(energy_mJ_per_image=float(kept["E_mJ"].mean()) / batch,
                       latency_ms_per_image=float(kept["dur_ms"].mean()) / batch,
                       mean_P_W=float(kept["E_mJ"].sum() / kept["dur_ms"].sum()),   # mJ / ms = W
                       dyn_energy_mJ_per_image=float((kept["E_mJ"] - base_w * kept["dur_ms"]).mean()) / batch)
        out.append({k: row[k] for k in COLUMNS})
    return out


def _cache_key(cat: pd.DataFrame) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(runlog.file_hash(Path(__file__)).encode())
    h.update(runlog.data_hash().encode())
    h.update("\n".join(sorted(cat["cell_id"].astype(str))).encode())
    return h.hexdigest()


def _read_cache(cache: Path, key: str) -> pd.DataFrame | None:
    """The cached table if it is readable and its key matches; any failure to read it is a miss."""
    if not cache.is_file():
        return None
    try:
        tbl = pq.read_table(cache)
    except (OSError, pa.ArrowException, ValueError):
        return None
    if (tbl.schema.metadata or {}).get(b"derive_key") != key.encode():
        return None
    return tbl.to_pandas()


def _write_cache(cache: Path, df: pd.DataFrame, key: str) -> None:
    """Write via a temp file in the same directory and os.replace, so a reader never sees a torn file."""
    tmp = cache.with_name(f"{cache.name}.tmp{os.getpid()}")
    try:
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl.replace_schema_metadata({"derive_key": key}), tmp)
        os.replace(tmp, cache)
    finally:
        tmp.unlink(missing_ok=True)


def derive_all(cat: pd.DataFrame, force: bool = False, cache: bool = False) -> pd.DataFrame:
    """derive_cell over every cell in the catalogue: one row per cell and mode.

    Cache rule: derived_dir()/derive.parquet holds the derivation of the FULL catalogue and is read
    and written only when the caller passes cache=True, which only the CLI commands and the pipelines
    that pass `data.catalogue()` do. The default cache=False (subset calls, tests) neither reads nor
    writes it, so a subset call can never replace the full-catalogue cache. The key is the hash of
    derive.py, the dataset index hash and the cell list, stored in the parquet metadata. An unreadable
    or mismatching cache is a miss; the write is atomic (temp file + os.replace). force=True
    recomputes (and, with cache=True, rewrites)."""
    path = settings.derived_dir() / CACHE_NAME if cache else None
    key = _cache_key(cat) if cache else ""
    if cache and not force:
        hit = _read_cache(path, key)
        if hit is not None:
            return hit
    rows = [r for p in cat["path"] for r in derive_cell(p)]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if cache:
        _write_cache(path, df, key)
    return df
