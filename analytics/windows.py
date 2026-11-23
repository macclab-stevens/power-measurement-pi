"""100 ms windows (one per perf counter interval) and 5 s PMIC windows for the live power model.

Causality (spec 7.1): a window may only use what is known at its end. Every feature is built from
counters.csv / sysfs.csv / context.csv with an explicit as-of rule:
  * counters: the window's own interval and the 3 previous intervals of the same cell (lags are taken
    on the full interval sequence before any window is dropped, so a lag is always "the previous
    counter interval");
  * sysfs: latest row with t_real_ns <= t_end_ns;
  * PMIC: latest context read with t0_real_ns + t_ms * 1e6 <= t_end_ns.
counters.csv is already offset-corrected to the Pi clock (the perf start offset is applied by the
harness), so nothing is shifted here. The label (P_W, E_mJ, phase) is the only thing read from the
trace; the trace's own ctr_* / sys_* columns are never used.

Sample-to-interval assignment follows host/align.py merge_counters: a sample belongs to the interval
with t_start_ns < t_pi_ns <= t_end_ns (contiguous intervals therefore partition the samples). PMIC
windows use [t_i, t_{i+1}).
"""
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import data
import runlog
import settings

log = logging.getLogger(__name__)

WINDOWS_CACHE = "windows.parquet"
PMIC_CACHE = "pmic_windows.parquet"
MIN_COVER = 0.8             # windows with cover_frac below this are dropped
N_CORES = 4
TASK_CLOCK_UNIT = 1e9       # ctr_task_clock is in nanoseconds (verified on real data, see PROTOCOL / report)
LAG_COLS = ("busy", "cycles_rate", "ipc")
LAGS = (1, 2, 3)
SYSFS_COLS = ["freq_mhz", "temp_C", "fan_state", "uv_alarm", "sysfs_age_s"]
GUARDED = ["ipc", "cache_miss_rate", "branch_mpki", "stall_frac", "llc_mpki"]
WINDOW_COLS = (
    ["cell_id", "w_idx", "t_start_ns", "t_end_ns", "dur_s", "n_samples", "cover_frac", "P_W", "E_mJ",
     "phase", "phase_frac", "busy", "cycles_rate", "instr_rate", "ipc", "cache_miss_rate", "branch_mpki",
     "stall_frac", "llc_mpki", "ctx_sw_rate", "pf_rate"]
    + [f"{c}_l{k}" for c in LAG_COLS for k in LAGS]
    + ["pct_min"] + SYSFS_COLS + ["pmic_W", "pmic_age_s", "cpu_key", "fan_mode", "governor"])
PMIC_COLS = ["cell_id", "t_start_ns", "t_end_ns", "dur_s", "P_W", "E_mJ", "phase", "pmic_W",
             "cpu_key", "fan_mode", "governor"]
_RE_A = re.compile(r"^rail_(.+)_A current\(\d+\)$")
_RE_V = re.compile(r"^rail_(.+)_V volt\(\d+\)$")
NAN = float("nan")


# ------------------------------------------------------------------ small helpers

def settings_of(meta: dict) -> dict:
    """cpu_key, fan_mode, governor from run_meta (v1 defaults: pinned, full); cpu_key as in deploy.py."""
    gov = meta.get("governor") or "pinned"
    mhz = meta.get("cpu_mhz_pinned")
    key = "ondemand" if gov == "ondemand" else (str(int(mhz)) if mhz is not None else "unknown")
    return {"cpu_key": key, "fan_mode": meta.get("fan_mode") or "full", "governor": gov}


def sample_dt_s(cell, meta: dict) -> float:
    """1 / alignment.json slip_rate_hz, falling back to run_meta achieved_sample_hz."""
    al = data.read_json_opt(cell, "alignment.json") or {}
    hz = al.get("slip_rate_hz") or meta.get("achieved_sample_hz")
    if not hz:
        raise ValueError(f"{cell}: neither alignment.json slip_rate_hz nor run_meta achieved_sample_hz")
    return 1.0 / float(hz)


def assign_intervals(t: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample index range [lo, hi) of the sorted time array `t` in each interval (t_start, t_end]
    (the host/align.py merge_counters rule)."""
    return np.searchsorted(t, starts, side="right"), np.searchsorted(t, ends, side="right")


def _read_trace(cell):
    """(t_pi_ns int64, P_W float64, phase codes int32 (-1 = null), phase names), sorted by time.

    Reads only these three columns, phase dictionary-encoded, to keep a 9 M-row soak small."""
    tbl = pq.read_table(Path(cell) / "power_trace.parquet", columns=["t_pi_ns", "P_W", "phase"],
                        read_dictionary=["phase"])
    t = tbl.column("t_pi_ns").to_numpy()
    P = tbl.column("P_W").to_numpy().astype(np.float64)
    ph = tbl.column("phase")
    arr = ph.unify_dictionaries().combine_chunks() if ph.num_chunks > 1 else ph.chunk(0)
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    codes = arr.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int32)
    names = [str(x) for x in arr.dictionary.to_pylist()]
    if len(t) > 1 and (np.diff(t) < 0).any():
        o = np.argsort(t, kind="stable")
        t, P, codes = t[o], P[o], codes[o]
    return t, P, codes, names


def _label(t, P, codes, names, lo, hi, dt_s):
    """Per [lo, hi) range: n_samples, mean P_W, E_mJ, majority phase and its sample fraction.

    Non-finite P_W samples are excluded from the mean and imputed with it in the energy (they are
    sensor dropouts, not zero power)."""
    n = (hi - lo).astype(np.int64)
    good = np.isfinite(P)
    cs = np.concatenate([[0.0], np.cumsum(np.where(good, P, 0.0))])
    cn = np.concatenate([[0], np.cumsum(good)])
    s, c = cs[hi] - cs[lo], (cn[hi] - cn[lo]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.where(c > 0, c, 1), NAN)
    energy = mean * n * dt_s * 1000.0
    phase, frac = _majority(codes, names, lo, hi)
    return n, mean, energy, phase, frac


def _majority(codes, names, lo, hi):
    """Majority label of each [lo, hi) range by run-length encoding the label (ties -> earliest run)."""
    nw = len(lo)
    out = np.full(nw, "", dtype=object)
    frac = np.full(nw, NAN)
    has = hi > lo
    if not has.any() or len(codes) == 0:
        return out, frac
    n = len(codes)
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    rs = np.concatenate([[0], change])
    re_ = np.concatenate([change, [n]])
    rc = codes[rs]
    a = np.searchsorted(rs, lo, side="right") - 1
    z = np.searchsorted(rs, np.maximum(hi - 1, 0), side="right") - 1
    cnt = np.where(has, z - a + 1, 0)
    win = np.repeat(np.arange(nw), cnt)
    run = a[win] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ov = np.minimum(hi[win], re_[run]) - np.maximum(lo[win], rs[run])
    g = (pd.DataFrame({"win": win, "code": rc[run], "ov": ov, "run": run})
         .groupby(["win", "code"], sort=False).agg(ov=("ov", "sum"), run=("run", "min")).reset_index()
         .sort_values(["win", "ov", "run"], ascending=[True, False, True]).drop_duplicates("win"))
    lab = np.array([names[c] if c >= 0 else "unlabelled" for c in g["code"]], dtype=object)
    out[g["win"].to_numpy()] = lab
    frac[g["win"].to_numpy()] = g["ov"].to_numpy() / (hi - lo)[g["win"].to_numpy()]
    return out, frac


# ------------------------------------------------------------------ as-of joins

def asof_index(times: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    """Index of the latest entry of sorted `times` that is <= t_end, -1 if none."""
    return np.searchsorted(times, t_end, side="right") - 1


def _sysfs_asof(cell, t_end):
    out = {c: np.full(len(t_end), NAN) for c in SYSFS_COLS}
    sy = data.read_csv_opt(cell, "sysfs.csv")
    if sy is None or sy.empty:
        return out
    sy = sy.sort_values("t_real_ns", kind="stable")
    tr = sy["t_real_ns"].to_numpy(dtype=np.int64)
    idx = asof_index(tr, t_end)
    ok = idx >= 0
    j = np.maximum(idx, 0)
    khz = np.column_stack([sy[f"cpu{i}_kHz"].to_numpy(dtype=np.float64) for i in range(4)])
    vals = {"freq_mhz": khz.mean(axis=1) / 1000.0, "temp_C": sy["temp_C"].to_numpy(dtype=np.float64),
            "fan_state": sy["fan_state"].to_numpy(dtype=np.float64),
            "uv_alarm": sy["uv_alarm"].to_numpy(dtype=np.float64)}
    for c, v in vals.items():
        out[c] = np.where(ok, v[j], NAN)
    out["sysfs_age_s"] = np.where(ok, (t_end - tr[j]) / 1e9, NAN)
    return out


def pmic_reads(cell, meta: dict):
    """(read times ns int64, pmic_W per read, n paired rails) from context.csv, or None if absent.

    Rails are paired by name: `rail_<NAME>_A current(n)` with `rail_<NAME>_V volt(m)`; rails present
    on only one side are ignored. Read time = t0_real_ns + t_ms * 1e6 (Pi clock)."""
    cx = data.read_csv_opt(cell, "context.csv")
    if cx is None or cx.empty or meta.get("t0_real_ns") is None:
        return None
    amps = {m.group(1): c for c in cx.columns if (m := _RE_A.match(c))}
    volts = {m.group(1): c for c in cx.columns if (m := _RE_V.match(c))}
    names = sorted(set(amps) & set(volts))
    w = np.zeros(len(cx))
    for nm in names:
        w = w + cx[amps[nm]].to_numpy(dtype=np.float64) * cx[volts[nm]].to_numpy(dtype=np.float64)
    t = int(meta["t0_real_ns"]) + np.round(cx["t_ms"].to_numpy(dtype=np.float64) * 1e6).astype(np.int64)
    o = np.argsort(t, kind="stable")
    return t[o], w[o], len(names)


# ------------------------------------------------------------------ counter features

def _ratio(num, den, scale=1.0):
    """num / den * scale, NaN where den == 0; returns (values, guard mask)."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    guard = den == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(guard, NAN, num / np.where(guard, 1.0, den) * scale), guard


def counter_features(ctr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features of each interval from its own counters only; (features, guard masks per ratio)."""
    dur = (ctr["t_end_ns"] - ctr["t_start_ns"]).to_numpy(dtype=np.float64) / 1e9
    g = lambda name: ctr["ctr_" + name].to_numpy(dtype=np.float64)   # noqa: E731
    f, guards = {}, {}
    f["busy"], _ = _ratio(g("task_clock"), dur * N_CORES * TASK_CLOCK_UNIT)
    f["cycles_rate"], _ = _ratio(g("cycles"), dur)
    f["instr_rate"], _ = _ratio(g("instructions"), dur)
    f["ipc"], guards["ipc"] = _ratio(g("instructions"), g("cycles"))
    f["cache_miss_rate"], guards["cache_miss_rate"] = _ratio(g("cache_misses"), g("cache_references"))
    f["branch_mpki"], guards["branch_mpki"] = _ratio(g("branch_misses"), g("instructions"), 1000.0)
    f["stall_frac"], guards["stall_frac"] = _ratio(g("stalled_cycles_backend"), g("cycles"))
    f["llc_mpki"], guards["llc_mpki"] = _ratio(g("LLC_load_misses"), g("instructions"), 1000.0)
    f["ctx_sw_rate"], _ = _ratio(g("context_switches"), dur)
    f["pf_rate"], _ = _ratio(g("page_faults"), dur)
    pct = [c for c in ctr.columns if c.startswith("pct_")]
    f["pct_min"] = ctr[pct].min(axis=1).to_numpy(dtype=np.float64) if pct else np.full(len(ctr), NAN)
    return pd.DataFrame(f), pd.DataFrame(guards)


# ------------------------------------------------------------------ cell windows

def cell_windows(cell, meta: dict) -> pd.DataFrame | None:
    """One row per counters.csv interval (see module docstring); None if the cell has no counters.

    attrs: dropped (cover_frac < 0.8), guards (kept windows with any zero-denominator ratio),
    guards_by_feature, pmic_pairs, dt_s, n_intervals, ctr_nan (kept windows whose counters are NaN:
    perf prints <not counted> for every event while the measured process is not running, mostly
    gaps and baselines; they are kept, with NaN counter features, and are not counted as guards)."""
    cell = Path(cell)
    if not (meta.get("perf_counters") or {}).get("enabled"):
        return None
    ctr = data.read_csv_opt(cell, "counters.csv")
    if ctr is None or ctr.empty:
        return None
    ctr = ctr.sort_values("t_end_ns", kind="stable").reset_index(drop=True)
    starts = ctr["t_start_ns"].to_numpy(dtype=np.int64)
    ends = ctr["t_end_ns"].to_numpy(dtype=np.int64)
    dt_s = sample_dt_s(cell, meta)

    feat, guard = counter_features(ctr)
    for c in LAG_COLS:                       # on the full interval sequence, before any drop
        for k in LAGS:
            feat[f"{c}_l{k}"] = feat[c].shift(k)

    t, P, codes, names = _read_trace(cell)
    lo, hi = assign_intervals(t, starts, ends)
    n, mean, energy, phase, pfrac = _label(t, P, codes, names, lo, hi, dt_s)
    dur_s = (ends - starts) / 1e9
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = np.where(dur_s > 0, n * dt_s / np.where(dur_s > 0, dur_s, 1), NAN)
    df = pd.DataFrame({"cell_id": cell.name, "w_idx": np.arange(len(ctr)), "t_start_ns": starts,
                       "t_end_ns": ends, "dur_s": dur_s, "n_samples": n, "cover_frac": cover,
                       "P_W": mean, "E_mJ": energy, "phase": phase, "phase_frac": pfrac})
    df = pd.concat([df, feat], axis=1)
    for c, v in _sysfs_asof(cell, ends).items():
        df[c] = v
    pm = pmic_reads(cell, meta)
    df["pmic_W"], df["pmic_age_s"] = NAN, NAN
    n_pairs = 0
    if pm is not None:
        rt, rw, n_pairs = pm
        idx = asof_index(rt, ends)
        ok, j = idx >= 0, np.maximum(idx, 0)
        df["pmic_W"] = np.where(ok, rw[j], NAN)
        df["pmic_age_s"] = np.where(ok, (ends - rt[j]) / 1e9, NAN)
    for k, v in settings_of(meta).items():
        df[k] = v

    keep = (df["cover_frac"] >= MIN_COVER - 1e-9).to_numpy()   # tolerance: 80 x 1 ms / 100 ms is 0.7999..., not a drop
    gk = guard[keep]
    df = df[keep][WINDOW_COLS].reset_index(drop=True)
    df.attrs.update(dropped=int((~keep).sum()), guards=int(gk.any(axis=1).sum()),
                    guards_by_feature={c: int(gk[c].sum()) for c in GUARDED}, pmic_pairs=n_pairs,
                    dt_s=dt_s, n_intervals=len(ctr), ctr_nan=int(df["busy"].isna().sum()))
    return df


def pmic_windows(cell, meta: dict) -> pd.DataFrame:
    """One row per pair of consecutive context reads [t_i, t_{i+1}) with the trace mean power, the
    energy, the majority phase and the PMIC power of read i (known at the window start). Empty frame
    if the cell has no context reads. Works for v1 and v2 cells."""
    cell = Path(cell)
    pm = pmic_reads(cell, meta)
    out = pd.DataFrame(columns=PMIC_COLS)
    if pm is None or len(pm[0]) < 2:
        out.attrs["pmic_pairs"] = 0 if pm is None else pm[2]
        return out
    rt, rw, n_pairs = pm
    dt_s = sample_dt_s(cell, meta)
    t, P, codes, names = _read_trace(cell)
    lo, hi = np.searchsorted(t, rt[:-1], side="left"), np.searchsorted(t, rt[1:], side="left")
    n, mean, energy, phase, _ = _label(t, P, codes, names, lo, hi, dt_s)
    keep = n > 0
    df = pd.DataFrame({"cell_id": cell.name, "t_start_ns": rt[:-1], "t_end_ns": rt[1:],
                       "dur_s": (rt[1:] - rt[:-1]) / 1e9, "P_W": mean, "E_mJ": energy, "phase": phase,
                       "pmic_W": rw[:-1]})
    for k, v in settings_of(meta).items():
        df[k] = v
    df = df[keep][PMIC_COLS].reset_index(drop=True)
    df.attrs.update(pmic_pairs=n_pairs, empty=int((~keep).sum()))
    return df


# ------------------------------------------------------------------ cached builders

def _cache_key(cat: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(runlog.file_hash(Path(__file__)).encode())
    h.update(runlog.data_hash().encode())
    h.update("\n".join(sorted(cat["cell_id"].astype(str))).encode())
    return h.hexdigest()


def _load_cache(path: Path, key: str):
    """The cached frame, or None on a missing, unreadable or stale cache."""
    try:
        if not path.is_file():
            return None
        tbl = pq.read_table(path)
        md = tbl.schema.metadata or {}
        if md.get(b"windows_key") != key.encode():
            return None
        df = tbl.to_pandas()
        df.attrs.update(json.loads(md.get(b"windows_attrs", b"{}")))
        return df
    except Exception as exc:   # noqa: BLE001 - any unreadable cache is a miss
        log.warning("ignoring unreadable cache %s: %s", path, exc)
        return None


def _save_cache(path: Path, df: pd.DataFrame, key: str) -> None:
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.replace_schema_metadata({"windows_key": key, "windows_attrs": json.dumps(df.attrs)})
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        pq.write_table(tbl, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KiB on Linux


def _build(cat: pd.DataFrame, fn, cache_name: str, force: bool, columns: list[str]) -> pd.DataFrame:
    path = settings.derived_dir() / cache_name
    key = _cache_key(cat)
    if not force:
        hit = _load_cache(path, key)
        if hit is not None:
            return hit
    parts, attrs = [], {"dropped": 0, "guards": 0, "guards_by_feature": {c: 0 for c in GUARDED},
                        "pmic_pairs": {}, "n_cells": 0, "n_intervals": 0, "empty": 0, "ctr_nan": 0}
    for _, row in cat.iterrows():           # one cell at a time: only that cell's trace is in memory
        cell = Path(row["path"])
        df = fn(cell, data.read_meta(cell))
        if df is None or df.empty:
            continue
        parts.append(df.assign(cpu_key="ondemand" if row["governor"] == "ondemand" else str(int(row["cpu_mhz"])),
                               fan_mode=row["fan_mode"], governor=row["governor"]))
        a = df.attrs
        attrs["n_cells"] += 1
        attrs["dropped"] += a.get("dropped", 0)
        attrs["guards"] += a.get("guards", 0)
        attrs["n_intervals"] += a.get("n_intervals", 0)
        attrs["empty"] += a.get("empty", 0)
        attrs["ctr_nan"] += a.get("ctr_nan", 0)
        for c, v in a.get("guards_by_feature", {}).items():
            attrs["guards_by_feature"][c] += v
        k = str(a.get("pmic_pairs", 0))
        attrs["pmic_pairs"][k] = attrs["pmic_pairs"].get(k, 0) + 1
        log.info("%s: %d rows, peak RSS %.0f MB", cell.name, len(df), _peak_rss_mb())
    out = pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
    attrs["peak_rss_mb"] = round(_peak_rss_mb())
    out.attrs.update(attrs)
    _save_cache(path, out, key)
    return out


def build_windows(cat: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """cell_windows over every catalogue cell with counters, cached in derived/windows.parquet.

    cpu_key, fan_mode and governor are taken from the catalogue row. The cache key is the hash of
    windows.py, the dataset index hash and the cell list; an unreadable cache is a miss and the
    cache is written atomically."""
    return _build(cat, cell_windows, WINDOWS_CACHE, force, WINDOW_COLS)


def build_pmic_windows(cat: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """pmic_windows over every catalogue cell (v1 and v2), cached in derived/pmic_windows.parquet."""
    return _build(cat, pmic_windows, PMIC_CACHE, force, PMIC_COLS)
