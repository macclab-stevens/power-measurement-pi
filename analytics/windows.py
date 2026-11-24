"""100 ms windows (one per perf counter interval) and 5 s PMIC windows for the live power model.

Causality (spec 7.1): a window may only use what is known at its end. Every feature is built from
counters.csv / sysfs.csv / context.csv with an explicit as-of rule:
  * counters: the window's own interval and the 3 previous intervals of the same cell (lags are taken
    on the full interval sequence before any window is dropped, so a lag is always "the previous
    counter interval");
  * sysfs: latest row with t_real_ns <= t_end_ns;
  * PMIC: latest context read whose availability time t_avail <= t_end_ns. context.csv `t_ms` is taken
    BEFORE the read starts (src/powersampler.py: _now(), then vcgencmd calls and the full dump, then a
    5 s sleep), so a read is only known once it finished. Reads are dated by availability: for a read
    with a successor, t_avail_i = t_{i+1} - 5000 ms (the loop sleeps 5 s after a read finishes). The
    last read of a cell has no known availability time and is never used (later windows keep the
    previous read, with a larger pmic_age_s). If the computed t_avail_i < t_i (clock oddity) it is
    t_i + 5000 ms (the next read cannot have finished earlier) and the event is counted in
    attrs["pmic_avail_fallback"]. t_i is t0_real_ns + t_ms * 1e6. pmic_age_s = (t_end - t_avail) / 1e9.
counters.csv is already offset-corrected to the Pi clock (the perf start offset is applied by the
harness), so nothing is shifted here. The label (P_W, E_mJ, phase) is the only thing read from the
trace; the trace's own ctr_* / sys_* columns are never used.

Sample-to-interval assignment follows host/align.py merge_counters: a sample belongs to the interval
with t_start_ns < t_pi_ns <= t_end_ns (contiguous intervals therefore partition the samples). PMIC
windows use [t_i, t_{i+1}). A window's features are available at its t_end_ns, and the causality tests
state that rule.

Not-counted intervals: perf prints <not counted> for every event when the attached process was never
scheduled in the interval, so its true counts are zero. Such an interval has counted == 0; its count-
derived rates (busy, cycles_rate, instr_rate, ctx_sw_rate, pf_rate) are 0.0, and the ratio features
(ipc, cache_miss_rate, branch_mpki, stall_frac, llc_mpki) stay NaN because they are undefined. Lags are
taken from the filled values.

Slip correction (R2): cells whose alignment.json lists sample-loss slips and lacks `"slip_corrected": true`
(collected before host/align.py removed the steps from its clock fit) get the step-model correction of
timefix.py added to their trace sample times BEFORE samples are assigned to windows, so the window power labels
(P_W, E_mJ, phase) sit on the corrected time axis. Cells without >= 12 sync pulse events are left as stored and
counted (attrs["slip_no_pulses"]). The published traces are never rewritten. attrs["slip_cells"] lists, per
corrected cell, how much the labels moved. Features never use the trace, so they are unaffected.

R2 features (CMOS dynamic power ~ V^2 f): `llc_miss_rate` (LLC_load_misses / s), `f_eff_mhz` (cycles_rate /
(N_CORES * busy) / 1e6 when busy >= 0.03, else the sysfs clock `freq_mhz`; clamped to [600, 2400]; NaN only when
both are unavailable), `vcore_eff` (DVFS table of config.toml at f_eff, tables.vcore_v) and `v2_cycles_rate`,
`v2_instr_rate` (vcore_eff^2 x the rate). Not-counted windows have rate 0 and so v2 rates 0 (a zero rate
gives 0 whatever the voltage), like busy.

PMIC rails: `rail_<NAME>_A current(n)` is paired with `rail_<NAME>_V volt(m)` by name. The Pi 5 PMIC
exposes 12 such pairs; EXT5V and BATT have a voltage but no current and are ignored. The pair count is
stored in attrs["pmic_pairs"].
"""
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import data
import runlog
import settings
import tables
import timefix

log = logging.getLogger(__name__)

WINDOWS_CACHE = "windows_r2.parquet"          # R2 caches have new names: the R1 caches stay for the frozen reports
PMIC_CACHE = "pmic_windows_r2.parquet"
CACHE_VERSION = "r2-slipfix-cmos-1"
MIN_COVER = 0.8             # windows with cover_frac below this are dropped
N_CORES = 4
BUSY_FEFF_MIN = 0.03        # below this the cycles-based clock is too noisy: use the sysfs clock
F_EFF_RANGE = (600.0, 2400.0)
CMOS_COLS = ["f_eff_mhz", "vcore_eff", "v2_cycles_rate", "v2_instr_rate"]
TASK_CLOCK_UNIT = 1e9       # ctr_task_clock is in nanoseconds (verified on real data, see PROTOCOL / report)
LAG_COLS = ("busy", "cycles_rate", "ipc")
LAGS = (1, 2, 3)
SYSFS_COLS = ["freq_mhz", "temp_C", "fan_state", "uv_alarm", "sysfs_age_s"]
GUARDED = ["ipc", "cache_miss_rate", "branch_mpki", "stall_frac", "llc_mpki"]
WINDOW_COLS = (
    ["cell_id", "w_idx", "t_start_ns", "t_end_ns", "dur_s", "n_samples", "cover_frac", "P_W", "E_mJ",
     "phase", "phase_frac", "counted", "busy", "cycles_rate", "instr_rate", "ipc", "cache_miss_rate", "branch_mpki",
     "stall_frac", "llc_mpki", "ctx_sw_rate", "pf_rate", "llc_miss_rate"]
    + [f"{c}_l{k}" for c in LAG_COLS for k in LAGS]
    + ["pct_min"] + SYSFS_COLS + CMOS_COLS + ["pmic_W", "pmic_age_s", "cpu_key", "fan_mode", "governor"])
PMIC_COLS = ["cell_id", "t_read_ns", "t_start_ns", "t_end_ns", "dur_s", "P_W", "E_mJ", "phase", "pmic_W",
             "cpu_key", "fan_mode", "governor"]
_RE_A = re.compile(r"^rail_(.+)_A current\(\d+\)$")
_RE_V = re.compile(r"^rail_(.+)_V volt\(\d+\)$")
CONTEXT_PERIOD_NS = 5_000_000_000   # src/powersampler.py CONTEXT_PERIOD_S: sleep after each read finishes
NEVER_AVAILABLE = np.iinfo(np.int64).max   # availability of a cell's last read: no known time, never used
LAST_WINDOW_MARGIN_NS = 500_000_000        # pmic_windows only: label bound after the last read's t_ms (not a feature)
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


def read_availability(t_read: np.ndarray) -> tuple[np.ndarray, int]:
    """(t_avail ns, n_fallback) for sorted read start times (see the module docstring).

    t_avail_i = t_{i+1} - 5000 ms, or t_i + 5000 ms when that is earlier than t_i (counted); the last
    read is NEVER_AVAILABLE. Forced non-decreasing so it can be searched."""
    t = np.asarray(t_read, dtype=np.int64)
    avail = np.full(len(t), NEVER_AVAILABLE, dtype=np.int64)
    n_fallback = 0
    if len(t) > 1:
        cand = t[1:] - CONTEXT_PERIOD_NS
        bad = cand < t[:-1]
        avail[:-1] = np.where(bad, t[:-1] + CONTEXT_PERIOD_NS, cand)
        n_fallback = int(bad.sum())
    return np.maximum.accumulate(avail), n_fallback


class PmicReads(NamedTuple):
    t_read: np.ndarray    # t0_real_ns + t_ms * 1e6, ns (start of the read)
    t_avail: np.ndarray   # when the read finished and is usable, ns (last read: NEVER_AVAILABLE)
    pmic_W: np.ndarray    # sum of current x voltage over the paired rails, per read
    n_pairs: int
    n_fallback: int


def pmic_reads(cell, meta: dict) -> PmicReads | None:
    """PMIC reads of context.csv sorted by time, or None if the cell has none.

    Rails are paired by name: `rail_<NAME>_A current(n)` with `rail_<NAME>_V volt(m)`; rails present
    on only one side are ignored. Read start = t0_real_ns + t_ms * 1e6 (Pi clock); availability time
    from read_availability."""
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
    t, w = t[o], w[o]
    avail, n_fb = read_availability(t)
    return PmicReads(t, avail, w, len(names), n_fb)


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
    f["llc_miss_rate"], _ = _ratio(g("LLC_load_misses"), dur)
    counted = np.isfinite(ctr[[c for c in ctr.columns if c.startswith("ctr_")]].to_numpy(dtype=np.float64)).any(axis=1)
    for c in ("busy", "cycles_rate", "instr_rate", "ctx_sw_rate", "pf_rate", "llc_miss_rate"):
        f[c] = np.where(counted, f[c], 0.0)       # not counted: the process never ran, true count is zero
    f["counted"] = counted.astype(np.int64)
    pct = [c for c in ctr.columns if c.startswith("pct_")]
    f["pct_min"] = ctr[pct].min(axis=1).to_numpy(dtype=np.float64) if pct else np.full(len(ctr), NAN)
    return pd.DataFrame(f), pd.DataFrame(guards)


def cmos_features(df: pd.DataFrame) -> pd.DataFrame:
    """The R2 CMOS features f_eff_mhz, vcore_eff, v2_cycles_rate, v2_instr_rate (see the module docstring)
    from the busy / cycles_rate / instr_rate / counted / freq_mhz columns of `df`; a copy with the four
    columns added.

    f_eff = cycles_rate / (N_CORES * busy) / 1e6 where busy >= 0.03 (so busy = 0, including every
    not-counted window, never divides), else the sysfs clock; clamped to [600, 2400]; NaN stays NaN. A
    rate of exactly 0 gives a v2 rate of 0 even when the voltage is unknown."""
    busy = df["busy"].to_numpy(dtype=np.float64)
    cyc = df["cycles_rate"].to_numpy(dtype=np.float64)
    ok = np.isfinite(busy) & (busy >= BUSY_FEFF_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        from_cycles = cyc / (N_CORES * np.where(ok, busy, 1.0)) / 1e6
    f_eff = np.clip(np.where(ok, from_cycles, df["freq_mhz"].to_numpy(dtype=np.float64)), *F_EFF_RANGE)
    v = tables.vcore_v(f_eff)                  # np.interp: NaN in, NaN out
    out = df.copy()
    out["f_eff_mhz"], out["vcore_eff"] = f_eff, v
    for name, col in (("v2_cycles_rate", "cycles_rate"), ("v2_instr_rate", "instr_rate")):
        rate = df[col].to_numpy(dtype=np.float64)
        out[name] = np.where(rate == 0, 0.0, v ** 2 * rate)
    return out


# ------------------------------------------------------------------ slip-corrected trace

def _pulse_times(cell) -> np.ndarray:
    """Pi-clock times (ns) of the sync pulse edges of events.csv(.gz); empty if there is no event log."""
    for name in ("events.csv", "events.csv.gz"):
        p = Path(cell) / name
        if p.is_file():
            ev = pd.read_csv(p, usecols=["t_real_ns", "kind"])
            return ev.loc[ev["kind"].isin(timefix.PULSE_KINDS), "t_real_ns"].to_numpy(dtype=np.float64)
    return np.empty(0)


def corrected_trace(cell, correct: bool = True):
    """`_read_trace(cell)` with the slip correction applied to the sample times where the cell needs it.

    Returns (t, P, codes, names, info). `info` is {"status": "none" | "corrected" | "no_pulses", "max_corr_ms"}
    ("none": no slips, or alignment.json already says slip_corrected). The corrected times are rounded to
    int64 ns; the trace is re-sorted by time only if the correction reordered samples (it does not for
    slip steps, which only move later samples later)."""
    t, P, codes, names = _read_trace(cell)
    info = {"status": "none", "max_corr_ms": 0.0}
    al = data.read_json_opt(cell, "alignment.json")
    if not correct or not timefix.needs_correction(al):
        return t, P, codes, names, info
    corr = timefix.slip_correction_ns(al, _pulse_times(cell), t)
    if corr is None:
        info["status"] = "no_pulses"
        return t, P, codes, names, info
    t = t + np.round(corr).astype(np.int64)
    if len(t) > 1 and (np.diff(t) < 0).any():
        o = np.argsort(t, kind="stable")
        t, P, codes = t[o], P[o], codes[o]
    info.update(status="corrected", max_corr_ms=float(np.abs(corr).max() / 1e6))
    return t, P, codes, names, info


# ------------------------------------------------------------------ cell windows

def cell_windows(cell, meta: dict, slip_correct: bool = True) -> pd.DataFrame | None:
    """One row per counters.csv interval (see module docstring); None if the cell has no counters.

    attrs: dropped (cover_frac < 0.8), guards (kept windows with any zero-denominator ratio),
    guards_by_feature, pmic_pairs, dt_s, n_intervals, not_counted (kept
    windows with counted == 0, see the module docstring; not counted as guards), slip (status of the slip
    correction: "none" / "corrected" / "no_pulses"; for a corrected cell also the per-cell size of the label
    change, see `_slip_effect`). `slip_correct=False` reproduces the labels on the stored time axis."""
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

    t, P, codes, names, slip = corrected_trace(cell, slip_correct)
    lo, hi = assign_intervals(t, starts, ends)
    n, mean, energy, phase, pfrac = _label(t, P, codes, names, lo, hi, dt_s)
    if slip["status"] == "corrected":       # how far the labels moved, on the same windows
        t0_, P0_, c0_, nm0_, _ = corrected_trace(cell, False)
        lo0, hi0 = assign_intervals(t0_, starts, ends)
        slip["_mean0"] = _label(t0_, P0_, c0_, nm0_, lo0, hi0, dt_s)[1]
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
        n_pairs = pm.n_pairs
        idx = asof_index(pm.t_avail, ends)          # latest read already finished at t_end
        ok, j = idx >= 0, np.maximum(idx, 0)
        df["pmic_W"] = np.where(ok, pm.pmic_W[j], NAN)
        df["pmic_age_s"] = np.where(ok, (ends - pm.t_avail[j]) / 1e9, NAN)
    for k, v in settings_of(meta).items():
        df[k] = v
    df = cmos_features(df)

    keep = (df["cover_frac"] >= MIN_COVER - 1e-9).to_numpy()   # tolerance: 80 x 1 ms / 100 ms is 0.7999..., not a drop
    gk = guard[keep]
    slip_attr = _slip_effect(cell.name, slip, df["P_W"].to_numpy(), keep)
    df = df[keep][WINDOW_COLS].reset_index(drop=True)
    df.attrs.update(dropped=int((~keep).sum()), guards=int(gk.any(axis=1).sum()),
                    guards_by_feature={c: int(gk[c].sum()) for c in GUARDED}, pmic_pairs=n_pairs,
                    pmic_avail_fallback=0 if pm is None else pm.n_fallback, dt_s=dt_s, n_intervals=len(ctr), not_counted=int((df["counted"] == 0).sum()),
                    slip=slip_attr)
    return df


def _slip_effect(cell_id: str, slip: dict, p_corrected: np.ndarray, keep: np.ndarray) -> dict:
    """The attrs["slip"] record: {"status", and for a corrected cell cell_id, max_corr_ms, n_windows and the
    median / p99 / max of |P_W(corrected axis) - P_W(stored axis)| in W over the kept windows with both finite}."""
    out = {"status": slip["status"]}
    if slip["status"] == "corrected":
        d = np.abs(p_corrected - slip["_mean0"])[keep]
        d = d[np.isfinite(d)]
        out.update(cell_id=cell_id, max_corr_ms=slip["max_corr_ms"], n_windows=int(len(d)),
                   dP_med_W=float(np.median(d)) if len(d) else NAN,
                   dP_p99_W=float(np.quantile(d, 0.99)) if len(d) else NAN,
                   dP_max_W=float(d.max()) if len(d) else NAN,
                   n_windows_changed=int((d > 1e-9).sum()), n_windows_gt_0p1W=int((d > 0.1).sum()))
    return out


def pmic_windows(cell, meta: dict) -> pd.DataFrame:
    """One row per pair of consecutive PMIC reads: window i is [t_avail_i, t_avail_{i+1}) with the
    trace mean power, the energy, the majority phase and the PMIC power of read i (finished at the
    window start, so a causal feature). t_start_ns / t_end_ns are the availability bounds and
    t_read_ns is the raw t_ms-based start of read i. The last read has no window of its own; the
    window of the second-to-last read ends 500 ms after the last read's t_ms (a label bound only).
    Empty frame if the cell has fewer than 2 reads. Works for v1 and v2 cells."""
    cell = Path(cell)
    pm = pmic_reads(cell, meta)
    out = pd.DataFrame(columns=PMIC_COLS)
    if pm is None or len(pm.t_read) < 2:
        out.attrs.update(pmic_pairs=0 if pm is None else pm.n_pairs, pmic_avail_fallback=0, empty=0)
        return out
    ta = pm.t_avail.copy()
    ta[-1] = max(int(pm.t_read[-1]) + LAST_WINDOW_MARGIN_NS, int(ta[-2]))   # label bound, not a feature
    dt_s = sample_dt_s(cell, meta)
    t, P, codes, names, _ = corrected_trace(cell)
    lo, hi = np.searchsorted(t, ta[:-1], side="left"), np.searchsorted(t, ta[1:], side="left")
    n, mean, energy, phase, _ = _label(t, P, codes, names, lo, hi, dt_s)
    keep = n > 0
    df = pd.DataFrame({"cell_id": cell.name, "t_read_ns": pm.t_read[:-1], "t_start_ns": ta[:-1],
                       "t_end_ns": ta[1:], "dur_s": (ta[1:] - ta[:-1]) / 1e9, "P_W": mean,
                       "E_mJ": energy, "phase": phase, "pmic_W": pm.pmic_W[:-1]})
    for k, v in settings_of(meta).items():
        df[k] = v
    df = df[keep][PMIC_COLS].reset_index(drop=True)
    df.attrs.update(pmic_pairs=pm.n_pairs, pmic_avail_fallback=pm.n_fallback, empty=int((~keep).sum()))
    return df


# ------------------------------------------------------------------ cached builders

def _cache_key(cat: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(runlog.file_hash(Path(__file__)).encode())
    h.update(runlog.file_hash(Path(timefix.__file__)).encode())
    h.update(json.dumps(settings.cfg().get("dvfs", {}), sort_keys=True).encode())    # vcore_eff / v2_* depend on it
    h.update(runlog.data_hash(sources=("v1", "v2")).encode())      # lockbox cells never enter a window table
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


def assign_settings(df: pd.DataFrame, row) -> pd.DataFrame:
    """`df` with cpu_key, fan_mode and governor taken from the catalogue row `row` (the one place that decides them, for the
    cached builders and for the lockbox windows): cpu_key is `ondemand` for ondemand cells, else the pinned clock in MHz."""
    return df.assign(cpu_key="ondemand" if row["governor"] == "ondemand" else str(int(row["cpu_mhz"])),
                     fan_mode=row["fan_mode"], governor=row["governor"])


def _build(cat: pd.DataFrame, fn, cache_name: str, force: bool, columns: list[str]) -> pd.DataFrame:
    path = settings.derived_dir() / cache_name
    cat = data.training_cells(cat)
    key = _cache_key(cat)
    if not force:
        hit = _load_cache(path, key)
        if hit is not None:
            return hit
    parts, attrs = [], {"dropped": 0, "guards": 0, "guards_by_feature": {c: 0 for c in GUARDED},
                        "pmic_pairs": {}, "n_cells": 0, "n_intervals": 0, "empty": 0, "not_counted": 0,
                        "pmic_avail_fallback": 0, "slip_corrected": 0, "slip_no_pulses": 0, "slip_cells": []}
    for _, row in cat.iterrows():           # one cell at a time: only that cell's trace is in memory
        cell = Path(row["path"])
        df = fn(cell, data.read_meta(cell))
        if df is None or df.empty:
            continue
        parts.append(assign_settings(df, row))
        a = df.attrs
        attrs["n_cells"] += 1
        attrs["dropped"] += a.get("dropped", 0)
        attrs["guards"] += a.get("guards", 0)
        attrs["n_intervals"] += a.get("n_intervals", 0)
        attrs["empty"] += a.get("empty", 0)
        attrs["not_counted"] += a.get("not_counted", 0)
        attrs["pmic_avail_fallback"] += a.get("pmic_avail_fallback", 0)
        sl = a.get("slip") or {}
        attrs["slip_corrected"] += sl.get("status") == "corrected"
        attrs["slip_no_pulses"] += sl.get("status") == "no_pulses"
        if sl.get("status") == "corrected":
            attrs["slip_cells"].append({k: v for k, v in sl.items() if k != "status"})
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
    """cell_windows over every catalogue cell with counters, cached in derived/windows_r2.parquet.

    cpu_key, fan_mode and governor are taken from the catalogue row. The cache key is the hash of
    windows.py and timefix.py, the DVFS table, the dataset index hash and the cell list; an unreadable cache is a miss and the
    cache is written atomically."""
    return _build(cat, cell_windows, WINDOWS_CACHE, force, WINDOW_COLS)


def build_pmic_windows(cat: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """pmic_windows over every catalogue cell (v1 and v2), cached in derived/pmic_windows_r2.parquet."""
    return _build(cat, pmic_windows, PMIC_CACHE, force, PMIC_COLS)
