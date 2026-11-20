"""Hardware counters via `perf stat -I` attached to the run.py process.

`perf stat -I <ms> -x , -o counters.raw.txt -e <events> -p <pid>` prints one block of lines per
interval: `time,count,unit,event,run_time_ns,pct_running[,metric,metric_unit]`. `time` is seconds
since perf started counting; `PerfStat.start()` records `t0_ns = time.time_ns()` (the lab wall
clock) immediately before spawning perf, so every interval end becomes
`t_end_ns = t0_ns + time*1e9` and counters.csv joins to power_trace.parquet on `t_pi_ns` with
pandas.merge_asof. perf only starts its interval clock after it has initialised (~33 ms measured
on the Pi), so start() polls the raw file until the first complete interval line appears and
derives `offset_ns = t_first_seen - t0 - first_time` (>= 0); all timestamps are then
`t0_ns + offset_ns + time*1e9`. If no line shows up within 2 s, offset_ns is 0 (method "none").

Facts about perf on the Pi 5 (6.18.60, perf_event_paranoid=2 => user-space only, events print as
`cycles:u`): the 10 default events (7 hardware + 3 software) fit the PMU without multiplexing, but
each interval still reports `pct_running` (the share of the interval an event was actually counted;
the count is already scaled up by perf), and more hardware events than counters would multiplex;
an event that was never scheduled prints `<not counted>`. perf also prints metric-only lines with
an empty event field (`0.1,,,,,,0.12,stalled cycles per insn`); those are skipped. With `-o` perf
writes each interval as it happens, and a SIGINT makes it flush a final partial interval and exit,
so the whole raw file can simply be parsed after stop().

Usage:
  ps = PerfStat(os.getpid(), out_path="counters.raw.txt"); ps.start() ... ps.stop()
  write_counters_csv("counters.csv", ps.wide_rows()); meta = ps.meta(enabled=True)
Tests drive everything through the injectable `popen`, `clock` and `which`.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import subprocess
import time

# The A76 PMU has the cycle counter plus 6 programmable counters. These 7 hardware events run
# 100 % of the time with no multiplexing (measured on the Pi); an 8th hardware event (LLC-loads)
# drops every event to 71-90 %, which would break the 95 % coverage gate. L1-dcache-loads and
# L1-dcache-load-misses counted identically to cache-references / cache-misses on this core
# (duplicates), so they are dropped, as are LLC-loads and bus-cycles.
DEFAULT_EVENTS = ("cycles", "instructions", "cache-references", "cache-misses",
                  "branch-misses", "stalled-cycles-backend", "LLC-load-misses",
                  "task-clock", "context-switches", "page-faults")
SOFTWARE_EVENTS = ("task-clock", "context-switches", "page-faults")   # always 100 % running
STOP_TIMEOUT_S = 5.0
FIRST_LINE_POLL_S = 0.002       # how often start() looks for perf's first interval line
FIRST_LINE_GIVE_UP_S = 2.0
PARANOID_PATH = "/proc/sys/kernel/perf_event_paranoid"
_CLEAN_EXIT = (0, -signal.SIGINT, 128 + signal.SIGINT)   # perf re-raises SIGINT after flushing


def _col(event: str) -> str:
    """Event name as used in column names: `L1-dcache-loads` -> `L1_dcache_loads`."""
    return event.replace("-", "_")


def _parse_count(field: str) -> int | None:
    """`<not counted>` / `<not supported>` / junk -> None."""
    try:
        return int(field)
    except ValueError:
        pass
    try:
        return round(float(field))
    except ValueError:
        return None


def _parse_pct(field: str) -> float | None:
    try:
        return float(field)
    except ValueError:
        return None


def parse_interval_csv(text: str, t0_ns: int) -> list[dict]:
    """Parse `perf stat -I -x,` output into one row per (interval, event).

    Row keys: t_end_ns (t0_ns + time), interval_s (time since the previous interval; the first
    interval counts from 0), event (':u'/':k' modifier stripped), count (None if not
    counted/supported), pct_running (0-100, None if missing). Comments, blank lines, metric-only
    lines (empty event) and unparseable lines are skipped.
    """
    rows: list[dict] = []
    prev_s, cur_s = 0.0, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 4 or not f[3]:
            continue
        try:
            time_s = float(f[0])
        except ValueError:
            continue
        if time_s != cur_s:
            if cur_s is not None:
                prev_s = cur_s
            cur_s = time_s
        rows.append({
            "t_end_ns": t0_ns + round(time_s * 1e9),
            "interval_s": time_s - prev_s,
            "event": re.sub(r":[A-Za-z]+$", "", f[3]),
            "count": _parse_count(f[1]),
            "pct_running": _parse_pct(f[5]) if len(f) > 5 else None,
        })
    return rows


def wide_columns(events=DEFAULT_EVENTS) -> list[str]:
    """counters.csv header: t_start_ns, t_end_ns, ctr_* then pct_*, both in `events` order."""
    return (["t_start_ns", "t_end_ns"] + [f"ctr_{_col(e)}" for e in events]
            + [f"pct_{_col(e)}" for e in events])


def to_wide(rows: list[dict], events=DEFAULT_EVENTS) -> list[dict]:
    """One row per interval: t_start_ns, t_end_ns and ctr_<event>/pct_<event> per event.

    t_start_ns is the previous interval's end; for the first interval it is t0 (end minus
    interval_s). Events in `events` that perf never printed for an interval are None; rows for
    other events are ignored (perf only prints what was requested).
    """
    cols = wide_columns(events)
    by_end: dict[int, dict] = {}
    first_dt_ns: dict[int, int] = {}
    for r in rows:
        end = r["t_end_ns"]
        w = by_end.setdefault(end, dict.fromkeys(cols))
        first_dt_ns.setdefault(end, round(r["interval_s"] * 1e9))
        w["t_end_ns"] = end
        if f"ctr_{_col(r['event'])}" in w:
            w[f"ctr_{_col(r['event'])}"] = r["count"]
            w[f"pct_{_col(r['event'])}"] = r["pct_running"]
    wide, prev_end = [], None
    for end in sorted(by_end):
        w = by_end[end]
        w["t_start_ns"] = prev_end if prev_end is not None else end - first_dt_ns[end]
        wide.append(w)
        prev_end = end
    return wide


def write_counters_csv(path: str, wide: list[dict], events=DEFAULT_EVENTS) -> None:
    """Write counters.csv (header only when there are no rows). None becomes an empty cell."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_columns(events))
        w.writeheader()
        w.writerows(wide)


def read_paranoid(path: str = PARANOID_PATH) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def min_pct_running(rows: list[dict]) -> float | None:
    """Lowest pct_running over all hardware-event cells (None if there is no data).

    A `<not counted>` cell reports 0 %, so multiplexing and gaps show up here.
    """
    pcts = [r["pct_running"] for r in rows
            if r["event"] not in SOFTWARE_EVENTS and r["pct_running"] is not None]
    return min(pcts) if pcts else None


def first_line_time_s(text: str) -> float | None:
    """Time field (seconds) of the first COMPLETE (newline-terminated) interval line, else None."""
    complete = text[:text.rfind("\n") + 1]
    rows = parse_interval_csv(complete, 0)
    return rows[0]["t_end_ns"] / 1e9 if rows else None


def _read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def counters_meta(enabled: bool, available: bool, interval_ms: int, events, t0_ns: int | None,
                  error: str | None, paranoid: int | None, offset_ns: int | None = None,
                  offset_method: str | None = None,
                  min_pct: float | None = None) -> dict:
    """The run_meta.json `perf_counters` block."""
    scope = None if paranoid is None else ("user" if paranoid >= 2 else "user+kernel")
    return {"enabled": enabled, "available": available, "interval_ms": interval_ms,
            "events": list(events), "t0_ns": t0_ns, "offset_ns": offset_ns,
            "offset_method": offset_method, "error": error,
            "paranoid": paranoid, "scope": scope, "min_pct_running": min_pct}


class PerfStat:
    """One `perf stat -I` child attached to `pid`. Failures are recorded in `.error`, never raised."""

    def __init__(self, pid: int, events=DEFAULT_EVENTS, interval_ms: int = 100,
                 out_path: str = "counters.raw.txt", popen=subprocess.Popen,
                 clock=time.time_ns, which=shutil.which, sleep=time.sleep, read_text=_read_file):
        self.pid = pid
        self.events = tuple(events)
        self.interval_ms = interval_ms
        self.out_path = out_path
        self._popen = popen
        self._clock = clock
        self._which = which
        self._sleep = sleep
        self._read_text = read_text
        self.t0_ns: int | None = None         # lab clock just before perf was spawned
        self.offset_ns: int | None = None     # perf's time zero minus t0_ns (see module docstring)
        self.offset_method: str | None = None  # "first-line" | "none"
        self._min_pct: float | None = None
        self.error: str | None = None
        self._proc = None

    def available(self) -> bool:
        return self._which("perf") is not None

    def note_error(self, msg: str) -> None:
        """Append to `.error` (reported in run_meta); callers use it for failures around perf."""
        self.error = msg if self.error is None else f"{self.error}; {msg}"

    def start(self) -> bool:
        """Spawn perf; True if it was launched. t0_ns is taken immediately before the spawn."""
        if not self.available():
            self.note_error("perf not found on PATH")
            return False
        argv = ["perf", "stat", "-I", str(self.interval_ms), "-x", ",", "-o", self.out_path,
                "-e", ",".join(self.events), "-p", str(self.pid)]
        try:
            os.remove(self.out_path)   # a stale file would fake the first-line detection
        except OSError:
            pass
        self.t0_ns = self._clock()
        try:
            self._proc = self._popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True)
        except OSError as e:
            self.note_error(f"perf failed to start: {e}")
            return False
        self._measure_offset()
        return True

    def _measure_offset(self) -> None:
        """Poll the raw file (<= 2 s) for perf's first interval line to find perf's time zero."""
        self.offset_ns, self.offset_method = 0, "none"
        for _ in range(round(FIRST_LINE_GIVE_UP_S / FIRST_LINE_POLL_S)):
            first_s = first_line_time_s(self._read_text(self.out_path))
            if first_s is not None:
                seen_ns = self._clock()
                self.offset_ns = max(0, seen_ns - self.t0_ns - round(first_s * 1e9))
                self.offset_method = "first-line"
                return
            if self._proc.poll() is not None:   # perf died; stop() reports why
                return
            self._sleep(FIRST_LINE_POLL_S)

    def stop(self) -> None:
        """SIGINT perf (it flushes the last interval), wait <= 5 s, then kill. Idempotent."""
        p, self._proc = self._proc, None
        if p is None:
            return
        if p.poll() is not None:
            _, err = p.communicate()
            self.note_error(f"perf exited early (rc={p.returncode}): {(err or '').strip()[-300:]}")
            return
        p.send_signal(signal.SIGINT)
        try:
            _, err = p.communicate(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            self.note_error(f"perf did not exit within {STOP_TIMEOUT_S:g} s of SIGINT; killed")
            return
        if p.returncode not in _CLEAN_EXIT:
            self.note_error(f"perf exited with rc={p.returncode}: {(err or '').strip()[-300:]}")

    def wide_rows(self) -> list[dict]:
        """Parse the raw file (call after stop()). [] if perf never started or wrote nothing."""
        if self.t0_ns is None:
            return []
        try:
            with open(self.out_path) as f:
                text = f.read()
        except OSError as e:
            self.note_error(f"raw counters unreadable: {e}")
            return []
        rows = parse_interval_csv(text, self.t0_ns + self.offset_ns)
        self._min_pct = min_pct_running(rows)
        return to_wide(rows, self.events)

    def meta(self, enabled: bool, paranoid: int | None = None) -> dict:
        return counters_meta(enabled, self.available(), self.interval_ms, self.events,
                             self.t0_ns, self.error, paranoid, self.offset_ns,
                             self.offset_method, self._min_pct)
