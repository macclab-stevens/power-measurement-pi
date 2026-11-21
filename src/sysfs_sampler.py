"""1 Hz sysfs sampler (per-core CPU clock, temperature, fan state) with a thermal watchdog.

Reads only sysfs files, no subprocesses (a sysfs read is far cheaper than a vcgencmd call):
  {root}/sys/devices/system/cpu/cpu{0..3}/cpufreq/scaling_cur_freq   kHz
  {root}/sys/class/thermal/thermal_zone0/temp                        milli-C
  {root}/sys/class/thermal/cooling_device0/cur_state                 fan state (pwm-fan)
  {root}/sys/class/hwmon/hwmon*/in0_lcrit_alarm                      uv_alarm (0/1), the hwmon
      whose `name` is "rpi_volt" (found by name, resolved once; the index is not stable). Unlike
      vcgencmd get_throttled it is LIVE: it is 1 only while the supply is under-voltage, so a
      sticky "happened since boot" bit cannot hide a new event. None when there is no rpi_volt.
Every row is stamped with `time.time_ns()` (the lab wall clock), so sysfs.csv joins to
power_trace.parquet on t_pi_ns with pandas.merge_asof. An unreadable value is None.

Watchdog: the first reading >= abort_c sets `thermal_abort` (level "abort"); the run loop polls
it and raises ThermalAbort (run.py exits 3). A reading >= hard_stop_c upgrades it to level
"hard_stop" and calls `interrupt()` once (KeyboardInterrupt in the main thread, run.py exits 4).

Usage:
  s = SysfsSampler(); s.start(); ...; s.stop(); s.write_csv("sysfs.csv"); s.max_temp_C()
Tests drive `read_once()` against a fake tree (`root=`) with a fake `interrupt` and `clock`.
"""
from __future__ import annotations

import _thread
import csv
import glob
import os
import threading
import time

THERMAL_ABORT_C = 85.0
THERMAL_HARD_STOP_C = 90.0
N_CPUS = 4
ROW_KEYS = ["t_real_ns"] + [f"cpu{i}_kHz" for i in range(N_CPUS)] + ["temp_C", "fan_state",
                                                "uv_alarm"]

EXIT_THERMAL_ABORT = 3
EXIT_THERMAL_HARD_STOP = 4


class ThermalAbort(Exception):
    """Raised by the run loop when the sysfs watchdog recorded a thermal abort."""

    def __init__(self, details: dict):
        super().__init__(f"thermal {details.get('level')}: {details.get('temp_C')} C "
                         f"at t_real_ns={details.get('t_real_ns')}")
        self.details = details


def thermal_exit_code(thermal_abort: dict | None) -> int | None:
    """Process exit code for a watchdog record: 3 abort, 4 hard stop, None if none."""
    if not thermal_abort:
        return None
    return EXIT_THERMAL_HARD_STOP if thermal_abort.get("level") == "hard_stop" \
        else EXIT_THERMAL_ABORT


def uv_alarm_errors(rows: list[dict]) -> list[str]:
    """Acceptance error when any 1 Hz row saw the live under-voltage alarm."""
    n = sum(1 for r in rows if r.get("uv_alarm") == 1)
    return [f"live under-voltage alarm during run ({n} samples)"] if n else []


def _read_text(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _find_uv_alarm_path(root: str) -> str | None:
    """`in0_lcrit_alarm` of the hwmon device named rpi_volt (the Pi's under-voltage detector), or
    None. Matched by name: the hwmon index depends on probe order."""
    pattern = os.path.join(root, "sys/class/hwmon/hwmon*/name")
    for name_path in sorted(glob.glob(pattern)):
        if _read_text(name_path) == "rpi_volt":
            return os.path.join(os.path.dirname(name_path), "in0_lcrit_alarm")
    return None


def _read_int(path: str) -> int | None:
    txt = _read_text(path)
    try:
        return int(txt) if txt else None
    except ValueError:
        return None


class SysfsSampler:
    def __init__(self, period_s: float = 1.0, root: str = "/",
                 abort_c: float = THERMAL_ABORT_C, hard_stop_c: float = THERMAL_HARD_STOP_C,
                 clock=time.time_ns, interrupt=_thread.interrupt_main):
        self.period_s = period_s
        self.root = root
        self.abort_c = abort_c
        self.hard_stop_c = hard_stop_c
        self._clock = clock
        self._interrupt = interrupt
        self.thermal_abort: dict | None = None
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interrupted = False
        self._uv_path: str | None = None
        self._uv_resolved = False

    def _path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def read_once(self) -> dict:
        """Take one reading, store it, run the thermal check, and return the row."""
        row = {"t_real_ns": self._clock()}
        for i in range(N_CPUS):
            row[f"cpu{i}_kHz"] = _read_int(
                self._path(f"sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"))
        milli_c = _read_int(self._path("sys/class/thermal/thermal_zone0/temp"))
        row["temp_C"] = milli_c / 1000.0 if milli_c is not None else None
        row["fan_state"] = _read_int(self._path("sys/class/thermal/cooling_device0/cur_state"))
        row["uv_alarm"] = self._read_uv_alarm()
        with self._lock:
            self._rows.append(row)
            self._check(row)
        return row

    def _read_uv_alarm(self) -> int | None:
        if not self._uv_resolved:       # resolved once, on the first reading (sysfs.start())
            self._uv_path = _find_uv_alarm_path(self.root)
            self._uv_resolved = True
        return _read_int(self._uv_path) if self._uv_path else None

    def _check(self, row: dict) -> None:   # caller holds the lock
        t = row["temp_C"]
        if t is None:
            return
        if t >= self.hard_stop_c:
            if not self._interrupted:
                self.thermal_abort = {"t_real_ns": row["t_real_ns"], "temp_C": t,
                                      "level": "hard_stop"}
                self._interrupted = True
                self._interrupt()
        elif t >= self.abort_c and self.thermal_abort is None:
            self.thermal_abort = {"t_real_ns": row["t_real_ns"], "temp_C": t, "level": "abort"}

    def _loop(self) -> None:  # pragma: no cover - thread body (timing only; read_once is tested)
        while not self._stop.is_set():
            try:
                self.read_once()
            except Exception:   # the watchdog thread must never die
                pass
            self._stop.wait(self.period_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sysfs-1hz")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)

    def max_temp_C(self) -> float | None:
        temps = [r["temp_C"] for r in self.rows() if r["temp_C"] is not None]
        return max(temps) if temps else None

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows())
