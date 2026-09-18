"""Power sampling from the Raspberry Pi 5 PMIC ADC via vcgencmd.

Fast loop reads VDD_CORE current + voltage (channels 7, 15) at ~200 Hz.
A separate context thread records the full rail dump + arm freq + temp +
throttle every ~5 s without ever gaping the fast loop.

Every raw sample carries a phase marker and window_id so it can be
re-sliced per operation after the fact. The start/stop/marker/window/
snapshot_env contract is the drop-in boundary for a future Monsoon backend.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque

VCMD = "/usr/bin/vcgencmd"
CONTEXT_PERIOD_S = 5.0
MAX_SAMPLES = 2_000_000


def _read_core() -> tuple[float, float]:
    """Return (VDD_CORE current A, VDD_CORE voltage V). Verify-hat: no root."""
    p = subprocess.run(
        [VCMD, "pmic_read_adc", "7", "15"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"vcgencmd pmic_read_adc rc={p.returncode}: {p.stderr}")
    ia = vb = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("VDD_CORE_A"):
            ia = float(line.split("=")[1].rstrip("A"))
        elif line.startswith("VDD_CORE_V"):
            vb = float(line.split("=")[1].rstrip("V"))
    if ia is None or vb is None:
        raise RuntimeError(f"parse failed on vcgencmd output:\n{p.stdout}")
    return ia, vb


def _full_dump() -> dict[str, float]:
    p = subprocess.run([VCMD, "pmic_read_adc"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"full dump rc={p.returncode}")
    out: dict[str, float] = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if "=" in line:
            name, _, val = line.partition("=")
            try:
                out[name] = float(val.rstrip("AV"))
            except ValueError:
                pass
    return out


def _measure_temp() -> float:
    p = subprocess.run([VCMD, "measure_temp"], capture_output=True, text=True)
    txt = p.stdout.strip() or p.stderr.strip()
    return float(txt.split("'")[0].split("=")[1])


def _measure_clock_arm() -> int:
    p = subprocess.run([VCMD, "measure_clock", "arm"], capture_output=True, text=True)
    return int(p.stdout.strip().split("=")[1])


def _get_throttled() -> int:
    p = subprocess.run([VCMD, "get_throttled"], capture_output=True, text=True)
    return int(p.stdout.strip().split("=")[1], 16)


class PowerSampler:
    """Sampling daemon. Started once, samples until stop()."""

    def __init__(self) -> None:
        self._rows: deque[dict] = deque(maxlen=MAX_SAMPLES)
        self._ctx: list[dict] = []
        self._phase = "idle"
        self._window_id = 0
        self._t0 = time.monotonic()
        self._fatal: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fast = threading.Thread(target=self._fast_loop, daemon=True, name="pmic-fast")
        self._ctx_thread = threading.Thread(target=self._ctx_loop, daemon=True, name="pmic-ctx")

    # ---- time helpers: everything is sampler-relative ms --------------------
    def _now(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def now_ms(self) -> float:
        return self._now()

    # ---- public control ------------------------------------------------------
    def start(self) -> None:
        self._t0 = time.monotonic()
        self._stop.clear()
        self._fast.start()
        self._ctx_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._fast.join(timeout=10)
        self._ctx_thread.join(timeout=10)
        if self._fatal:
            raise RuntimeError(f"power sampling failed: {self._fatal}")

    def marker(self, name: str | None) -> None:
        """Set the current phase label. Each new non-None active bench label bumps window_id."""
        with self._lock:
            if name and name != self._phase:
                self._window_id += 1
            self._phase = name or "idle"

    # ---- data access ----------------------------------------------------------
    def window(self, t0_ms: float, t1_ms: float) -> dict:
        """Mean of P across samples with t in [t0, t1). Poison rows excluded."""
        with self._lock:
            rows = list(self._rows)
        ps = [r["P_W"] for r in rows if t0_ms <= r["t_ms"] < t1_ms and r["P_W"] is not None]
        if not ps:
            return {"count": 0, "mean_P": None, "max_P": None}
        return {"count": len(ps), "mean_P": sum(ps) / len(ps), "max_P": max(ps)}

    def snapshot_env(self) -> tuple[float | None, int | None, int | None]:
        """(temp_C, arm_freq_MHz, throttled_bits) measured synchronously; used to bracket bench windows."""
        try:
            return _measure_temp(), _measure_clock_arm() // 1_000_000, _get_throttled()
        except Exception as e:  # noqa: BLE001 - env snapshot must never kill the bench
            return None, None, None

    @property
    def fatal(self) -> str | None:
        with self._lock:
            return self._fatal

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)

    def context(self) -> list[dict]:
        with self._lock:
            return list(self._ctx)

    def elapsed_seconds(self) -> float:
        return self._now() / 1000.0

    def achieved_hz(self) -> float:
        rows = self.rows()
        good = [r for r in rows if r["P_W"] is not None]
        if len(good) < 2 or self.elapsed_seconds() <= 0:
            return 0.0
        return len(good) / self.elapsed_seconds()

    # ---- internal loops --------------------------------------------------------
    def _fast_loop(self) -> None:  # pragma: no cover - thread body
        while not self._stop.is_set():
            try:
                ia, vb = _read_core()
                with self._lock:
                    self._rows.append({
                        "t_ms": self._now(), "phase": self._phase,
                        "window_id": self._window_id, "I_A": ia, "V_V": vb,
                        "P_W": ia * vb,
                    })
            except Exception as e:  # noqa: BLE001 - record poison, surface at stop()
                self._fatal = f"{type(e).__name__}: {e}"
                self._stop.set()
                return

    def _ctx_loop(self) -> None:  # pragma: no cover - thread body
        while not self._stop.is_set():
            try:
                row = {
                    "t_ms": self._now(), "temp_C": _measure_temp(),
                    "arm_MHz": _measure_clock_arm() // 1_000_000,
                    "throttle_bits": _get_throttled(), "dump": _full_dump(),
                }
            except Exception as e:  # noqa: BLE001
                row = {"t_ms": self._now(), "error": f"{type(e).__name__}: {e}"}
            with self._lock:
                self._ctx.append(row)
            self._stop.wait(CONTEXT_PERIOD_S)