"""Pin the Pi 5 CPU clock and fan through the whitelisted root helper, then verify.

The helper (tools/pi-power-pin -> /usr/local/sbin/pi-power-pin) is the only thing
maccpi may run via passwordless sudo. A pinned clock removes DVFS voltage/frequency
variation from the power data. A full-speed fan turns its 5 V draw into a constant
offset instead of temperature-triggered power steps.
"""
from __future__ import annotations

import glob
import subprocess
import time

HELPER = "/usr/local/sbin/pi-power-pin"
ALLOWED_MHZ = tuple(range(1500, 2401, 100))
FAN = "/sys/class/thermal/cooling_device0"
CURRENT_THROTTLE_MASK = 0xF   # under-voltage / capped / throttled / soft-limit *now*
STICKY_UNDERVOLT = 0x10000    # under-voltage has occurred since boot


def apply(cpu_mhz: int | None, fan_full: bool, run=subprocess.run) -> str:
    if cpu_mhz is not None and cpu_mhz not in ALLOWED_MHZ:
        raise ValueError(f"--cpu-mhz {cpu_mhz} not in {ALLOWED_MHZ[0]}..{ALLOWED_MHZ[-1]} step 100")
    cmds = ([["fan", "full"]] if fan_full else []) + \
           ([["pin", str(cpu_mhz)]] if cpu_mhz is not None else [])
    status = ""
    for c in cmds:
        p = run(["sudo", "-n", HELPER, *c], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"pi-power-pin {' '.join(c)} failed (rc={p.returncode}): "
                               f"{p.stderr.strip()} - is /etc/sudoers.d/pi-power-pin installed?")
        status = p.stdout.strip()
    return status


def status(run=subprocess.run) -> str:
    """Helper `status` output, or 'unavailable: <reason>'. Never raises."""
    try:
        p = run(["sudo", "-n", HELPER, "status"], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {e}"
    if p.returncode != 0:
        return f"unavailable: {p.stderr.strip()}"
    return p.stdout.strip()


def _read_fan() -> tuple[int, int]:
    with open(f"{FAN}/type") as t:
        kind = t.read().strip()
    if kind != "pwm-fan":
        raise RuntimeError(f"{FAN}/type is '{kind}', expected 'pwm-fan'")
    with open(f"{FAN}/cur_state") as a, open(f"{FAN}/max_state") as b:
        return int(a.read()), int(b.read())


def wait_fan_full(read=_read_fan, timeout_s: float = 20.0, sleep=time.sleep) -> tuple[int, int]:
    """step_wise raises the fan one state per thermal poll; wait until it reaches max."""
    deadline = time.monotonic() + timeout_s
    while True:
        cur, mx = read()
        if cur >= mx:
            return cur, mx
        if time.monotonic() >= deadline:
            raise RuntimeError(f"fan stuck at state {cur}/{mx} after {timeout_s}s")
        sleep(0.5)


def throttle_errors(ctx_rows: list[dict]) -> list[str]:
    """One error per context row whose *current* throttle bits (under-voltage etc.) are set."""
    errs = []
    for r in ctx_rows:
        thr = r.get("throttle_bits")
        if thr is not None and thr & CURRENT_THROTTLE_MASK:
            errs.append(f"throttle bits 0x{thr:x} at t={r.get('t_ms', 0):.0f} ms")
    return errs


def new_undervoltage(thr0, thr1) -> bool:
    """True if the sticky under-voltage bit was newly set between two get_throttled reads."""
    if thr0 is None or thr1 is None:
        return False
    return bool((thr1 & ~thr0) & STICKY_UNDERVOLT)


def clock_errors(ctx_rows: list[dict], cpu_mhz: int, tol_mhz: int = 25) -> list[str]:
    errs = []
    seen = False
    for r in ctx_rows:
        f = r.get("arm_MHz")
        if f is None:
            continue
        seen = True
        if abs(f - cpu_mhz) > tol_mhz:
            errs.append(f"arm clock {f} MHz != pinned {cpu_mhz} MHz at t={r.get('t_ms', 0):.0f} ms")
    errs += throttle_errors(ctx_rows)
    if not seen:
        errs.append("no arm clock readings in context")
    return errs
