"""Pin the Pi 5 CPU clock and fan through the whitelisted root helper, then verify.

The helper (tools/pi-power-pin -> /usr/local/sbin/pi-power-pin) is the only thing
maccpi may run via passwordless sudo. A pinned clock removes DVFS voltage/frequency
variation from the power data. A full-speed fan turns its 5 V draw into a constant
offset instead of temperature-triggered power steps. The helper can also unpin (ondemand,
for DVFS cells) and set the fan to stock trips (auto) or off (backstop at 85 C).
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
FAN_MODES = ("full", "auto", "off", "none")
GOVERNORS = ("pinned", "ondemand")


def _helper(cmd: list[str], run) -> str:
    p = run(["sudo", "-n", HELPER, *cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"pi-power-pin {' '.join(cmd)} failed (rc={p.returncode}): "
                           f"{p.stderr.strip()} - is /etc/sudoers.d/pi-power-pin installed?")
    return p.stdout.strip()


def apply_mode(cpu_mhz: int | None, fan: str, governor: str, run=subprocess.run) -> str:
    """Set fan and clock policy through the helper; returns the last status line ('' if no-op).

    fan: "full" (max fan), "auto" (stock trips), "off" (fan only as a backstop at 85 C),
         "none" (leave the fan alone).
    governor: "pinned" (pin cpu_mhz; no clock command if cpu_mhz is None) or
              "ondemand" (unpin: ondemand 1500-2400 MHz, fan untouched; cpu_mhz is ignored).
    The fan command is issued before the clock command.
    """
    if cpu_mhz is not None and cpu_mhz not in ALLOWED_MHZ:
        raise ValueError(f"--cpu-mhz {cpu_mhz} not in {ALLOWED_MHZ[0]}..{ALLOWED_MHZ[-1]} step 100")
    if fan not in FAN_MODES:
        raise ValueError(f"fan {fan!r} not in {FAN_MODES}")
    if governor not in GOVERNORS:
        raise ValueError(f"governor {governor!r} not in {GOVERNORS}")
    cmds = [] if fan == "none" else [["fan", fan]]
    if governor == "ondemand":
        cmds.append(["unpin"])
    elif cpu_mhz is not None:
        cmds.append(["pin", str(cpu_mhz)])
    status = ""
    for c in cmds:
        status = _helper(c, run)
    return status


def apply(cpu_mhz: int | None, fan_full: bool, run=subprocess.run) -> str:
    return apply_mode(cpu_mhz, "full" if fan_full else "none", "pinned", run=run)


def pin(mhz: int, run=subprocess.run) -> str:
    """Lock all cores at mhz (helper `pin`); the fan is untouched."""
    if mhz is None:
        raise ValueError("pin() needs a clock in MHz")
    return apply_mode(mhz, "none", "pinned", run=run)


def unpin(run=subprocess.run) -> str:
    """Back to ondemand 1500-2400 MHz (helper `unpin`); the fan is untouched."""
    return apply_mode(None, "none", "ondemand", run=run)


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


def clock_range_errors(sysfs_rows: list[dict], lo_mhz: int = 1500, hi_mhz: int = 2400,
                       tol_mhz: int = 25) -> list[str]:
    """One error per sysfs row with any cpuN_kHz outside [lo-tol, hi+tol] MHz (unpinned runs)."""
    errs = []
    seen = False
    lo_khz, hi_khz = (lo_mhz - tol_mhz) * 1000, (hi_mhz + tol_mhz) * 1000
    for r in sysfs_rows:
        vals = {k: v for k, v in r.items()
                if k.startswith("cpu") and k.endswith("_kHz") and v is not None}
        if not vals:
            continue
        seen = True
        bad = {k: v for k, v in vals.items() if not lo_khz <= v <= hi_khz}
        if bad:
            what = ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
            errs.append(f"cpu clock outside {lo_mhz}-{hi_mhz} MHz (+/-{tol_mhz}): {what} "
                        f"at t_real_ns={r.get('t_real_ns')}")
    if not seen:
        errs.append("no cpu frequency readings")
    return errs
