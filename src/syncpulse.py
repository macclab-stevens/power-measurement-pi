"""Power-signature sync pulses for aligning Pi events with the Monsoon trace.

All cores busy-spin during ON slots and sleep during OFF slots. That gives a
~2-3 W step on the Pi's 5 V input that the Monsoon sees within a sample or two.
Each edge's Pi wall-clock time is logged immediately before the shared flag
flips. Workers poll one lock-free shared flag (0 off, 1 on, 2 quit), so falling
edges do not contend with the spinners and the rising-edge wake lag is at most
POLL_S. Distinct slot widths make the train unambiguous in cross-correlation.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

PATTERN_MS = ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))
GUARD_MS = 500
POLL_S = 0.0002


def _spin(flag):  # pragma: no cover - runs in child processes
    x = 0
    while True:
        s = flag.value
        if s == 1:
            x += 1
        elif s == 0:
            time.sleep(POLL_S)
        else:
            return


def emit(events, tag: str, pattern=PATTERN_MS, guard_ms: int = GUARD_MS,
         ncores: int | None = None) -> None:
    ctx = mp.get_context("fork")
    flag = ctx.RawValue("b", 0)  # 0 = off, 1 = on, 2 = quit
    workers = [ctx.Process(target=_spin, args=(flag,), daemon=True)
               for _ in range(ncores or os.cpu_count() or 4)]
    try:
        for w in workers:
            w.start()
        events.marker(f"sync:{tag}")
        time.sleep(guard_ms / 1000.0)
        for on, ms in pattern:
            if on:
                events.edge("pulse_on", tag)
                flag.value = 1
            else:
                events.edge("pulse_off", tag)
                flag.value = 0
            time.sleep(ms / 1000.0)
        events.edge("pulse_off", tag)
        flag.value = 0
        time.sleep(guard_ms / 1000.0)
    finally:
        flag.value = 2
        for w in workers:
            if w.pid is not None:
                w.join(timeout=2)
