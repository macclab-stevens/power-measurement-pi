"""Power-signature sync pulses for aligning Pi events with the Monsoon trace.

All cores busy-spin during ON slots and sleep during OFF slots. That gives a
~2-3 W step on the Pi's 5 V input that the Monsoon sees within a sample or two.
Each edge's Pi wall-clock time is logged right before the workers are
released/parked. Distinct slot widths make the train unambiguous in
cross-correlation.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

PATTERN_MS = ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))
GUARD_MS = 500


def _spin(go, quit_):  # pragma: no cover - runs in child processes
    x = 0
    while not quit_.is_set():
        if go.is_set():
            x += 1
        else:
            go.wait(0.05)


def emit(events, tag: str, pattern=PATTERN_MS, guard_ms: int = GUARD_MS,
         ncores: int | None = None) -> None:
    ctx = mp.get_context("fork")
    go, quit_ = ctx.Event(), ctx.Event()
    workers = [ctx.Process(target=_spin, args=(go, quit_), daemon=True)
               for _ in range(ncores or os.cpu_count() or 4)]
    try:
        for w in workers:
            w.start()
        events.marker(f"sync:{tag}")
        time.sleep(guard_ms / 1000.0)
        for on, ms in pattern:
            if on:
                events.edge("pulse_on", tag)
                go.set()
            else:
                events.edge("pulse_off", tag)
                go.clear()
            time.sleep(ms / 1000.0)
        events.edge("pulse_off", tag)
        go.clear()
        time.sleep(guard_ms / 1000.0)
    finally:
        quit_.set()
        go.set()  # wake parked workers so they see quit_
        for w in workers:
            if w.pid is not None:
                w.join(timeout=2)
