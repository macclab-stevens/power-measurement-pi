"""Second workload for `run.py --concurrent`: a thread inside the measured process.

`--concurrent <model>.pt` runs a second ImageTask (tag "m1") next to the primary (tag "m0");
`--concurrent spinner` runs a fixed 1-thread matrix-multiply loop (tag "spin"). Both are plain
threads of the run.py process, so there is ONE process, ONE event log, ONE sync-pulse envelope
(the main thread emits it) and `perf stat -p <pid>` sees the worker's CPU time as well.

Why a thread and not a separate process for the spinner: perf attaches with -p and counts only
that pid's threads, a second process would be invisible to the counters; and a process would need
its own start/stop signalling. torch.mm releases the GIL, so the spinner does not slow the primary
through Python. (An un-pinned thread is used on purpose: pinning the spinner to a core would not
keep the primary's OpenMP threads off it.)

Lifecycle (driven by run.py):
  start()      thread created right after model load: it does its first call + warm-up passes
               (concurrently with the primary's), then idles on an Event. Every thread this
               worker will ever use (its own and its OpenMP team) therefore exists before perf
               attaches (perf -p counts only threads that exist at attach time).
  wait_warm()  main thread blocks until that warm-up is done (re-raises a worker failure).
  release()    called by the main thread after the primary's frame 0 (the torch.profiler frame
               runs alone, so op_table holds primary ops only): the worker loops until
  stop()       the primary's measured loop is finished; joins the thread.
  check()      raises RuntimeError in the main thread if the worker died.

Thread split (thread_split): torch's intra-op thread count is per calling thread (OpenMP ICV), so
the worker calls torch.set_num_threads itself and run.py sets the main thread's share.
"""
from __future__ import annotations

import threading
import time

import torch

SPINNER = "spinner"
PRIMARY_TAG = "m0"
SPIN_DIM = 256            # torch.mm of two 256x256 float32 matrices ...
SPIN_MM_PER_STEP = 32     # ... 32 times per "spin" step (tens of ms per step on the Pi 5)
SPIN_WARMUP_STEPS = 5
JOIN_TIMEOUT_S = 120.0


def worker_tag(concurrent: str) -> str:
    """Event tag of the second workload: 'spin' for the spinner, 'm1' for a second model."""
    return "spin" if concurrent == SPINNER else "m1"


def thread_split(concurrent: str, threads: int) -> dict:
    """torch threads per tag. Model + model: threads // 2 each (the remainder goes to m1).
    Model + spinner: the spinner takes one core, the primary keeps threads - 1. Never below 1
    (with --threads 1 the two workloads share a core)."""
    if concurrent == SPINNER:
        return {PRIMARY_TAG: max(1, threads - 1), "spin": 1}
    m0 = max(1, threads // 2)
    return {PRIMARY_TAG: m0, "m1": max(1, threads - m0)}


class SecondWorkload:
    """The concurrent worker thread (see module docstring). `step` is the tagged
    EventLog.step_for(tag) (or a no-op factory without an event log); `task` is a loaded
    ImageTask for a model worker and None for the spinner."""

    def __init__(self, tag: str, threads: int, step, task=None, warmup: int = 0,
                 clock=time.perf_counter) -> None:
        self.tag, self.threads, self.task, self.warmup = tag, threads, task, warmup
        self._step, self._clock = step, clock
        if task is not None:
            task.step = step
        self.iterations = 0
        self.lat_ms: list[float] = []
        self.error: BaseException | None = None
        self.first_call_ms: float | None = None
        self._warm = threading.Event()
        self._go = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name=f"concurrent-{tag}", daemon=True)

    # ---- one unit of work -------------------------------------------------------------
    def _spin_step(self) -> None:
        with self._step("spin"):
            for _ in range(SPIN_MM_PER_STEP):
                torch.mm(self._a, self._b, out=self._c)

    def _one(self) -> None:
        if self.task is not None:
            self.task.run_one("forward")
        else:
            self._spin_step()

    # ---- thread body ------------------------------------------------------------------
    def _main(self) -> None:
        try:
            torch.set_num_threads(self.threads)   # per-thread setting: the main thread has its own
            if self.task is None:
                g = torch.Generator().manual_seed(0)
                self._a = torch.randn(SPIN_DIM, SPIN_DIM, generator=g)
                self._b = torch.randn(SPIN_DIM, SPIN_DIM, generator=g)
                self._c = torch.empty(SPIN_DIM, SPIN_DIM)
                n_warm = SPIN_WARMUP_STEPS
            else:
                n_warm = self.warmup
            t0 = self._clock()
            self._one()                           # first call: lazy init, thread pools
            self.first_call_ms = round((self._clock() - t0) * 1000.0, 3)
            for _ in range(n_warm):
                self._one()
        except BaseException as e:                 # reported to the main thread by check()
            self.error = e
            self._warm.set()
            return
        self._warm.set()
        self._go.wait()
        try:
            while not self._stop.is_set():
                t = self._clock()
                self._one()
                self.lat_ms.append((self._clock() - t) * 1000.0)
                self.iterations += 1
        except BaseException as e:
            self.error = e

    # ---- control (main thread) ----------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def wait_warm(self, check=lambda: None, poll_s: float = 1.0) -> None:
        """Block until the worker finished its first call and warm-up, calling check() (the
        thermal watchdog) every poll_s; raise if the worker failed."""
        while not self._warm.wait(poll_s):
            check()
        self.check()

    def release(self) -> None:
        self._go.set()

    def stop(self) -> None:
        """Ask the worker to finish its current iteration and join it. Idempotent; safe to call
        before the thread was ever started or released."""
        self._stop.set()
        self._go.set()
        if self._thread.is_alive():
            self._thread.join(JOIN_TIMEOUT_S)

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"concurrent worker '{self.tag}' failed: {self.error!r}") \
                from self.error

    def alive(self) -> bool:
        return self._thread.is_alive()
