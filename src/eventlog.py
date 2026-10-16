"""Pi-side event log: what the workload is doing, with nanosecond timestamps.

Primary clock is time.time_ns() (lab PTP/GPS-disciplined wall clock, ~0.03 ms
from the TM2500C), so events line up with the EliteBook's Monsoon recording.
monotonic_ns is kept as a secondary column. Appends are lock-guarded list
appends, cheap enough for per-layer begin/end events.
"""
from __future__ import annotations

import contextlib
import csv
import threading
import time

FIELDS = ["t_real_ns", "t_mono_ns", "kind", "name", "depth", "mode", "iter",
          "layer_path", "config_id", "window_id"]


class EventLog:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._depth = 0
        self.window_id = 0
        self.phase = "idle"
        self.labels: dict = {"mode": "", "iter": ""}

    def _add(self, kind: str, name: str, depth: int, layer_path: str = "",
             config_id: str = "") -> None:
        row = {"t_real_ns": time.time_ns(), "t_mono_ns": time.monotonic_ns(),
               "kind": kind, "name": name, "depth": depth,
               "mode": self.labels["mode"], "iter": self.labels["iter"],
               "layer_path": layer_path, "config_id": config_id,
               "window_id": self.window_id}
        with self._lock:
            self._rows.append(row)

    def marker(self, name: str) -> None:
        """Start a new top-level phase. A changed name bumps window_id (PowerSampler semantics)."""
        if name and name != self.phase:
            self.window_id += 1
        self.phase = name or "idle"
        cid = self.phase.split(":", 1)[1] if self.phase.startswith("bench:") else ""
        self._add("marker", self.phase, 0, config_id=cid)

    def begin(self, name: str, layer_path: str = "") -> None:
        self._depth += 1
        self._add("begin", name, self._depth, layer_path=layer_path)

    def end(self, name: str, layer_path: str = "") -> None:
        self._add("end", name, self._depth, layer_path=layer_path)
        self._depth = max(0, self._depth - 1)

    @contextlib.contextmanager
    def step(self, name: str):
        """Sub-phase span inside one iteration (preprocess, forward, backward, ...).

        Logged as step_begin/step_end; does not change the layer depth."""
        self._add("step_begin", name, 0)
        try:
            yield
        finally:
            self._add("step_end", name, 0)

    def edge(self, kind: str, name: str) -> None:
        self._add(kind, name, 0)

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)

    def write_csv(self, path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows())
