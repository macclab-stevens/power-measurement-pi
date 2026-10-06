"""Append-only on-disk store for the always-on Monsoon recording.

samples.f32 : interleaved float32 (I_A, V_V), one row per 5 kHz sample, global index = row.
blocks.csv  : one row per USB block: host arrival time_ns + cumulative device-dropped count.
Dropped samples are filled with NaN rows *before* the block that reports them, so
index-based time never slips.
"""
from __future__ import annotations

import collections
import csv
import os
import threading
from pathlib import Path

import numpy as np

HEADER = ["block_seq", "first_idx", "n", "host_ns", "dropped_cum", "gap_inserted"]
RECENT_BLOCKS = 20_000   # ~10 min of blocks kept in RAM for live clock fits


class SampleStore:
    def __init__(self, run_dir) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._samples_path = self.dir / "samples.f32"
        self._blocks_path = self.dir / "blocks.csv"
        new = not self._blocks_path.exists()
        self._f = open(self._samples_path, "ab")
        self._b = open(self._blocks_path, "a", buffering=1, newline="")
        self._w = csv.writer(self._b)
        if new:
            self._w.writerow(HEADER)
        self.lock = threading.Lock()
        self.n_total = os.path.getsize(self._samples_path) // 8
        self.dropped = 0
        self.seq = 0
        self._recent = collections.deque(maxlen=RECENT_BLOCKS)
        if not new:
            b = load_blocks(self.dir)
            if len(b["block_seq"]):
                self.seq = int(b["block_seq"][-1]) + 1
                self.dropped = int(b["dropped_cum"][-1])

    def append(self, I_A, V_V, host_ns: int, dropped_cum: int) -> None:
        I_A = np.asarray(I_A, dtype=np.float32)
        V_V = np.asarray(V_V, dtype=np.float32)
        with self.lock:
            gap = max(0, int(dropped_cum) - self.dropped)
            if gap:
                np.full((gap, 2), np.nan, dtype=np.float32).tofile(self._f)
                self.n_total += gap
                self.dropped = int(dropped_cum)
            first = self.n_total
            np.column_stack([I_A, V_V]).astype(np.float32).tofile(self._f)
            self._f.flush()
            self.n_total += len(I_A)
            self._w.writerow([self.seq, first, len(I_A), int(host_ns), self.dropped, gap])
            self._recent.append((first + len(I_A) - 1, int(host_ns)))
            self.seq += 1

    def read(self, i0: int, i1: int):
        with self.lock:
            n = self.n_total
        i0, i1 = max(0, i0), min(i1, n)
        if i1 <= i0:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        m = np.memmap(self._samples_path, dtype=np.float32, mode="r", shape=(n, 2))
        out = np.array(m[i0:i1])
        return out[:, 0], out[:, 1]

    def recent_blocks(self):
        with self.lock:
            arr = np.array(self._recent, dtype=np.float64).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    def close(self) -> None:
        self._f.close()
        self._b.close()


def load_blocks(run_dir) -> dict:
    p = Path(run_dir) / "blocks.csv"
    data = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    if data.size == 0:
        data = np.empty((0, len(HEADER)), dtype=np.int64)
    return {k: data[:, j] for j, k in enumerate(HEADER)}


def open_samples(run_dir):
    p = Path(run_dir) / "samples.f32"
    n = os.path.getsize(p) // 8
    return np.memmap(p, dtype=np.float32, mode="r", shape=(n, 2))
