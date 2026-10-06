"""Map Monsoon sample index -> EliteBook wall-clock ns (PTP-disciplined).

A USB block reaches the host at least some minimum latency after its last sample.
So the true clock line is the *lower envelope* of (last_idx, host_ns): take the
minimum-latency block in each of `segments` equal index segments and fit a
line through those. The remaining constant (minimum USB latency) is removed
later by the sync pulses in align.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOMINAL_NS = 200_000.0  # 1 / 5 kHz


@dataclass
class SampleClock:
    a_ns: float
    b_ns: float

    def to_ns(self, idx):
        return self.a_ns + self.b_ns * np.asarray(idx, dtype=np.float64)

    def to_idx(self, t_ns):
        return (np.asarray(t_ns, dtype=np.float64) - self.a_ns) / self.b_ns


def fit_sample_clock(last_idx, host_ns, rate_hz: int = 5000, segments: int = 20) -> SampleClock:
    x = np.asarray(last_idx, dtype=np.float64)
    y = np.asarray(host_ns, dtype=np.float64)
    b0 = 1e9 / rate_hz
    if x.size < 2 * segments:
        k = int(np.argmin(y - b0 * x))
        return SampleClock(a_ns=y[k] - b0 * x[k], b_ns=b0)
    r = y - b0 * x
    edges = np.linspace(x.min(), x.max() + 1, segments + 1)
    px, py = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.any():
            k = np.flatnonzero(m)[np.argmin(r[m])]
            px.append(x[k])
            py.append(y[k])
    px, py = np.array(px), np.array(py)
    b, a = np.polyfit(px - px[0], py - py[0], 1)   # centered: keeps ns precision at ~1.8e18
    return SampleClock(a_ns=py[0] + a - b * px[0], b_ns=b)
