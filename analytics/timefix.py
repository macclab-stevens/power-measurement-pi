"""Sample-loss slip correction of the power-trace time axis (R2, audit section 3.1, B-1).

Older cells (collected before host/align.py learned to remove slip steps, alignment.json without
`"slip_corrected": true`) carry one straight line `t_pi = c0 + (1 + c1) * t_host` fitted through the
start and end sync-pulse trains. A sample-loss slip between the trains (K samples dropped by the
meter link) is a permanent step of `step_ms` in the index-vs-host relation; one line through both trains
turns that step into fake linear drift and shears every timestamp in between by up to ~step/2.

Step model (reference: the audit lever `slip_shear.py`, `corr_fn`): the true offset of the samples against
the line steps by +step at each slip (samples after a loss of K samples really occurred LATER), while the
fitted line is exact at the mean times of the first 6 and last 6 pulse events (`ta`, `tb`). The line
absorbs the part of each step that interpolates linearly between `ta` and `tb`, so the per-sample
correction to ADD to the stored time is

    corr(t) = sum_j step_j * (H(t - ts_j) - (Ha_j + (Hb_j - Ha_j) * (t - ta) / (tb - ta)))

with H the unit step, `Ha_j = H(ta - ts_j)`, `Hb_j = H(tb - ts_j)` and the slip time on the Pi clock
`ts_j = (slip.t_ns - c0_ns + c1 * ref_ns) / (1 + c1)`, the same host-to-Pi map align.py applies to the samples. A slip midway between the trains therefore gives -step/2 just before it
and +step/2 just after it. A slip before the start train (`ts < ta`) leaves the samples after the start
train untouched and moves the samples before the slip by -step (the line is exact at the pulses, so the
pulse-anchored part of the run keeps its stored times); a slip after the end train mirrors that.
"""
import numpy as np

PULSE_KINDS = ("pulse_on", "pulse_off")
MIN_PULSES = 12            # 6 per train; fewer and the fit cannot be reproduced
TRAIN_EDGES = 6


def needs_correction(alignment: dict | None) -> bool:
    """True for an alignment.json that records slips and was not already corrected by host/align.py."""
    if not alignment or alignment.get("slip_corrected"):
        return False
    return bool(alignment.get("slips"))


def slip_correction_ns(alignment: dict, pulse_times_ns, t_ns) -> np.ndarray | None:
    """Per-sample correction in ns to add to the sample times `t_ns` (Pi clock).

    `alignment` is the cell's alignment.json (`c0_ns`, `slips` with `t_ns` host time and `step_ms`);
    `pulse_times_ns` are the Pi-clock times of the sync pulse edges (events.csv kinds pulse_on / pulse_off).
    Returns zeros when the cell has no slips, and None when it has slips but fewer than 12 pulse
    events (the correction cannot be anchored; the caller counts and skips the cell)."""
    t = np.asarray(t_ns, dtype=np.float64)
    slips = alignment.get("slips") or []
    if not slips:
        return np.zeros_like(t)
    pul = np.sort(np.asarray(pulse_times_ns, dtype=np.float64))
    if len(pul) < MIN_PULSES:
        return None
    ta, tb = pul[:TRAIN_EDGES].mean(), pul[-TRAIN_EDGES:].mean()
    c0, c1 = float(alignment["c0_ns"]), float(alignment.get("c1", 0.0))
    ref = float(alignment.get("ref_ns", ta))
    x = t - ta                                    # relative times: float64 has ~256 ns resolution at 1.8e18
    out = np.zeros_like(t)
    for s in slips:
        # Pi time of the slip: (t_host - c0 + c1 ref) / (1 + c1) = (t_host - c0 - ref) / (1 + c1) + ref (align.py's map),
        # taken relative to ta with the large terms differenced first (float64 precision)
        ts = (float(s["t_ns"]) - c0 - ref) / (1.0 + c1) + (ref - ta)
        ha, hb = float(0.0 > ts), float(tb - ta > ts)     # H(ta - ts_abs) = [0 > ts_rel], H(tb - ts_abs)
        out += float(s["step_ms"]) * 1e6 * ((x > ts) - (ha + (hb - ha) * x / (tb - ta)))
    return out
