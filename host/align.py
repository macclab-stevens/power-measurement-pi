#!/usr/bin/env python3
"""Align a Pi run's events.csv with the always-on Monsoon recording and label every sample.

1. Coarse clock: lower-envelope fit of block arrival stamps around the run (clockfit).
2. Sync pulses: detect power edges near each logged Pi edge, then fit
   offset(t) = c0 + c1 * (t - ref) over all edges (USB latency + oscillator drift).
3. Label: each sample gets the latest marker (phase), the latest event's mode/iter,
   and the innermost enclosing layer span.

    uv run python align.py --recorder-dir data/<session> --pi-run ../runs/<cell>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clockfit import fit_sample_clock
from store import load_blocks, open_samples

PASS_MS = 1.0
SEARCH_MS = 60.0          # edge search half-window around the coarse prediction
SMOOTH = 5                # centered moving average (1 ms), symmetric -> no edge shift
PAD_S = 5.0
ENVELOPE_MAX_DEV_MS = 1.0  # coarse-clock lower envelope must be linear to this (index-slip check)
MAX_DRIFT_PPM = 50.0
ENVELOPE_SEGMENTS = 20     # same rule as fit_sample_clock
MIN_ENVELOPE_BLOCKS = 40
LABEL_KINDS = ("marker", "begin", "end", "pulse_on", "pulse_off")


class AlignError(RuntimeError):
    pass


def _smooth(p):
    k = np.ones(SMOOTH) / SMOOTH
    p = np.asarray(p, dtype=np.float64)
    bad = ~np.isfinite(p)
    if bad.any():
        good = np.flatnonzero(~bad)
        if good.size == 0:
            raise AlignError("sync pulses not found (no finite power samples)")
        p = p.copy()
        p[bad] = np.interp(np.flatnonzero(bad), good, p[good])
    return np.convolve(p, k, mode="same")


def _find_edge(t_ns, p, t_guess, rising, lo, hi):
    m = (t_ns >= t_guess - SEARCH_MS * 1e6) & (t_ns <= t_guess + SEARCH_MS * 1e6)
    idx = np.flatnonzero(m)
    if idx.size < 3:
        return None
    th = (lo + hi) / 2.0
    seg = p[idx]
    above = seg > th
    flips = np.flatnonzero(above[1:] != above[:-1])
    flips = [f for f in flips if above[f + 1] == rising]
    if not flips:
        return None
    f = min(flips, key=lambda f: abs(t_ns[idx[f]] - t_guess))
    i, j = idx[f], idx[f + 1]
    frac = (th - p[i]) / (p[j] - p[i]) if p[j] != p[i] else 0.0
    return t_ns[i] + frac * (t_ns[j] - t_ns[i])


def _edges(ev):
    e = ev[ev["kind"].isin(["pulse_on", "pulse_off"])]
    return list(e.groupby("name", sort=False))


def _envelope_max_dev_ms(x, y, clock):
    """Max deviation of the per-segment minimum coarse-clock residual from its median (ms)."""
    if x.size < MIN_ENVELOPE_BLOCKS:
        return None
    r = y - clock.to_ns(x)
    edges = np.linspace(x.min(), x.max() + 1, ENVELOPE_SEGMENTS + 1)
    mins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.any():
            mins.append(r[m].min())
    mins = np.array(mins)
    return float(np.abs(mins - np.median(mins)).max() / 1e6)


def align(recorder_dir, pi_run_dir, out_dir) -> dict:
    recorder_dir, pi_run_dir, out_dir = Path(recorder_dir), Path(pi_run_dir), Path(out_dir)
    ev = pd.read_csv(pi_run_dir / "events.csv", keep_default_na=False,
                     dtype={"kind": str, "name": str, "mode": str, "iter": str,
                            "layer_path": str, "config_id": str})
    ev = ev.sort_values("t_real_ns", kind="stable").reset_index(drop=True)
    t_first, t_last = int(ev["t_real_ns"].iloc[0]), int(ev["t_real_ns"].iloc[-1])

    b = load_blocks(recorder_dir)
    last_idx = b["first_idx"] + b["n"] - 1
    sel = (b["host_ns"] >= t_first - PAD_S * 1e9) & (b["host_ns"] <= t_last + PAD_S * 1e9)
    if sel.sum() < 2:
        raise AlignError("recording does not cover this run's time range")
    clock = fit_sample_clock(last_idx[sel], b["host_ns"][sel])

    samples = open_samples(recorder_dir)
    i0 = max(0, int(np.floor(clock.to_idx(t_first - 1e9))))
    i1 = min(len(samples), int(np.ceil(clock.to_idx(t_last + 1e9))) + 1)
    IV = np.array(samples[i0:i1], dtype=np.float64)
    t_m = clock.to_ns(np.arange(i0, i1))
    P = IV[:, 0] * IV[:, 1]
    ps = _smooth(P)

    pi_t, mon_t, rising_ = [], [], []
    trains = []
    for name, grp in _edges(ev):
        ts = grp["t_real_ns"].to_numpy(dtype=np.float64)
        win = (t_m >= ts[0] - 0.45e9) & (t_m < ts[0] - 0.05e9)
        busy = (t_m >= ts[0]) & (t_m < ts[-1])
        if not win.any() or not busy.any():
            raise AlignError(f"sync pulses not found for train '{name}' (no samples)")
        lo, hi = np.median(ps[win]), np.percentile(ps[busy], 90)
        if hi - lo < 0.5:
            raise AlignError(f"sync pulses not found for train '{name}' "
                             f"(step {hi - lo:.2f} W < 0.5 W)")
        # guess = Pi time + offset estimated from the first rising edge
        first = _find_edge(t_m, ps, ts[0], True, lo, hi)
        if first is None:
            raise AlignError(f"sync pulses not found for train '{name}' (no first edge)")
        off = first - ts[0]
        found = 0
        for kind, t in zip(grp["kind"], ts):
            got = _find_edge(t_m, ps, t + off, kind == "pulse_on", lo, hi)
            if got is not None:
                pi_t.append(t)
                mon_t.append(got)
                rising_.append(kind == "pulse_on")
                found += 1
        trains.append({"name": name, "edges_found": found, "edges_logged": len(ts)})
        if found < len(ts) - 1:
            raise AlignError(f"sync pulses not found for train '{name}' "
                             f"({found}/{len(ts)} edges)")

    pi_t, mon_t, rising_ = np.array(pi_t), np.array(mon_t), np.array(rising_, dtype=bool)
    ref = pi_t[0]
    offs = mon_t - pi_t
    if len(trains) >= 2:
        c1, c0 = np.polyfit(pi_t - ref, offs, 1)
    else:
        c1, c0 = 0.0, float(np.median(offs))
    resid_ms = (offs - (c0 + c1 * (pi_t - ref))) / 1e6
    rise_bias = float(resid_ms[rising_].mean()) if rising_.any() else 0.0
    fall_bias = float(resid_ms[~rising_].mean()) if (~rising_).any() else 0.0
    res = {
        "rise_bias_ms": rise_bias, "fall_bias_ms": fall_bias,
        "rise_minus_fall_ms": rise_bias - fall_bias,
        "lag_note": "c0 includes USB latency + any common Pi-timestamp-to-power lag; "
                    "not separable from pulses alone",
        "c0_ns": float(c0), "c1": float(c1), "drift_ppm": float(c1 * 1e6), "ref_ns": float(ref),
        "max_residual_ms": float(np.abs(resid_ms).max()),
        "rms_residual_ms": float(np.sqrt(np.mean(resid_ms ** 2))),
        "n_edges": int(len(pi_t)), "trains": trains,
        "coarse_clock": {"a_ns": clock.a_ns, "b_ns": clock.b_ns},
    }
    fail_reasons = []
    resets = int(b["counter_reset"][sel].sum())
    res["counter_resets_in_run"] = resets
    if resets > 0:
        fail_reasons.append("counter_reset in run")
    env = _envelope_max_dev_ms(last_idx[sel].astype(np.float64),
                               b["host_ns"][sel].astype(np.float64), clock)
    res["envelope_max_dev_ms"] = env
    if env is not None and env > ENVELOPE_MAX_DEV_MS:
        fail_reasons.append("coarse clock envelope nonlinear (sample-index slip?)")
    if len(trains) >= 2 and abs(res["drift_ppm"]) > MAX_DRIFT_PPM:
        fail_reasons.append("drift > 50 ppm")
    if res["max_residual_ms"] >= PASS_MS:
        fail_reasons.append("residual >= 1.0 ms")
    res["nan_fraction"] = float(np.isnan(IV).any(axis=1).mean()) if len(IV) else 0.0
    hs = b["host_ns"][sel]
    res["max_block_gap_ms"] = float(np.diff(hs).max() / 1e6)
    res["fail_reasons"] = fail_reasons
    res["pass"] = bool(res["max_residual_ms"] < PASS_MS and not fail_reasons)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alignment.json").write_text(json.dumps(res, indent=2))
    if not res["pass"]:
        raise AlignError(f"alignment failed: {'; '.join(fail_reasons)} "
                         f"(max residual {res['max_residual_ms']:.3f} ms)")

    # ---- label samples on the Pi clock ------------------------------------------
    t_pi = (t_m - c0 + c1 * ref) / (1.0 + c1)
    keep = (t_pi >= t_first) & (t_pi <= t_last)
    t_pi, IV, P = t_pi[keep], IV[keep], P[keep]

    # mode/iter come from the latest *workload* event; ctx_begin/ctx_end are written by the
    # sampler's context thread, so their labels are arbitrary
    evw = ev[ev["kind"].isin(LABEL_KINDS)]
    et = evw["t_real_ns"].to_numpy(dtype=np.float64)
    k = np.clip(np.searchsorted(et, t_pi, side="right") - 1, 0, None)
    mode = evw["mode"].astype(str).to_numpy()[k].astype(object)
    it = evw["iter"].astype(str).to_numpy()[k].astype(object)
    mk = ev[ev["kind"] == "marker"]
    mt = mk["t_real_ns"].to_numpy(dtype=np.float64)
    km = np.searchsorted(mt, t_pi, side="right") - 1
    phase = np.where(km >= 0, mk["name"].to_numpy()[np.clip(km, 0, None)], "idle")
    # labels persist on the Pi between phases; only a mode's own phase marker (== mode name)
    # makes mode/iter meaningful
    outside = phase != mode
    mode[outside] = ""
    it[outside] = ""

    layer = np.full(t_pi.size, "", dtype=object)
    depth = np.zeros(t_pi.size, dtype=np.int16)
    spans = _layer_spans(ev)
    for d in sorted(spans):
        s = spans[d]
        st, en, paths = s[:, 0].astype(np.float64), s[:, 1].astype(np.float64), s[:, 2]
        j = np.searchsorted(st, t_pi, side="right") - 1
        ok = (j >= 0) & (t_pi < en[np.clip(j, 0, None)])
        layer[ok] = paths[j[ok]]
        depth[ok] = d

    cb = np.sort(ev[ev["kind"] == "ctx_begin"]["t_real_ns"].to_numpy(dtype=np.float64))
    ce = np.sort(ev[ev["kind"] == "ctx_end"]["t_real_ns"].to_numpy(dtype=np.float64))
    n_ctx = min(len(cb), len(ce))
    cb, ce = cb[:n_ctx], ce[:n_ctx]
    ctx_read = np.zeros(t_pi.size, dtype=bool)
    if n_ctx:
        j = np.searchsorted(cb, t_pi, side="right") - 1
        ctx_read = (j >= 0) & (t_pi < ce[np.clip(j, 0, None)])

    df = pd.DataFrame({
        "t_s": (t_pi - t_first) / 1e9, "t_pi_ns": t_pi.astype(np.int64),
        "I_A": IV[:, 0].astype(np.float32), "V_V": IV[:, 1].astype(np.float32),
        "P_W": P.astype(np.float32), "phase": phase, "mode": mode, "iter": it,
        "layer_path": layer, "depth": depth, "ctx_read": ctx_read,
    })
    df.to_parquet(out_dir / "power_trace.parquet", index=False)
    return res


def _layer_spans(ev) -> dict:
    """depth -> array[(start_ns, end_ns, path)] sorted by start (same-depth spans don't overlap)."""
    lay = ev[ev["kind"].isin(["begin", "end"])]
    open_: dict = {}
    out: dict = {}
    for kind, path, d, t in zip(lay["kind"], lay["layer_path"], lay["depth"], lay["t_real_ns"]):
        key = (path, int(d))
        if kind == "begin":
            open_.setdefault(key, []).append(t)
        elif open_.get(key):
            out.setdefault(int(d), []).append((open_[key].pop(), t, path))
    return {d: np.array(sorted(v, key=lambda r: r[0]), dtype=object) for d, v in out.items()}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recorder-dir", required=True)
    ap.add_argument("--pi-run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        res = align(a.recorder_dir, a.pi_run, a.out or a.pi_run)
    except AlignError as e:
        raise SystemExit(f"[FAIL] {e}")
    print(f"[ok] edges={res['n_edges']} max_residual={res['max_residual_ms']:.3f} ms "
          f"rms={res['rms_residual_ms']:.3f} ms drift={res['drift_ppm']:.2f} ppm "
          f"usb_offset={res['c0_ns'] / 1e6:.3f} ms")


if __name__ == "__main__":
    main()
