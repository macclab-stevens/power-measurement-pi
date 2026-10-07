# host/tests/test_align.py
import csv
import json

import numpy as np
import pandas as pd
import pytest

from align import AlignError, align, detect_slips
from store import SampleStore

FIELDS = ["t_real_ns", "t_mono_ns", "kind", "name", "depth", "mode", "iter",
          "layer_path", "config_id", "window_id"]
PATTERN = ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))
T0 = 1_800_000_000_000_000_000           # recording start (true time of sample 0)
B = 200_000.0 * (1 + 30e-6)              # Monsoon oscillator 30 ppm fast
USB_MIN_MS = 2.5                         # constant latency the pulses must remove


def _train_edges(start_ns):
    t, out = start_ns, []
    for on, ms in PATTERN:
        out.append(("pulse_on" if on else "pulse_off", t))
        t += ms * 1_000_000
    out.append(("pulse_off", t))
    return out


def _build(tmp_path, with_pulses=True, extra_events=(), it="0", nan_in_on=False,
           train_tags=("start", "end"), slip=False, reset_at=None):
    rec, pi = tmp_path / "rec", tmp_path / "pi"
    pi.mkdir()
    n = int(20 * 5000)
    true_ns = T0 + B * np.arange(n)
    P = np.full(n, 3.2)
    trains = {k: v for k, v in {"start": T0 + 2_000_000_000,
                                "end": T0 + 15_000_000_000}.items() if k in train_tags}
    events = [("marker", "warmup", 0, "", T0 + 1_000_000_000)]
    for tag, s in trains.items():
        events.append(("marker", f"sync:{tag}", 0, "", s - 500_000_000))
        for kind, t in _train_edges(s):
            events.append((kind, tag, 0, "", t))
    events.append(("marker", "forward", 0, "", T0 + 5_000_000_000))
    L0, L1 = T0 + 6_000_000_000, T0 + 6_400_000_000          # layer "0" span
    C0, C1 = T0 + 6_100_000_000, T0 + 6_200_000_000          # nested "0.0" span
    events += [("begin", "layer", 1, "0", L0), ("begin", "layer", 2, "0.0", C0),
               ("end", "layer", 2, "0.0", C1), ("end", "layer", 1, "0", L1)]
    events.append(("marker", "done", 0, "", T0 + 17_000_000_000))
    events += list(extra_events)
    if with_pulses:
        for tag, s in trains.items():
            e = _train_edges(s)
            for (k0, t0), (_, t1) in zip(e, e[1:]):
                if k0 == "pulse_on":
                    P[(true_ns >= t0) & (true_ns < t1)] = 6.0
    if nan_in_on:
        a = int((2.35e9) / B)
        P[a:a + 15] = np.nan                                  # 3 ms dropout in 300 ms ON slot
    P += np.random.default_rng(1).normal(0, 0.05, n)
    if slip:                                  # 50 samples (10 ms) vanish mid-run, no NaN gap
        P = np.delete(P, np.s_[50_000:50_050])
        true_ns = np.delete(true_ns, np.s_[50_000:50_050])
        n = len(P)
    s = SampleStore(rec)
    blk, rng = 50, np.random.default_rng(2)
    raw = 0
    for i0 in range(0, n, blk):
        last = i0 + blk - 1
        host = true_ns[last] + USB_MIN_MS * 1e6 + rng.exponential(2e6)
        if reset_at is not None and i0 >= reset_at:
            raw = 14
        s.append(P[i0:i0 + blk] / 5.0, np.full(blk, 5.0), int(host), raw)
    s.close()
    events.sort(key=lambda e: e[4])
    with open(pi / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for kind, name, depth, path, t, *lab in events:
            w.writerow({"t_real_ns": int(t), "t_mono_ns": int(t), "kind": kind, "name": name,
                        "depth": depth, "mode": lab[0] if lab else "forward",
                        "iter": lab[1] if lab else it, "layer_path": path,
                        "config_id": "", "window_id": 0})
    return rec, pi, (L0, L1, C0, C1)


def test_align_recovers_offset_and_passes(tmp_path):
    rec, pi, _ = _build(tmp_path)
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["n_edges"] == 12
    assert res["slip_warning"] is False and res["slips"] == []
    assert res["drift_warning"] is False
    assert res["max_residual_ms"] < 0.5
    assert abs(res["drift_ppm"]) < 5
    assert abs(res["rise_minus_fall_ms"]) < 0.2
    assert res["raw_dropped_nonzero_blocks_in_run"] == 0
    assert res["envelope_max_dev_ms"] < 0.5
    assert 0 <= res["nan_fraction"] < 0.01
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["pass"] is True


def test_align_labels_innermost_layer(tmp_path):
    rec, pi, (L0, L1, C0, C1) = _build(tmp_path)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    assert at((C0 + C1) // 2)["layer_path"] == "0.0"
    assert at((C1 + L1) // 2)["layer_path"] == "0"
    gap = at(L1 + 200_000_000)
    assert gap["layer_path"] == "" and gap["phase"] == "forward"
    on = at(T0 + 2_000_000_000 + 400_000_000)          # inside the 300 ms ON slot
    assert on["phase"] == "sync:start" and on["P_W"] > 5.0


def test_align_raises_when_no_pulses(tmp_path):
    rec, pi, _ = _build(tmp_path, with_pulses=False)
    with pytest.raises(AlignError, match="sync pulses not found"):
        align(rec, pi, pi)
    assert not (pi / "power_trace.parquet").exists()


def test_mode_iter_blank_outside_mode_phases(tmp_path):
    rec, pi, _ = _build(tmp_path, it="3",
                        extra_events=[("marker", "bench:abc", 0, "", T0 + 8_000_000_000)])
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    r = at(T0 + 9_000_000_000)
    assert r["phase"] == "bench:abc" and r["mode"] == "" and r["iter"] == ""
    r = at(T0 + 5_500_000_000)
    assert r["phase"] == "forward" and r["mode"] == "forward" and r["iter"] == "3"


def test_align_tolerates_dropped_samples(tmp_path):
    rec, pi, _ = _build(tmp_path, nan_in_on=True)
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["max_residual_ms"] < 0.5


def test_single_train_path(tmp_path):
    rec, pi, _ = _build(tmp_path, train_tags=("start",))
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["drift_ppm"] == 0.0 and res["n_edges"] == 6


def test_align_flags_mid_run_index_slip(tmp_path):
    rec, pi, _ = _build(tmp_path, slip=True)
    res = align(rec, pi, pi)
    assert res["pass"] is True
    assert (pi / "power_trace.parquet").exists()
    assert res["slip_warning"] is True
    assert len(res["slips"]) >= 1
    assert abs(res["slip_samples_total"] - 50) <= 5
    assert any("nonlinear" in w for w in res["warnings"])
    assert res["envelope_max_dev_ms"] > 1.0
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["slip_warning"] is True


T_BASE = 1_800_000_000_000_000_000


def _stream(n_blocks=1000, blk=50, rate=5000.0, lat_ms=2.0, jit_ms=0.2, losses=(), plateau=None,
            seed=3):
    """losses: [(block, n_samples)] permanent index losses; host time follows true samples."""
    rng = np.random.default_rng(seed)
    true_idx = np.arange(n_blocks) * blk + (blk - 1)
    idx = true_idx.copy()
    for b_, n_ in losses:
        idx[b_:] -= n_
    lat = lat_ms + rng.uniform(0, jit_ms, n_blocks)
    if plateau:
        lat[plateau[0]:plateau[1]] += plateau[2]
    host = (T_BASE + true_idx * 1e9 / rate + lat * 1e6).astype(np.int64)
    return idx, host, true_idx


def test_detect_slips_pure():
    idx, host, true_idx = _stream(losses=[(400, 30)])
    slips = detect_slips(idx, host, 5000.0)
    assert len(slips) == 1
    assert abs(slips[0]["samples"] - 30) <= 3
    assert abs(slips[0]["t_ns"] - host[400]) <= 3 * 10_000_000
    assert detect_slips(true_idx, host, 5000.0) == []
    assert len(detect_slips(idx, host)) == 1          # rate estimated robustly


def test_detect_slips_no_false_positive_jitter():
    idx, host, _ = _stream(n_blocks=10_000, jit_ms=35.0)
    assert detect_slips(idx, host) == []


def test_detect_slips_ignores_temporary_latency_plateau():
    idx, host, _ = _stream(plateau=(300, 600, 5.0))
    assert detect_slips(idx, host) == []


def test_detect_slips_near_end():
    idx, host, _ = _stream(losses=[(970, 30)])
    slips = detect_slips(idx, host)
    assert len(slips) == 1 and abs(slips[0]["samples"] - 30) <= 3


def test_detect_slips_multiple():
    idx, host, _ = _stream(losses=[(400, 30), (420, 30)])
    slips = detect_slips(idx, host)
    assert len(slips) == 2
    for sl, blk in zip(slips, (400, 420)):
        assert abs(sl["samples"] - 30) <= 3
        assert abs(sl["t_ns"] - host[blk]) <= 3 * 10_000_000


def test_align_flags_mid_run_index_slip(tmp_path):
    rec, pi, _ = _build(tmp_path, slip=True)
    res = align(rec, pi, pi)
    assert res["pass"] is True
    assert (pi / "power_trace.parquet").exists()
    assert res["slip_warning"] is True
    assert len(res["slips"]) >= 1
    assert abs(res["slip_samples_total"] - 50) <= 5
    assert any("nonlinear" in w for w in res["warnings"])
    assert res["envelope_max_dev_ms"] > 1.0
    assert res["slip_rate_hz"] == pytest.approx(5000 / (1 + 30e-6), rel=1e-4)
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["slip_warning"] is True


def test_align_reports_raw_dropped_but_does_not_gate(tmp_path):
    rec, pi, _ = _build(tmp_path, reset_at=50_000)   # blocks after 50k carry raw=14
    res = align(rec, pi, pi)
    assert res["pass"] is True
    assert res["raw_dropped_nonzero_blocks_in_run"] > 0


def test_ctx_read_flag_and_labels_unaffected(tmp_path):
    c0, c1 = T0 + 5_200_000_000, T0 + 5_250_000_000
    rec, pi, _ = _build(tmp_path, it="3", extra_events=[
        ("ctx_begin", "ctx", 0, "", c0, "", "99"), ("ctx_end", "ctx", 0, "", c1, "", "99")])
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    inside = df[(df["t_pi_ns"] >= c0) & (df["t_pi_ns"] < c1)]
    assert len(inside) > 100 and inside["ctx_read"].all()
    assert df["ctx_read"].sum() == len(inside)
    assert df["ctx_read"].dtype == bool
    for t in (c0 + 25_000_000, c1 + 100_000_000):
        r = df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
        assert r["mode"] == "forward" and r["iter"] == "3" and r["phase"] == "forward"
