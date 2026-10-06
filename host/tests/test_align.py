# host/tests/test_align.py
import csv
import json

import numpy as np
import pandas as pd
import pytest

from align import AlignError, align
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
            raw = 40_000
        s.append(P[i0:i0 + blk] / 5.0, np.full(blk, 5.0), int(host), raw)
    s.close()
    events.sort(key=lambda e: e[4])
    with open(pi / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for kind, name, depth, path, t in events:
            w.writerow({"t_real_ns": int(t), "t_mono_ns": int(t), "kind": kind, "name": name,
                        "depth": depth, "mode": "forward", "iter": it, "layer_path": path,
                        "config_id": "", "window_id": 0})
    return rec, pi, (L0, L1, C0, C1)


def test_align_recovers_offset_and_passes(tmp_path):
    rec, pi, _ = _build(tmp_path)
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["n_edges"] == 12
    assert res["max_residual_ms"] < 0.5
    assert abs(res["drift_ppm"]) < 5
    assert abs(res["rise_minus_fall_ms"]) < 0.2
    assert res["counter_resets_in_run"] == 0
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


def test_align_rejects_mid_run_index_slip(tmp_path):
    rec, pi, _ = _build(tmp_path, slip=True)
    with pytest.raises(AlignError, match="nonlinear"):
        align(rec, pi, pi)
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["pass"] is False and saved["envelope_max_dev_ms"] > 1.0


def test_align_rejects_counter_reset_in_run(tmp_path):
    rec, pi, _ = _build(tmp_path, reset_at=50_000)
    with pytest.raises(AlignError, match="counter_reset"):
        align(rec, pi, pi)
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["pass"] is False and saved["counter_resets_in_run"] == 1
