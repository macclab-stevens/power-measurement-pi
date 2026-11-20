# host/tests/test_align.py
import csv
import json

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from align import (AlignError, _robust_rate_hz, align, attach_pi_logs, detect_slips,
                   merge_counters, merge_sysfs)
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
        tagged = any(len(e) > 7 for e in events)      # v1-shaped files have no tag column
        w = csv.DictWriter(f, fieldnames=FIELDS + (["tag"] if tagged else []))
        w.writeheader()
        for kind, name, depth, path, t, *lab in events:
            w.writerow({"t_real_ns": int(t), "t_mono_ns": int(t), "kind": kind, "name": name,
                        "depth": depth, "mode": lab[0] if lab else "forward",
                        "iter": lab[1] if lab else it, "layer_path": path,
                        "config_id": "", "window_id": 0, **({"tag": lab[2] if len(lab) > 2 else ""} if tagged else {})})
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


def test_detect_slips_real_run():
    """m4_smoke_yolo11n (237 s, EliteBook): ~113 samples really lost (sync-pulse ground truth)."""
    import datetime as dt
    d = np.loadtxt(Path(__file__).parent / "fixtures" / "real_slips_blocks.csv",
                   delimiter=",", skiprows=1, dtype=np.int64)
    last_idx, host = d[:, 0] + d[:, 1] - 1, d[:, 2]
    assert abs(_robust_rate_hz(last_idx, host) - 4992.33) <= 0.15
    slips = detect_slips(last_idx, host)
    assert 90 <= sum(s["samples"] for s in slips) <= 125
    assert len(slips) >= 3

    def ns(hms):   # 2026-10-06 EDT (UTC-4)
        t = dt.datetime(2026, 10, 6, *hms, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        return int(t.timestamp()) * 1_000_000_000
    # the 19:45:01 loss precedes the fixture range (events start 19:45:10, minus 5 s pad)
    for hms in ((19, 46, 0), (19, 47, 0), (19, 47, 45)):
        assert any(abs(s["t_ns"] - ns(hms)) <= 1_000_000_000 for s in slips), hms


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


def test_align_labels_innermost_step(tmp_path):
    S0, S1 = T0 + 5_800_000_000, T0 + 6_600_000_000          # step "forward" wraps layer "0"
    N0, N1 = T0 + 6_050_000_000, T0 + 6_150_000_000          # nested step inside it
    extra = [("step_begin", "forward", 0, "", S0), ("step_begin", "inner", 0, "", N0),
             ("step_end", "inner", 0, "", N1), ("step_end", "forward", 0, "", S1)]
    rec, pi, (L0, L1, C0, C1) = _build(tmp_path, extra_events=extra)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    assert at((N0 + N1) // 2)["step"] == "inner"
    r = at((C1 + L1) // 2)
    assert r["step"] == "forward" and r["layer_path"] == "0"
    assert at(S1 + 100_000_000)["step"] == ""
    assert at(S0 - 100_000_000)["step"] == ""


# ---- concurrent runs: tagged step events ---------------------------------------------------

def _tagged_run(tmp_path):
    """m0 steps (iter 4) every 300 ms and m1 steps (its own labels, iter 99) overlapping them."""
    ev = []
    for i in range(6):
        a = T0 + 7_000_000_000 + i * 300_000_000
        ev += [("step_begin", "forward", 0, "", a, "forward", "4", "m0"),
               ("step_end", "forward", 0, "", a + 200_000_000, "forward", "4", "m0")]
    for i in range(3):
        a = T0 + 7_100_000_000 + i * 500_000_000
        ev += [("step_begin", "spin", 0, "", a, "forward", "99", "m1"),
               ("step_end", "spin", 0, "", a + 300_000_000, "forward", "99", "m1")]
    return _build(tmp_path, extra_events=ev)


def test_align_concurrent_labels_tags_and_keeps_primary_mode_iter(tmp_path):
    rec, pi, _ = _tagged_run(tmp_path)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")

    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    a0 = T0 + 7_000_000_000
    both = at(a0 + 150_000_000)                    # m0 step 0 and m1 step 0 are both open
    assert both["tag"] == "m0+m1" and both["step"] == "forward" and both["step_m1"] == "spin"
    only_m1 = at(a0 + 250_000_000)                 # m0 between steps, m1 (until +400 ms) running
    assert only_m1["tag"] == "m1" and only_m1["step"] == "" and only_m1["step_m0"] == ""
    only_m0 = at(a0 + 450_000_000)                 # m1 step 0 over (+100..+400), step 1 not yet
    assert only_m0["tag"] == "m0" and only_m0["step_m1"] == ""
    assert at(a0 - 200_000_000)["tag"] == ""
    # mode/iter follow the primary's events only (m1 events carry iter 99)
    assert set(df["iter"]) <= {"", "0", "4"} and "99" not in set(df["iter"])
    assert {c for c in df.columns if c.startswith("step_")} == {"step_m0", "step_m1"}


def test_align_drain_marker_ends_the_primary_frame_label(tmp_path):
    D = T0 + 8_000_000_000            # the primary's last step ends at D - 0.1 s; the worker drains until D + 0.3 s
    ev = [("step_begin", "forward", 0, "", D - 400_000_000, "forward", "7", "m0"),
          ("step_end", "forward", 0, "", D - 100_000_000, "forward", "7", "m0"),
          ("marker", "concurrent_drain", 0, "", D - 100_000_000, "forward", "7", ""),
          ("step_begin", "spin", 0, "", D - 300_000_000, "forward", "7", "m1"),
          ("step_end", "spin", 0, "", D + 300_000_000, "forward", "7", "m1"),
          ("marker", "baseline_B", 0, "", D + 400_000_000, "forward", "7", "")]
    rec, pi, _ = _build(tmp_path, extra_events=ev)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")

    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    inside = at(D - 200_000_000)
    assert inside["phase"] == "forward" and inside["iter"] == "7" and inside["tag"] == "m0+m1"
    drain = at(D + 100_000_000)       # the worker's tail: its own phase, no mode/iter of the frame
    assert drain["phase"] == "concurrent_drain" and drain["mode"] == "" and drain["iter"] == ""
    assert drain["tag"] == "m1" and drain["step_m1"] == "spin"


def test_align_without_tags_adds_no_tag_columns(tmp_path):
    rec, pi, _ = _build(tmp_path)                  # the v1-shaped events.csv has no tag column
    assert "tag" not in open(pi / "events.csv").readline()
    align(rec, pi, pi)
    cols = set(pd.read_parquet(pi / "power_trace.parquet").columns)
    assert "tag" not in cols and not any(c.startswith("step_") for c in cols)


def test_align_untagged_events_with_a_tag_column_add_no_tag_columns(tmp_path):
    ev = [("step_begin", "forward", 0, "", T0 + 7_000_000_000, "forward", "4", ""),
          ("step_end", "forward", 0, "", T0 + 7_200_000_000, "forward", "4", "")]
    rec, pi, _ = _build(tmp_path, extra_events=ev)
    align(rec, pi, pi)
    cols = set(pd.read_parquet(pi / "power_trace.parquet").columns)
    assert "tag" not in cols and "step_m0" not in cols


# ---- counters.csv / sysfs.csv merge ----------------------------------------------------

def _ctr(rows):
    return pd.DataFrame(rows, columns=["t_start_ns", "t_end_ns", "ctr_instructions",
                                       "pct_instructions"])


def test_merge_counters_interval_is_open_closed_and_edges_are_nan():
    ctr = _ctr([(1000, 2000, 10, 100.0), (2000, 3000, 20, 90.0), (4000, 5000, 40, 80.0)])
    df = pd.DataFrame({"t_pi_ns": np.array([500, 1000, 1001, 2000, 2001, 3000, 3500, 4000,
                                            4500, 5000, 5001], dtype=np.int64)})
    out = merge_counters(df, ctr)
    got = out["ctr_instructions"].tolist()
    nan = float("nan")
    # 500 before the first interval; 1000 == t_start of the first is NOT inside it;
    # 2000 == end of the first belongs to the first; 3500 is in the gap; 5000 is the last end
    expect = [nan, nan, 10, 10, 20, 20, nan, nan, 40, 40, nan]
    assert all((np.isnan(g) and np.isnan(e)) or g == e for g, e in zip(got, expect)), got
    assert out["pct_instructions"].iloc[4] == 90.0
    assert out["ctr_instructions"].dtype == np.float64
    assert "t_start_ns" not in out.columns and "t_end_ns" not in out.columns
    assert out["t_pi_ns"].tolist() == df["t_pi_ns"].tolist()


def test_merge_counters_header_only_gives_nan_columns():
    out = merge_counters(pd.DataFrame({"t_pi_ns": np.arange(3, dtype=np.int64)}),
                         _ctr([]))
    assert out["ctr_instructions"].isna().all() and out["pct_instructions"].isna().all()


def test_merge_counters_keeps_missing_cells_as_nan():
    ctr = _ctr([(0, 1000, np.nan, 0.0)])
    out = merge_counters(pd.DataFrame({"t_pi_ns": np.array([500], dtype=np.int64)}), ctr)
    assert np.isnan(out["ctr_instructions"].iloc[0]) and out["pct_instructions"].iloc[0] == 0.0


def test_merge_sysfs_backward_with_2s_tolerance():
    sysfs = pd.DataFrame({"t_real_ns": [1_000_000_000, 2_000_000_000],
                          "cpu0_kHz": [1500000, 2400000], "temp_C": [50.0, 55.0],
                          "fan_state": [0, 1]})
    t = np.array([500_000_000, 1_000_000_000, 1_999_999_999, 2_000_000_000,
                  3_999_999_999, 4_000_000_001], dtype=np.int64)
    out = merge_sysfs(pd.DataFrame({"t_pi_ns": t}), sysfs)
    assert out["sys_cpu0_kHz"].tolist()[1:5] == [1500000, 1500000, 2400000, 2400000]
    assert np.isnan(out["sys_cpu0_kHz"].iloc[0])          # before the first reading
    assert np.isnan(out["sys_cpu0_kHz"].iloc[5])          # reading is older than 2 s
    assert out["sys_temp_C"].iloc[3] == 55.0 and out["sys_fan_state"].iloc[1] == 0
    assert "sys_t_real_ns" not in out.columns


def test_attach_pi_logs_missing_files_add_nothing(tmp_path):
    df = pd.DataFrame({"t_pi_ns": np.arange(3, dtype=np.int64), "P_W": [1.0, 2.0, 3.0]})
    assert attach_pi_logs(df, tmp_path).equals(df)


def test_align_merges_counters_and_sysfs(tmp_path):
    rec, pi, _ = _build(tmp_path)
    t_s = T0 + 5_000_000_000
    rows = [(t_s + i * 100_000_000, t_s + (i + 1) * 100_000_000, 1000 + i, 100.0)
            for i in range(20)]
    _ctr(rows).to_csv(pi / "counters.csv", index=False)
    pd.DataFrame({"t_real_ns": [T0 + k * 1_000_000_000 for k in range(18)],
                  "cpu0_kHz": 2400000, "cpu1_kHz": 2400000, "cpu2_kHz": 2400000,
                  "cpu3_kHz": 2400000, "temp_C": 61.5, "fan_state": 4}).to_csv(
        pi / "sysfs.csv", index=False)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    t = df["t_pi_ns"].to_numpy()
    inside = (t > t_s) & (t <= t_s + 2_000_000_000)
    assert df.loc[inside, "ctr_instructions"].notna().all()
    assert df.loc[~inside, "ctr_instructions"].isna().all()
    k = np.minimum((t[inside] - t_s - 1) // 100_000_000, 19)
    assert (df.loc[inside, "ctr_instructions"].to_numpy() == 1000 + k).all()
    assert df["sys_cpu0_kHz"].notna().all() and (df["sys_temp_C"] == 61.5).all()
    assert (df["sys_fan_state"] == 4).all()
    assert df["P_W"].dtype == np.float32


def test_align_without_pi_logs_has_v1_columns(tmp_path):
    rec, pi, _ = _build(tmp_path)
    align(rec, pi, pi)
    cols = pd.read_parquet(pi / "power_trace.parquet").columns
    assert not any(c.startswith(("ctr_", "pct_", "sys_")) for c in cols)
