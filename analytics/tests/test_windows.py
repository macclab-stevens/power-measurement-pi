import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import data
import runlog
import settings
import windows
from conftest import make_cell

B = 10_000_000_000          # Pi-clock ns at trace start (= run_meta t0_real_ns)
MS = 1_000_000
DT_HZ = 1000.0              # 1 ms samples
FEATURES = ["busy", "cycles_rate", "instr_rate", "ipc", "cache_miss_rate", "branch_mpki", "stall_frac",
            "llc_mpki", "ctx_sw_rate", "pf_rate", "busy_l1", "busy_l2", "busy_l3", "cycles_rate_l1",
            "cycles_rate_l2", "cycles_rate_l3", "ipc_l1", "ipc_l2", "ipc_l3", "pct_min", "freq_mhz", "temp_C",
            "fan_state", "uv_alarm", "sysfs_age_s", "pmic_W", "pmic_age_s", "cpu_key", "fan_mode", "governor"]
CTR_COLS = ["cycles", "instructions", "cache_references", "cache_misses", "branch_misses",
            "stalled_cycles_backend", "LLC_load_misses", "task_clock", "context_switches", "page_faults"]


def counters_frame(starts, ends, rng=None, scale=1.0):
    n = len(starts)
    rng = rng or np.random.default_rng(0)
    d = {"t_start_ns": starts, "t_end_ns": ends}
    for c in CTR_COLS:
        d["ctr_" + c] = (rng.integers(1_000, 1_000_000, n) * scale).astype(float)
    dur = (np.asarray(ends) - np.asarray(starts)) / 1e9
    d["ctr_task_clock"] = rng.uniform(0.1, 1.0, n) * dur * 4e9
    for c in CTR_COLS:
        d["pct_" + c] = 100.0
    return pd.DataFrame(d)


def sysfs_frame(ts, ids):
    return pd.DataFrame({"t_real_ns": ts, "cpu0_kHz": 2_400_000, "cpu1_kHz": 2_400_000, "cpu2_kHz": 1_200_000,
                         "cpu3_kHz": 1_200_000, "temp_C": ids, "fan_state": 4, "uv_alarm": 0})


def context_frame(t_ms, ids):
    """Two paired rails (I = id, V = 2, and I = 1, V = 3) and one unpaired voltage rail."""
    return pd.DataFrame({"t_ms": t_ms, "temp_C": 40.0, "arm_MHz": 2400, "throttle_bits": 0,
                         "rail_A_A current(0)": np.asarray(ids, float), "rail_B_A current(1)": 1.0,
                         "rail_A_V volt(2)": 2.0, "rail_B_V volt(3)": 3.0, "rail_EXT5V_V volt(4)": 5.0})


def write_cell(tmp_path, name, trace_t, trace_P, ctr, sysfs=None, ctx=None, phase=None, meta=None,
               hz=DT_HZ, source="v2"):
    files = {"alignment.json": {"slip_rate_hz": hz, "slips": []}}
    cell = make_cell(tmp_path, source, name, meta={"t0_real_ns": B, "perf_counters": {
        "enabled": True, "interval_ms": 100, "offset_ns": 0, "events": []}, **(meta or {})}, files=files)
    phase = ["forward"] * len(trace_t) if phase is None else phase
    pq.write_table(pa.table({"t_pi_ns": np.asarray(trace_t, dtype=np.int64),
                             "P_W": np.asarray(trace_P, dtype=np.float32), "phase": phase}),
                   cell / "power_trace.parquet")
    if ctr is not None:
        ctr.to_csv(cell / "counters.csv", index=False)
    if sysfs is not None:
        sysfs.to_csv(cell / "sysfs.csv", index=False)
    if ctx is not None:
        ctx.to_csv(cell / "context.csv", index=False)
    return cell


def simple_cell(tmp_path, n_win=5, name="c0", drop_samples=None, **kw):
    """n_win contiguous 100 ms intervals; a 1 ms trace with P = 3 + 0.01 k at B + k ms (k = 1..)."""
    k = np.arange(1, n_win * 100 + 1)
    if drop_samples is not None:
        k = k[~np.isin(k, drop_samples)]
    starts = B + np.arange(n_win) * 100 * MS
    ctr = counters_frame(starts, starts + 100 * MS)
    ctr["ctr_task_clock"] = (np.arange(n_win) + 1) * 4e7          # busy = (i + 1) / 10
    ctr["ctr_cycles"] = (np.arange(n_win) + 1) * 1e6
    return write_cell(tmp_path, name, B + k * MS, 3 + 0.01 * k, ctr, **kw), ctr


def run(cell):
    return windows.cell_windows(cell, data.read_meta(cell))


# ------------------------------------------------------------------ the brief's synthetic tests

def test_sysfs_one_ns_after_end_not_used_and_pmic_at_end_used(tmp_path):
    sysfs = sysfs_frame([B + 50 * MS, B + 100 * MS + 1], [10.0, 99.0])
    ctx = context_frame([0.0, 100.0], [7, 8])          # second read exactly at window 0's end
    cell, _ = simple_cell(tmp_path, sysfs=sysfs, ctx=ctx)
    df = run(cell)
    w0 = df[df["w_idx"] == 0].iloc[0]
    assert w0["temp_C"] == 10.0                         # the 1 ns later row is not used
    assert w0["sysfs_age_s"] == pytest.approx(0.05)
    assert w0["pmic_W"] == pytest.approx(8 * 2 + 1 * 3)  # read at the window end is used (<=)
    assert w0["pmic_age_s"] == pytest.approx(0.0)
    w1 = df[df["w_idx"] == 1].iloc[0]
    assert w1["temp_C"] == 99.0 and w1["sysfs_age_s"] == pytest.approx(0.1 - 1e-9 + 0.0, abs=1e-6)
    assert w1["pmic_W"] == pytest.approx(8 * 2 + 3)
    assert df.attrs["pmic_pairs"] == 2


def test_no_sysfs_or_pmic_before_window_end_gives_nan(tmp_path):
    sysfs = sysfs_frame([B + 150 * MS], [10.0])
    ctx = context_frame([100.5], [7])
    cell, _ = simple_cell(tmp_path, sysfs=sysfs, ctx=ctx)
    df = run(cell)
    w0 = df.iloc[0]
    assert np.isnan(w0["temp_C"]) and np.isnan(w0["sysfs_age_s"])
    assert np.isnan(w0["pmic_W"]) and np.isnan(w0["pmic_age_s"])
    assert df.iloc[1]["pmic_W"] == pytest.approx(7 * 2 + 3)    # read at 100.5 ms <= end of window 1 (200 ms)


def test_lags_come_from_previous_interval_only(tmp_path):
    cell, _ = simple_cell(tmp_path)
    df = run(cell).sort_values("w_idx").reset_index(drop=True)
    assert df["busy"].round(6).tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert df.loc[0, ["busy_l1", "busy_l2", "busy_l3", "cycles_rate_l1", "ipc_l1"]].isna().all()
    assert np.isnan(df.loc[1, "busy_l2"]) and np.isnan(df.loc[2, "busy_l3"])
    assert df.loc[4, "busy_l1"] == pytest.approx(0.4)
    assert df.loc[4, "busy_l2"] == pytest.approx(0.3)
    assert df.loc[4, "busy_l3"] == pytest.approx(0.2)
    assert df.loc[3, "cycles_rate_l1"] == pytest.approx(df.loc[2, "cycles_rate"])
    assert df.loc[3, "ipc_l2"] == pytest.approx(df.loc[1, "ipc"])


def test_lags_taken_before_dropping_low_coverage_windows(tmp_path):
    cell, _ = simple_cell(tmp_path, drop_samples=np.arange(101, 200))   # window 1 nearly empty
    df = run(cell)
    assert df.attrs["dropped"] == 1 and 1 not in df["w_idx"].tolist()
    w2 = df[df["w_idx"] == 2].iloc[0]
    assert w2["busy_l1"] == pytest.approx(0.2)     # still window 1, not window 0
    assert w2["busy_l2"] == pytest.approx(0.1)


def test_P_W_is_trace_mean_and_energy_sum(tmp_path):
    cell, _ = simple_cell(tmp_path)
    df = run(cell).sort_values("w_idx")
    for i, p in enumerate(df["P_W"]):
        k = np.arange(i * 100 + 1, i * 100 + 101)      # samples in (t_start, t_end]
        assert p == pytest.approx((3 + 0.01 * k).mean(), abs=1e-4)
    assert df["n_samples"].tolist() == [100] * 5
    assert df["cover_frac"].tolist() == pytest.approx([1.0] * 5)
    assert df["E_mJ"].iloc[0] == pytest.approx((3 + 0.01 * np.arange(1, 101)).sum() * 1e-3 * 1000, rel=1e-5)
    assert df["dur_s"].iloc[0] == pytest.approx(0.1)
    assert (df["phase"] == "forward").all() and (df["phase_frac"] == 1.0).all()
    assert df.attrs["dropped"] == 0


def test_window_with_half_its_samples_missing_is_dropped(tmp_path):
    cell, _ = simple_cell(tmp_path, drop_samples=np.arange(201, 251))   # 50 of window 2's 100 samples
    df = run(cell)
    assert df.attrs["dropped"] == 1
    assert df["w_idx"].tolist() == [0, 1, 3, 4]


def test_coverage_boundary(tmp_path):
    cell, _ = simple_cell(tmp_path, drop_samples=np.arange(1, 21))      # 80 of 100: kept (>= 0.8)
    df = run(cell)
    assert df.attrs["dropped"] == 0 and df["cover_frac"].iloc[0] == pytest.approx(0.8)


def test_majority_phase(tmp_path):
    k = np.arange(1, 501)
    ph = ["warmup" if x <= 30 else "forward" for x in k]
    starts = B + np.arange(5) * 100 * MS
    ctr = counters_frame(starts, starts + 100 * MS)
    cell = write_cell(tmp_path, "c0", B + k * MS, np.full(500, 3.0), ctr, phase=ph)
    df = run(cell).sort_values("w_idx")
    assert df["phase"].tolist() == ["forward"] * 5
    assert df["phase_frac"].iloc[0] == pytest.approx(0.7)
    assert df["phase_frac"].iloc[1] == pytest.approx(1.0)


def test_no_counters_returns_none(tmp_path):
    cell = write_cell(tmp_path, "c0", [B + MS], [3.0], None)
    assert windows.cell_windows(cell, data.read_meta(cell)) is None
    cell = write_cell(tmp_path, "c1", [B + MS], [3.0], None,
                      meta={"perf_counters": {"enabled": False}})
    ctr = counters_frame([B], [B + 100 * MS])
    ctr.to_csv(cell / "counters.csv", index=False)
    assert windows.cell_windows(cell, data.read_meta(cell)) is None


def test_zero_denominators_guarded(tmp_path):
    starts = B + np.arange(3) * 100 * MS
    ctr = counters_frame(starts, starts + 100 * MS)
    ctr.loc[1, ["ctr_cycles", "ctr_instructions", "ctr_cache_references"]] = 0.0
    k = np.arange(1, 301)
    cell = write_cell(tmp_path, "c0", B + k * MS, np.full(300, 3.0), ctr)
    df = run(cell).sort_values("w_idx")
    r = df.iloc[1]
    for c in ("ipc", "stall_frac", "cache_miss_rate", "branch_mpki", "llc_mpki"):
        assert np.isnan(r[c]), c
    assert r["cycles_rate"] == 0.0
    assert df.iloc[0][["ipc", "stall_frac", "cache_miss_rate"]].notna().all()
    assert df.attrs["guards"] == 1


def test_settings_columns(tmp_path):
    cell, _ = simple_cell(tmp_path, meta={"governor": "ondemand", "fan_mode": "auto", "cpu_mhz_pinned": None})
    df = run(cell)
    assert set(df["cpu_key"]) == {"ondemand"} and set(df["fan_mode"]) == {"auto"}
    assert set(df["governor"]) == {"ondemand"}
    cell, _ = simple_cell(tmp_path, name="c1", meta={"cpu_mhz_pinned": 1500})
    assert set(run(cell)["cpu_key"]) == {"1500"}


def test_freq_mhz_is_mean_of_cores(tmp_path):
    cell, _ = simple_cell(tmp_path, sysfs=sysfs_frame([B], [5.0]))
    assert run(cell)["freq_mhz"].iloc[0] == pytest.approx(1800.0)


def test_interval_assignment_semantics():
    t = np.array([5, 10, 11, 20, 21, 30], dtype=np.int64)
    lo, hi = windows.assign_intervals(t, np.array([10, 20]), np.array([20, 30]))
    # (t_start, t_end]: window 0 = samples 11, 20; window 1 = 21, 30
    assert t[lo[0]:hi[0]].tolist() == [11, 20] and t[lo[1]:hi[1]].tolist() == [21, 30]


# ------------------------------------------------------------------ causality (the important ones)

def random_cell(tmp_path, name, seed, n_win=40):
    rng = np.random.default_rng(seed)
    gaps = rng.integers(60, 140, n_win) * MS
    ends = B + np.cumsum(gaps)
    starts = np.concatenate([[B], ends[:-1]])
    ctr = counters_frame(starts, ends, rng)
    t_last = int(ends[-1])
    # irregular sysfs / context times, with deliberate hits exactly on and 1 ns after window ends
    st = np.sort(np.concatenate([B + rng.integers(0, t_last - B, 25), ends[[3, 10]], ends[[5, 12]] + 1]))
    sysfs = sysfs_frame(st, np.arange(len(st), dtype=float))
    ct = np.sort(np.concatenate([rng.integers(0, (t_last - B) // MS, 12) + rng.random(12), [(ends[7] - B) / MS]]))
    ctx = context_frame(ct, np.arange(len(ct)) + 1)
    k = np.arange(1, (t_last - B) // MS + 1)
    return write_cell(tmp_path, name, B + k * MS, np.full(len(k), 3.0), ctr, sysfs, ctx), ctr, sysfs, ctx


def test_sysfs_and_pmic_are_latest_at_or_before_window_end(tmp_path):
    cell, ctr, sysfs, ctx = random_cell(tmp_path, "c0", 1)
    df = run(cell)
    assert len(df) > 30
    ctx_t = B + np.round(ctx["t_ms"].to_numpy() * 1e6).astype(np.int64)
    for _, r in df.iterrows():
        end = int(ctr["t_end_ns"].iloc[int(r["w_idx"])])
        ok = sysfs["t_real_ns"] <= end
        if ok.any():
            j = np.flatnonzero(ok)[-1]
            assert r["temp_C"] == sysfs["temp_C"].iloc[j]
            assert r["sysfs_age_s"] == pytest.approx((end - sysfs["t_real_ns"].iloc[j]) / 1e9)
            assert r["sysfs_age_s"] >= 0
        ok = ctx_t <= end
        if ok.any():
            j = np.flatnonzero(ok)[-1]
            assert r["pmic_W"] == pytest.approx(ctx["rail_A_A current(0)"].iloc[j] * 2 + 3)
            assert r["pmic_age_s"] == pytest.approx((end - ctx_t[j]) / 1e9) and r["pmic_age_s"] >= 0


@pytest.mark.parametrize("cut", [4, 17, 29])
def test_features_do_not_change_when_the_future_is_altered(tmp_path, cut):
    """Replace everything that ends after window `cut` (later counters, sysfs rows, PMIC reads) with
    garbage: no feature of windows 0..cut may move. A look-ahead anywhere would show here."""
    cell, ctr, sysfs, ctx = random_cell(tmp_path, "orig", 2)
    base = run(cell).sort_values("w_idx")
    end = int(ctr["t_end_ns"].iloc[cut])
    rng = np.random.default_rng(99)
    ctr2 = ctr.copy()
    late = ctr2.index > cut
    for c in ctr2.columns[2:]:
        ctr2.loc[late, c] = rng.uniform(1, 1e9, late.sum())
    sys2 = sysfs.copy()
    late_s = sys2["t_real_ns"] > end
    sys2.loc[late_s, ["temp_C", "cpu0_kHz", "fan_state", "uv_alarm"]] = [-1.0, 1.0, 9, 9]
    ctx2 = ctx.copy()
    late_c = (B + np.round(ctx2["t_ms"] * 1e6)) > end
    ctx2.loc[late_c, "rail_A_A current(0)"] = -1000.0
    k = np.arange(1, (int(ctr["t_end_ns"].iloc[-1]) - B) // MS + 1)
    cell2 = write_cell(tmp_path, "altered", B + k * MS, rng.uniform(1, 9, len(k)), ctr2, sys2, ctx2)
    alt = run(cell2).sort_values("w_idx")
    a, b = base[base["w_idx"] <= cut], alt[alt["w_idx"] <= cut]
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a[FEATURES].reset_index(drop=True), b[FEATURES].reset_index(drop=True))
    # sanity: the alteration was not a no-op for later windows
    assert not base[base["w_idx"] > cut][["busy", "temp_C"]].reset_index(drop=True).equals(
        alt[alt["w_idx"] > cut][["busy", "temp_C"]].reset_index(drop=True))


def test_no_feature_uses_a_counter_interval_ending_after_the_window(tmp_path):
    """Window i's counter features equal a pure function of counters row i (and lags of rows i-1..i-3)."""
    cell, ctr, _, _ = random_cell(tmp_path, "c0", 3)
    df = run(cell).set_index("w_idx")
    dur = (ctr["t_end_ns"] - ctr["t_start_ns"]) / 1e9
    busy = ctr["ctr_task_clock"] / (dur * 4 * 1e9)
    for i, r in df.iterrows():
        assert r["busy"] == pytest.approx(busy[i])
        for lag in (1, 2, 3):
            want = busy[i - lag] if i - lag >= 0 else np.nan
            got = r[f"busy_l{lag}"]
            assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want)


def test_perf_start_offset_is_not_reapplied(tmp_path):
    """counters.csv is already offset-corrected: a nonzero offset_ns in run_meta must not shift windows."""
    cell, _ = simple_cell(tmp_path, name="a", meta={"perf_counters": {"enabled": True, "offset_ns": 39_000_000}})
    ref, _ = simple_cell(tmp_path, name="b")
    pd.testing.assert_series_equal(run(cell)["t_start_ns"], run(ref)["t_start_ns"])
    assert run(cell)["n_samples"].tolist() == [100] * 5


# ------------------------------------------------------------------ pmic windows

def test_pmic_windows_pair_of_consecutive_reads(tmp_path):
    ctx = context_frame([0.0, 100.0, 300.0], [1, 2, 3])
    cell, _ = simple_cell(tmp_path, ctx=ctx)
    df = windows.pmic_windows(cell, data.read_meta(cell))
    assert len(df) == 2
    r0, r1 = df.iloc[0], df.iloc[1]
    assert r0["t_start_ns"] == B and r0["t_end_ns"] == B + 100 * MS
    assert r0["pmic_W"] == pytest.approx(1 * 2 + 3)          # read i, known at the window start
    assert r1["pmic_W"] == pytest.approx(2 * 2 + 3)
    assert r1["dur_s"] == pytest.approx(0.2)
    assert r0["P_W"] == pytest.approx((3 + 0.01 * np.arange(1, 100)).mean(), abs=1e-4)   # [t_i, t_i+1): k = 1..99
    assert r1["E_mJ"] == pytest.approx((3 + 0.01 * np.arange(100, 300)).sum() * 1e-3 * 1000, rel=1e-5)
    assert set(df["cpu_key"]) == {"2400"} and set(df["fan_mode"]) == {"full"}
    assert df.attrs["pmic_pairs"] == 2


def test_pmic_windows_v1_cell_and_missing_context(tmp_path):
    ctx = context_frame([0.0, 100.0], [1, 2])
    cell = write_cell(tmp_path, "v1c", B + np.arange(1, 301) * MS, np.full(300, 3.0), None, ctx=ctx, source="v1")
    df = windows.pmic_windows(cell, data.read_meta(cell))
    assert len(df) == 1 and df["governor"].iloc[0] == "pinned" and df["fan_mode"].iloc[0] == "full"
    cell = write_cell(tmp_path, "v1d", B + np.arange(1, 301) * MS, np.full(300, 3.0), None, source="v1")
    assert len(windows.pmic_windows(cell, data.read_meta(cell))) == 0


# ------------------------------------------------------------------ build + cache

def test_build_windows_caches_and_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    simple_cell(tmp_path, name="c0", ctx=context_frame([0.0, 100.0], [1, 2]))
    make_cell(tmp_path, "v2", "v1ref_x")          # no counters, no trace
    cat = data.catalogue()
    df = windows.build_windows(cat)
    assert len(df) == 5 and set(df["cell_id"]) == {"c0"}
    assert df.attrs["dropped"] == 0 and df.attrs["n_cells"] == 1
    cache = settings.derived_dir() / "windows.parquet"
    assert cache.is_file()
    m0 = cache.stat().st_mtime_ns
    df2 = windows.build_windows(cat)
    assert cache.stat().st_mtime_ns == m0 and len(df2) == 5 and df2.attrs["dropped"] == 0
    cache.write_bytes(b"garbage")                  # unreadable cache is a miss
    assert len(windows.build_windows(cat)) == 5
    assert not list(settings.derived_dir().glob("*.tmp*"))
    pm = windows.build_pmic_windows(cat)
    assert len(pm) == 1 and (settings.derived_dir() / "pmic_windows.parquet").is_file()


# ------------------------------------------------------------------ real data

def _real_v2_cell(real_data, name="yolo11n_480_t4_2400_r00"):
    cell = real_data / "v2" / "cells" / name
    if not cell.is_dir():
        pytest.skip("cell not present")
    return cell


def test_real_cell_ages_energy_and_interval_assignment(real_data):
    cell = _real_v2_cell(real_data)
    meta = data.read_meta(cell)
    df = windows.cell_windows(cell, meta)
    assert len(df) > 1000
    assert (df["sysfs_age_s"].dropna() >= 0).all() and (df["pmic_age_s"].dropna() >= 0).all()
    assert df["pmic_age_s"].notna().mean() > 0.99 and df.attrs["pmic_pairs"] == 12
    tr = data.read_trace(cell, ["t_pi_ns", "P_W"])
    dt = 1.0 / json.loads((cell / "alignment.json").read_text())["slip_rate_hz"]
    ctr = pd.read_csv(cell / "counters.csv").sort_values("t_end_ns")
    t = tr["t_pi_ns"].to_numpy()
    m = (t > ctr["t_start_ns"].iloc[0]) & (t <= ctr["t_end_ns"].iloc[-1])
    trace_E = float(tr["P_W"].to_numpy(dtype=float)[m].sum() * dt * 1000)
    kept_E = float(df["E_mJ"].sum())
    assert abs(kept_E - trace_E) / trace_E < 0.01
    # the interval assignment agrees with the trace's own ctr_* columns written by the harness
    full = data.read_trace(cell, ["t_pi_ns", "ctr_cycles"])
    lo, hi = windows.assign_intervals(full["t_pi_ns"].to_numpy(), ctr["t_start_ns"].to_numpy(),
                                      ctr["t_end_ns"].to_numpy())
    cyc, want = full["ctr_cycles"].to_numpy(), ctr["ctr_cycles"].to_numpy()
    for w in range(0, len(ctr), 7):
        seg = cyc[lo[w]:hi[w]]
        assert hi[w] > lo[w]
        assert ((seg == want[w]) | (np.isnan(seg) & np.isnan(want[w]))).all()   # NaN intervals match too
