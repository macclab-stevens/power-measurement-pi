"""Slip correction of the trace time axis (timefix.py): the step model, its sign, and its use in windows.py."""
import json

import numpy as np
import pandas as pd
import pytest

import data
import timefix
import windows
from test_windows import B, MS, counters_frame, run, write_cell

S = 1_000_000_000


def pulses(t0_s, t1_s):
    """Two trains of 6 edges, 10 ms apart, starting at t0_s and ending at t1_s (Pi-clock ns)."""
    return np.concatenate([t0_s * S + np.arange(6) * 10 * MS, t1_s * S - np.arange(6)[::-1] * 10 * MS])


def al(slips, c0=0.0):
    return {"c0_ns": c0, "slips": slips}


def test_midway_slip_gives_minus_half_then_plus_half_step():
    p = pulses(0, 100)                                   # ta ~ 25 ms, tb ~ 99.975 s: midpoint ~ 50 s
    ta, tb = p[:6].mean(), p[-6:].mean()
    mid = (ta + tb) / 2
    step_ms = 8.0
    a = al([{"t_ns": mid, "step_ms": step_ms, "samples": 40}])          # c0 = 0: host time == Pi time
    t = np.array([mid - 1e6, mid + 1e6, ta, tb])
    c = timefix.slip_correction_ns(a, p, t)
    assert c[0] == pytest.approx(-step_ms * 1e6 / 2, abs=2e3)         # just before the slip: -step/2
    assert c[1] == pytest.approx(+step_ms * 1e6 / 2, abs=2e3)         # just after: +step/2 (later in time)
    assert abs(c[2]) < 1.0 and abs(c[3]) < 1.0                           # exact at the pulse-train means


def test_slip_time_is_host_time_minus_c0():
    p = pulses(0, 100)
    mid = (p[:6].mean() + p[-6:].mean()) / 2
    a = al([{"t_ns": mid + 3e6, "step_ms": 6.0, "samples": 30}], c0=3e6)     # Pi-clock slip time = t_ns - c0 = mid
    c = timefix.slip_correction_ns(a, p, np.array([mid - 1e5, mid + 1e5]))
    assert c[0] < -2.9e6 and c[1] > 2.9e6


def test_no_slip_cell_gets_zero_correction_and_no_pulses_gives_none():
    t = np.linspace(0, 100e9, 50)
    assert np.array_equal(timefix.slip_correction_ns(al([]), pulses(0, 100), t), np.zeros(50))
    assert np.array_equal(timefix.slip_correction_ns({"c0_ns": 0.0}, np.empty(0), t), np.zeros(50))
    slipped = al([{"t_ns": 50e9, "step_ms": 5.0, "samples": 25}])
    assert timefix.slip_correction_ns(slipped, pulses(0, 100)[:11], t) is None


def test_slip_outside_the_pulse_trains():
    p = pulses(10, 90)
    t = np.array([5e9, 20e9, 95e9])
    early = timefix.slip_correction_ns(al([{"t_ns": 1e9, "step_ms": 4.0, "samples": 20}]), p, t)
    assert early == pytest.approx([0.0, 0.0, 0.0], abs=1.0)           # slip before the start train: all samples after it
    before = timefix.slip_correction_ns(al([{"t_ns": 7e9, "step_ms": 4.0, "samples": 20}]), p, np.array([5e9, 20e9]))
    assert before == pytest.approx([-4e6, 0.0], abs=1.0)             # samples before the slip move by -step
    late = timefix.slip_correction_ns(al([{"t_ns": 93e9, "step_ms": 4.0, "samples": 20}]), p, np.array([20e9, 95e9]))
    assert late == pytest.approx([0.0, 4e6], abs=1.0)


def test_two_slips_add_up():
    p = pulses(0, 100)
    s1, s2 = 30e9, 70e9
    t = np.array([50e9])
    both = timefix.slip_correction_ns(al([{"t_ns": s1, "step_ms": 5.0, "samples": 25},
                                          {"t_ns": s2, "step_ms": 3.0, "samples": 15}]), p, t)
    one = [timefix.slip_correction_ns(al([s]), p, t) for s in ({"t_ns": s1, "step_ms": 5.0}, {"t_ns": s2, "step_ms": 3.0})]
    assert both == pytest.approx(one[0] + one[1])


def test_needs_correction():
    slip = [{"t_ns": 1.0, "step_ms": 2.0, "samples": 10}]
    assert timefix.needs_correction(al(slip))
    assert not timefix.needs_correction({**al(slip), "slip_corrected": True})
    assert not timefix.needs_correction(al([])) and not timefix.needs_correction(None)


def test_slip_time_uses_the_drift_term_of_the_fit():
    """align.py maps host to Pi time as (t - c0 + c1 ref) / (1 + c1); with 100 ppm fake drift and the slip 50 s
    after ref the slip sits ~5 ms earlier on the Pi clock than t - c0 says."""
    p = pulses(0, 100)
    mid = (p[:6].mean() + p[-6:].mean()) / 2
    c1, ref = 1e-4, float(p[0])
    t_host = mid * (1 + c1) - c1 * ref                    # host time that maps exactly to Pi time `mid`
    a = {"c0_ns": 0.0, "c1": c1, "ref_ns": ref, "slips": [{"t_ns": t_host, "step_ms": 6.0, "samples": 30}]}
    c = timefix.slip_correction_ns(a, p, np.array([mid - 1e5, mid + 1e5]))
    assert c[0] < -2.9e6 and c[1] > 2.9e6                 # the step sits at `mid`, not ~5 ms later


# ------------------------------------------------------------------ in windows.py

def slip_cell(tmp_path, name, alignment, with_pulses=True, n_win=5):
    """5 x 100 ms windows, 1 ms samples, P = 3 + 0.01 k at B + k ms; start train at ~3.5 ms, end train at ~496.5 ms."""
    starts = B + np.arange(n_win) * 100 * MS
    ctr = counters_frame(starts, starts + 100 * MS)
    k = np.arange(1, n_win * 100 + 1)
    cell = write_cell(tmp_path, name, B + k * MS, 3 + 0.01 * k, ctr)
    (cell / "alignment.json").write_text(json.dumps({"slip_rate_hz": 1000.0, **alignment}))
    if with_pulses:
        t = B + np.concatenate([np.arange(1, 7), np.arange(494, 500)]) * MS
        pd.DataFrame({"t_real_ns": t, "kind": ["pulse_on", "pulse_off"] * 6}).to_csv(cell / "events.csv", index=False)
    return cell, k


def expected_means(cell, alignment, k):
    t = B + k * MS
    ev = pd.read_csv(cell / "events.csv")
    t2 = t + np.round(timefix.slip_correction_ns(alignment, ev["t_real_ns"].to_numpy(), t)).astype(np.int64)
    P = 3 + 0.01 * k
    starts = B + np.arange(5) * 100 * MS
    return np.array([P[(t2 > s) & (t2 <= s + 100 * MS)].mean() for s in starts])


SLIP = {"c0_ns": 0.0, "slips": [{"t_ns": B + 250 * MS, "step_ms": 10.0, "samples": 10}]}


def test_windows_labels_use_the_corrected_time_axis(tmp_path):
    cell, k = slip_cell(tmp_path, "s0", SLIP)
    df = run(cell).sort_values("w_idx")
    plain = 3 + 0.01 * (np.arange(5) * 100 + 50.5)
    assert df["P_W"].to_numpy() == pytest.approx(expected_means(cell, SLIP, k))
    assert not np.allclose(df["P_W"], plain, atol=1e-6)
    assert df["P_W"].iloc[2] != pytest.approx(plain[2], abs=1e-3)           # the window holding the slip moves most
    s = df.attrs["slip"]
    assert s["status"] == "corrected" and 4.0 < s["max_corr_ms"] <= 5.0 and s["dP_max_W"] > 0.01
    assert s["cell_id"] == "s0" and s["n_windows"] == 5


def test_already_corrected_or_slip_free_cells_are_untouched(tmp_path):
    plain = 3 + 0.01 * (np.arange(5) * 100 + 50.5)
    cell, _ = slip_cell(tmp_path, "s1", {**SLIP, "slip_corrected": True})
    df = run(cell)
    assert df["P_W"].to_numpy() == pytest.approx(plain) and df.attrs["slip"]["status"] == "none"
    cell, _ = slip_cell(tmp_path, "s2", {"c0_ns": 0.0, "slips": []})
    assert run(cell)["P_W"].to_numpy() == pytest.approx(plain)
    cell, _ = slip_cell(tmp_path, "s3", SLIP)
    off = windows.cell_windows(cell, data.read_meta(cell), slip_correct=False)
    assert off["P_W"].to_numpy() == pytest.approx(plain)


def test_slip_cell_without_pulses_is_left_alone_and_counted(tmp_path):
    cell, _ = slip_cell(tmp_path, "s4", SLIP, with_pulses=False)
    df = run(cell)
    assert df["P_W"].to_numpy() == pytest.approx(3 + 0.01 * (np.arange(5) * 100 + 50.5))
    assert df.attrs["slip"]["status"] == "no_pulses"


def test_build_windows_aggregates_slip_attrs(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    slip_cell(tmp_path, "s5", SLIP)
    slip_cell(tmp_path, "s6", SLIP, with_pulses=False)
    slip_cell(tmp_path, "s7", {"c0_ns": 0.0, "slips": []})
    df = windows.build_windows(data.catalogue())
    a = df.attrs
    assert a["slip_corrected"] == 1 and a["slip_no_pulses"] == 1 and [c["cell_id"] for c in a["slip_cells"]] == ["s5"]


def test_phase_labels_are_read_at_the_corrected_times(tmp_path):
    """A sample moved later by the correction takes the label in force at its corrected time."""
    cell, k = slip_cell(tmp_path, "s3", SLIP)
    t0, P0, c0, names, _ = windows.corrected_trace(cell, False)
    t1, P1, c1, names1, info = windows.corrected_trace(cell, True)
    assert info["status"] == "corrected" and list(names1) == list(names)
    j = np.clip(np.searchsorted(t0, t1, side="right") - 1, 0, len(t0) - 1)
    assert (c1 == c0[j]).all()
