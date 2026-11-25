"""Supply-harness loss: R * integral of I^2 dt per window."""
import math

import numpy as np
import pytest

import data
import harness


def test_rigs_and_resistances():
    assert harness.R_OHM == {"oct2026": 0.179, "nov2026": 0.212}
    assert data.rig_of({"kernel": "6.18.50+rpt-rpi-2712"}) == "oct2026"
    assert data.rig_of({"kernel": "6.18.60+rpt-rpi-2712"}) == "nov2026" and data.rig_of({}) == "nov2026"


def test_constant_current_is_r_i2_duration():
    t = np.linspace(0, 10, 10_001)
    h = harness.harness_mJ(t, np.full_like(t, 2.0), [1.0, 3.0], [1.5, 9.0], 0.212)
    assert h == pytest.approx([0.212 * 4.0 * 0.5 * 1000, 0.212 * 4.0 * 6.0 * 1000])


def test_window_edges_are_interpolated_between_samples():
    t = np.array([0.0, 1.0, 2.0])
    I = np.array([1.0, 1.0, 3.0])          # I^2 1, 1, 9: cumulative trapezoid 0, 1, 6, read linearly at the edges
    h = harness.harness_mJ(t, I, [0.25, 0.0], [1.5, 2.0], 0.5)
    assert h == pytest.approx([0.5 * (3.5 - 0.25) * 1000, 0.5 * 6.0 * 1000])


def test_zero_length_window_is_zero():
    t = np.linspace(0, 1, 11)
    assert harness.harness_mJ(t, np.ones(11), [0.5], [0.5], 0.2) == pytest.approx([0.0])


def test_windows_outside_the_trace_are_nan():
    t = np.linspace(0, 10, 101)
    h = harness.harness_mJ(t, np.ones(101), [-1.0, 9.0, 5.0, 0.0], [2.0, 11.0, 4.0, 10.0], 0.2)
    assert np.isnan(h[:3]).all()            # starts before, ends after, end before start
    assert h[3] == pytest.approx(0.2 * 10 * 1000)


def test_nan_inputs_give_nan():
    t = np.linspace(0, 10, 101)
    assert np.isnan(harness.harness_mJ(t, np.ones(101), [1.0, 2.0], [2.0, 3.0], math.nan)).all()
    assert np.isnan(harness.harness_mJ(t, np.ones(101), [math.nan, 2.0], [2.0, math.nan], 0.2)).all()
    bad = np.ones(101)
    bad[3] = np.nan
    assert np.isnan(harness.harness_mJ(t, bad, [1.0], [2.0], 0.2)).all()
    assert np.isnan(harness.harness_mJ([0.0], [1.0], [0.0], [0.0], 0.2)).all()    # one sample
    assert harness.harness_mJ(t, np.ones(101), [], [], 0.2).shape == (0,)
