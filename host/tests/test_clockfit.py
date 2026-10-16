import numpy as np

from clockfit import SampleClock, fit_sample_clock


def _synthetic(seed=0, ppm=40.0, secs=300, block=50, lat_min_ms=2.0):
    rng = np.random.default_rng(seed)
    b = 200_000.0 * (1 + ppm * 1e-6)
    a = 1.7e18
    last_idx = np.arange(block - 1, int(secs * 5000), block)
    true_ns = a + b * last_idx
    host_ns = true_ns + lat_min_ms * 1e6 + rng.exponential(3e6, size=last_idx.size)
    return last_idx, host_ns, a + lat_min_ms * 1e6, b


def test_fit_recovers_slope_and_lower_envelope():
    last_idx, host_ns, a_env, b = _synthetic()
    c = fit_sample_clock(last_idx, host_ns)
    assert abs(c.b_ns / b - 1) < 3e-6
    probe = np.array([0, 500_000, 1_400_000])
    err_ms = np.abs(c.to_ns(probe) - (a_env + b * probe)) / 1e6
    assert err_ms.max() < 0.5, err_ms


def test_to_idx_inverts_to_ns():
    c = SampleClock(a_ns=1e18, b_ns=200_000.0)
    assert np.allclose(c.to_idx(c.to_ns(np.array([0, 12345]))), [0, 12345])


def test_short_series_falls_back_to_nominal_rate():
    c = fit_sample_clock(np.array([49]), np.array([1e18]))
    assert c.b_ns == 200_000.0 and c.to_ns(49) == 1e18
