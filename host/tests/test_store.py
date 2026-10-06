import numpy as np

from store import SampleStore, load_blocks, open_samples


def test_append_and_read_roundtrip(tmp_path):
    s = SampleStore(tmp_path)
    s.append(np.array([0.5, 0.6]), np.array([5.0, 5.1]), host_ns=1000, dropped_cum=0)
    s.append(np.array([0.7]), np.array([5.2]), host_ns=2000, dropped_cum=0)
    i, v = s.read(0, 3)
    assert np.allclose(i, [0.5, 0.6, 0.7]) and np.allclose(v, [5.0, 5.1, 5.2])
    assert s.n_total == 3
    b = load_blocks(tmp_path)
    assert list(b["first_idx"]) == [0, 2] and list(b["host_ns"]) == [1000, 2000]
    assert open_samples(tmp_path).shape == (3, 2)


def test_dropped_samples_insert_nan_gap(tmp_path):
    s = SampleStore(tmp_path)
    s.append(np.ones(2), np.ones(2), host_ns=1, dropped_cum=0)
    s.append(np.ones(2), np.ones(2), host_ns=2, dropped_cum=3)
    i, _ = s.read(0, s.n_total)
    assert s.n_total == 7 and s.dropped == 3
    assert np.isnan(i[2:5]).all() and np.isfinite(i[[0, 1, 5, 6]]).all()
    b = load_blocks(tmp_path)
    assert list(b["first_idx"]) == [0, 5] and list(b["gap_inserted"]) == [0, 3]


def test_reopen_appends_not_truncates(tmp_path):
    SampleStore(tmp_path).append(np.ones(2), np.ones(2), host_ns=1, dropped_cum=0)
    s2 = SampleStore(tmp_path)
    assert s2.n_total == 2
    s2.append(np.ones(1), np.ones(1), host_ns=2, dropped_cum=0)
    assert open_samples(tmp_path).shape == (3, 2)
