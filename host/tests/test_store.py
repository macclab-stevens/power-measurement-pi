import numpy as np

from store import SampleStore, load_blocks, open_samples


def test_append_and_read_roundtrip(tmp_path):
    s = SampleStore(tmp_path)
    s.append(np.array([0.5, 0.6]), np.array([5.0, 5.1]), host_ns=1000, dropped_raw=0)
    s.append(np.array([0.7]), np.array([5.2]), host_ns=2000, dropped_raw=0)
    i, v = s.read(0, 3)
    assert np.allclose(i, [0.5, 0.6, 0.7]) and np.allclose(v, [5.0, 5.1, 5.2])
    assert s.n_total == 3
    b = load_blocks(tmp_path)
    assert list(b["first_idx"]) == [0, 2] and list(b["host_ns"]) == [1000, 2000]
    assert open_samples(tmp_path).shape == (3, 2)
    s.close()


def test_reopen_appends_not_truncates(tmp_path):
    s1 = SampleStore(tmp_path)
    s1.append(np.ones(2), np.ones(2), host_ns=1, dropped_raw=0)
    s1.close()
    s2 = SampleStore(tmp_path)
    assert s2.n_total == 2
    s2.append(np.ones(1), np.ones(1), host_ns=2, dropped_raw=0)
    assert open_samples(tmp_path).shape == (3, 2)
    s2.close()


def test_dropped_raw_is_recorded_not_interpreted(tmp_path):
    s = SampleStore(tmp_path)
    sizes = [127, 129, 128, 127]
    for n, raw in zip(sizes, [0, 14, 0, 14]):
        s.append(np.ones(n), np.ones(n), host_ns=n * 1000 + raw, dropped_raw=raw)
    i, _ = s.read(0, s.n_total)
    assert not np.isnan(i).any()
    assert s.n_total == sum(sizes) and s.dropped == 0
    b = load_blocks(tmp_path)
    assert list(b["dropped_raw"]) == [0, 14, 0, 14]
    assert not b["dropped_cum"].any() and not b["gap_inserted"].any()
    assert not b["counter_reset"].any()
    assert s.raw_dropped_nonzero_blocks == 2
    s.close()


def test_reopen_restores_raw_dropped_count(tmp_path):
    s = SampleStore(tmp_path)
    for k, raw in enumerate([0, 14, 14, 0]):
        s.append(np.ones(2), np.ones(2), host_ns=k, dropped_raw=raw)
    s.close()
    s2 = SampleStore(tmp_path)
    assert s2.raw_dropped_nonzero_blocks == 2 and s2.dropped == 0
    s2.append(np.ones(2), np.ones(2), host_ns=9, dropped_raw=14)
    assert s2.raw_dropped_nonzero_blocks == 3
    s2.close()
