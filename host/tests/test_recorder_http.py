import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from recorder import make_server, validate_args, window_stats
from store import SampleStore


def _store_with_known_power(tmp_path, t0_ns):
    s = SampleStore(tmp_path)
    for k in range(100):  # 100 blocks x 50 samples = 1 s at 5 kHz, P = 2 W then 4 W
        I = np.full(50, 0.4 if k < 50 else 0.8)
        last_idx = k * 50 + 49
        s.append(I, np.full(50, 5.0), host_ns=int(t0_ns + last_idx * 200_000), dropped_raw=0)
    return s


def test_window_stats_by_wall_clock(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    w = window_stats(s, t0 + 600_000_000, t0 + 800_000_000)  # second half only
    assert w["complete"] is True
    assert abs(w["mean_P"] - 4.0) < 1e-6 and w["count"] == 1000
    assert window_stats(s, t0 + 600_000_000, t0 + 5_000_000_000)["complete"] is False
    s.close()


def test_window_ignores_nan(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = SampleStore(tmp_path)
    I = np.full(100, 0.4)
    s.append(I, np.full(100, 5.0), host_ns=t0 + 99 * 200_000, dropped_raw=0)
    I2 = np.full(100, 0.4)
    I2[10:20] = np.nan
    s.append(I2, np.full(100, 5.0), host_ns=t0 + 199 * 200_000, dropped_raw=0)
    w = window_stats(s, t0, t0 + 200 * 200_000)
    assert w["count"] == 190 and abs(w["mean_P"] - 2.0) < 1e-6
    s.close()


def test_current_limit_bounds():
    validate_args(5.25, 6.0)
    validate_args(5.25, 8.0)
    for bad in (0, -1.0, 8.01):
        with pytest.raises(SystemExit):
            validate_args(5.25, bad)
    with pytest.raises(SystemExit):
        validate_args(9.0, 6.0)


def test_http_status_and_window(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    srv = make_server(s, lambda: {"recording": True, "sample_hz": 5000.0}, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        st = json.loads(urllib.request.urlopen(base + "/status").read())
        assert st["recording"] is True and st["n_total"] == 5000
        assert "disk_free_gb" in st and st["disk_free_gb"] > 0
        w = json.loads(urllib.request.urlopen(
            f"{base}/window?t0_ns={t0}&t1_ns={t0 + 200_000_000}").read())
        assert abs(w["mean_P"] - 2.0) < 1e-6
    finally:
        srv.shutdown()
        srv.server_close()
        s.close()


def test_window_rejects_huge_or_inverted_span(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    srv = make_server(s, lambda: {"recording": True}, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for a, b in ((t0, t0 + 601 * 10**9), (t0 + 10**9, t0), (t0, t0)):
            try:
                urllib.request.urlopen(f"{base}/window?t0_ns={a}&t1_ns={b}")
                raise AssertionError("expected HTTP 400")
            except urllib.error.HTTPError as e:
                assert e.code == 400 and "error" in json.loads(e.read())
        ok = json.loads(urllib.request.urlopen(
            f"{base}/window?t0_ns={t0}&t1_ns={t0 + 600 * 10**9}").read())
        assert "count" in ok
    finally:
        srv.shutdown()
        srv.server_close()
        s.close()


def test_clock_fit_is_cached_while_blocks_arrive(tmp_path, monkeypatch):
    import recorder
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    fits = []
    real = recorder.fit_sample_clock
    monkeypatch.setattr(recorder, "fit_sample_clock", lambda *a: fits.append(1) or real(*a))
    now = [100.0]
    monkeypatch.setattr(recorder.time, "monotonic", lambda: now[0])
    recorder._clock(s)
    for k in range(100, 110):                    # new blocks keep arriving within 1 s
        last_idx = k * 50 + 49
        s.append(np.full(50, 0.4), np.full(50, 5.0), host_ns=int(t0 + last_idx * 200_000),
                 dropped_raw=0)
        now[0] += 0.05
        recorder._clock(s)
    assert len(fits) == 1
    now[0] += 1.0                                # cache expired and new data -> refit once
    recorder._clock(s)
    assert len(fits) == 2
    now[0] += 2.0                                # expired but no new blocks -> keep the fit
    recorder._clock(s)
    assert len(fits) == 2
    s.close()
