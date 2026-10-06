import json
import threading
import time
import urllib.request

import numpy as np

from recorder import make_server, window_stats
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
    s.append(np.full(100, 0.4), np.full(100, 5.0), host_ns=t0 + 99 * 200_000, dropped_raw=0)
    s.append(np.full(100, 0.4), np.full(100, 5.0), host_ns=t0 + 299 * 200_000, dropped_raw=100)
    w = window_stats(s, t0, t0 + 300 * 200_000)
    assert w["count"] == 200 and abs(w["mean_P"] - 2.0) < 1e-6
    s.close()


def test_http_status_and_window(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    srv = make_server(s, lambda: {"recording": True, "sample_hz": 5000.0}, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        st = json.loads(urllib.request.urlopen(base + "/status").read())
        assert st["recording"] is True and st["n_total"] == 5000
        w = json.loads(urllib.request.urlopen(
            f"{base}/window?t0_ns={t0}&t1_ns={t0 + 200_000_000}").read())
        assert abs(w["mean_P"] - 2.0) < 1e-6
    finally:
        srv.shutdown()
        srv.server_close()
        s.close()
