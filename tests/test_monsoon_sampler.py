# tests/test_monsoon_sampler.py
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from powersampler import MonsoonSampler, PowerSampler, make_sampler


class Fake:
    recording = True
    incomplete_calls = 0
    last_query = None


def _server(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/status":
                body = {"recording": fake.recording, "sample_hz": 4999.0, "serial": 36517,
                        "vout_V": 5.2, "dropped": 0, "n_total": 10}
            else:
                fake.last_query = {k: int(v[0]) for k, v in parse_qs(u.query).items()}
                done = fake.incomplete_calls <= 0
                fake.incomplete_calls -= 1
                body = {"count": 500, "mean_P": 3.25, "max_P": 4.0, "complete": done}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def fake_recorder():
    fake = Fake()
    srv, url = _server(fake)
    yield fake, url
    srv.shutdown()


def _no_ctx(s):  # don't call vcgencmd on the NUC
    s._ctx_thread = threading.Thread(target=lambda: None)


def test_make_sampler_kinds():
    assert type(make_sampler("pmic", None)) is PowerSampler
    assert isinstance(make_sampler("monsoon", "http://x:1"), MonsoonSampler)
    with pytest.raises(ValueError):
        make_sampler("monsoon", None)


def test_monsoon_sampler_start_fails_fast():
    s = MonsoonSampler("http://127.0.0.1:9", timeout_s=0.5)
    _no_ctx(s)
    with pytest.raises(RuntimeError, match="recorder"):
        s.start()


def test_window_converts_ms_to_wall_ns_and_returns_stats(fake_recorder):
    fake, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    w = s.window(100.0, 200.0)
    assert w == {"count": 500, "mean_P": 3.25, "max_P": 4.0}
    q = fake.last_query
    assert q["t1_ns"] - q["t0_ns"] == 100_000_000
    assert q["t0_ns"] == s._t0_ns + 100_000_000
    assert s.achieved_hz() == 4999.0
    assert s.rows() == []
    s.stop()


def test_window_waits_until_complete(fake_recorder):
    fake, url = fake_recorder
    fake.incomplete_calls = 3
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    assert s.window(0.0, 10.0)["count"] == 500
    assert fake.incomplete_calls < 0
    s.stop()


def test_marker_goes_to_event_log_with_window_ids(fake_recorder):
    _, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    s.marker("forward")
    s.marker("gap")
    rows = s.events.rows()
    assert [r["name"] for r in rows] == ["forward", "gap"]
    assert s._window_id == 2 and s._phase == "gap"
    s.stop()
