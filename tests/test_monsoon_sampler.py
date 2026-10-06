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
    run_dir = "/data/rec1"
    serial = 36517
    n_total = 10
    window_count = 500


def _server(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/status":
                body = {"recording": fake.recording, "sample_hz": 4999.0, "serial": fake.serial,
                        "vout_V": 5.2, "dropped": 0, "n_total": fake.n_total,
                        "run_dir": fake.run_dir}
            else:
                fake.last_query = {k: int(v[0]) for k, v in parse_qs(u.query).items()}
                done = fake.incomplete_calls <= 0
                fake.incomplete_calls -= 1
                body = {"count": fake.window_count, "mean_P": 3.25, "max_P": 4.0, "complete": done}
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
    assert w == {"count": fake.window_count, "mean_P": 3.25, "max_P": 4.0}
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


def test_window_raises_when_never_complete(fake_recorder):
    fake, url = fake_recorder
    fake.incomplete_calls = 10**9
    s = MonsoonSampler(url, timeout_s=0.3)
    _no_ctx(s)
    s.start()
    with pytest.raises(RuntimeError, match="still incomplete"):
        s.window(0.0, 10.0)
    s.stop()


def test_window_retries_transient_errors_then_raises():
    s = MonsoonSampler("http://127.0.0.1:9", timeout_s=0.3)
    s._t0_ns = 0
    with pytest.raises(RuntimeError, match="recorder window query failed"):
        s.window(0.0, 10.0)


def test_window_raises_on_short_count(fake_recorder):
    fake, url = fake_recorder
    fake.window_count = 500          # 1 s window at 5 kHz expects 5000
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    with pytest.raises(RuntimeError, match=r"500/5000 samples \(<95%\)"):
        s.window(0.0, 1000.0)
    s.stop()


def test_continuity_ok(fake_recorder):
    fake, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    assert s.status_start["run_dir"] == "/data/rec1"
    fake.n_total = 20
    assert s.check_recorder_continuity() == []
    assert s.status_end["n_total"] == 20
    s.stop()


def test_continuity_detects_restart(fake_recorder):
    fake, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    fake.run_dir = "/data/rec2"
    fake.serial = 99
    fake.n_total = 3
    fake.recording = False
    errs = " | ".join(s.check_recorder_continuity())
    for word in ("run_dir", "serial", "n_total", "recording"):
        assert word in errs
    s.stop()


def test_continuity_fetch_failure_is_error_not_raise(fake_recorder):
    fake, url = fake_recorder
    s = MonsoonSampler(url, timeout_s=0.3)
    _no_ctx(s)
    s.start()
    s.url = "http://127.0.0.1:9"
    errs = s.check_recorder_continuity()
    assert len(errs) == 1 and "unreachable" in errs[0] and s.status_end is None
    s.stop()


def test_ctx_reads_are_bracketed_by_events(fake_recorder):
    _, url = fake_recorder
    s = MonsoonSampler(url)
    def fake_read():
        s._stop.set()                     # one iteration only
        return {"t_ms": 1.0, "temp_C": 40.0}
    s._ctx_read = fake_read
    s._ctx_loop()
    kinds = [(r["kind"], r["name"], r["depth"]) for r in s.events.rows()]
    assert kinds == [("ctx_begin", "ctx", 0), ("ctx_end", "ctx", 0)]
    assert s.context() == [{"t_ms": 1.0, "temp_C": 40.0}]


def test_sync_pulse_pauses_ctx_reads(fake_recorder, monkeypatch):
    import powersampler
    _, url = fake_recorder
    s = MonsoonSampler(url)
    seen = []
    monkeypatch.setattr(powersampler.syncpulse, "emit",
                        lambda ev, tag: seen.append((tag, s.ctx_pause.is_set())))
    s.sync_pulse("start")
    assert seen == [("start", True)]
    assert not s.ctx_pause.is_set()
