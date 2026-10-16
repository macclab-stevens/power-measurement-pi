# tests/test_syncpulse.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eventlog import EventLog
from syncpulse import PATTERN_MS, emit


def test_default_pattern_is_the_spec_pattern():
    assert PATTERN_MS == ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))


def test_emit_logs_alternating_edges_with_pattern_spacing():
    ev = EventLog()
    pattern = ((1, 40), (0, 30), (1, 60))
    emit(ev, "t0", pattern=pattern, guard_ms=20, ncores=2)
    rows = ev.rows()
    assert rows[0]["kind"] == "marker" and rows[0]["name"] == "sync:t0"
    edges = [r for r in rows if r["kind"].startswith("pulse_")]
    assert [e["kind"] for e in edges] == ["pulse_on", "pulse_off", "pulse_on", "pulse_off"]
    assert all(e["name"] == "t0" for e in edges)
    gaps_ms = [(b["t_real_ns"] - a["t_real_ns"]) / 1e6 for a, b in zip(edges, edges[1:])]
    for got, want in zip(gaps_ms, (40, 30, 60)):
        assert abs(got - want) < 15, (gaps_ms, want)
