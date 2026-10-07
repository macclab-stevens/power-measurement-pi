# tests/test_eventlog.py
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eventlog import EventLog, FIELDS


def test_rows_have_wall_and_mono_ns_and_are_ordered():
    ev = EventLog()
    before = time.time_ns()
    ev.marker("warmup")
    ev.marker("forward")
    after = time.time_ns()
    rows = ev.rows()
    assert [r["kind"] for r in rows] == ["marker", "marker"]
    assert before <= rows[0]["t_real_ns"] <= rows[1]["t_real_ns"] <= after
    assert rows[0]["t_mono_ns"] <= rows[1]["t_mono_ns"]


def test_marker_bumps_window_id_only_on_new_phase():
    ev = EventLog()
    ev.marker("forward")
    ev.marker("forward")
    ev.marker("gap")
    assert [r["window_id"] for r in ev.rows()] == [1, 1, 2]
    assert ev.phase == "gap"


def test_begin_end_track_nesting_depth():
    ev = EventLog()
    ev.begin("layer", layer_path="model.0")
    ev.begin("layer", layer_path="model.0.conv")
    ev.end("layer", layer_path="model.0.conv")
    ev.end("layer", layer_path="model.0")
    assert [(r["kind"], r["depth"]) for r in ev.rows()] == [
        ("begin", 1), ("begin", 2), ("end", 2), ("end", 1)]


def test_labels_and_bench_config_id_are_recorded():
    ev = EventLog()
    ev.labels["mode"] = "decode"
    ev.labels["iter"] = 3
    ev.marker("bench:abc123")
    r = ev.rows()[0]
    assert (r["mode"], r["iter"], r["config_id"]) == ("decode", 3, "abc123")


def test_write_csv_roundtrip(tmp_path):
    ev = EventLog()
    ev.marker("warmup")
    ev.edge("pulse_on", "sync_start")
    p = tmp_path / "events.csv"
    ev.write_csv(p)
    rows = list(csv.DictReader(open(p)))
    assert list(rows[0].keys()) == FIELDS
    assert rows[1]["kind"] == "pulse_on" and rows[1]["name"] == "sync_start"


def test_step_spans_nest_without_touching_layer_depth():
    ev = EventLog()
    ev.labels.update(mode="train", iter=3)
    with ev.step("forward"):
        ev.begin("layer", layer_path="0")
        ev.end("layer", layer_path="0")
    rows = ev.rows()
    assert [(r["kind"], r["name"], r["depth"]) for r in rows] == [
        ("step_begin", "forward", 0), ("begin", "layer", 1),
        ("end", "layer", 1), ("step_end", "forward", 0)]
    assert {(r["mode"], r["iter"]) for r in rows} == {("train", 3)}


def test_step_end_is_logged_when_the_body_raises():
    ev = EventLog()
    try:
        with ev.step("loss"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert [r["kind"] for r in ev.rows()] == ["step_begin", "step_end"]
