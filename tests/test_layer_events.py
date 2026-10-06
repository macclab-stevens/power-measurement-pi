# tests/test_layer_events.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn

from eventlog import EventLog
from layers import LayerProfiler


def test_layer_begin_end_nest_and_carry_labels():
    model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()))
    prof = LayerProfiler(model)
    prof.events = EventLog()
    prof.set_mode("forward")
    prof.set_frame(2)
    with torch.no_grad():
        model(torch.zeros(1, 3, 8, 8))
    rows = [r for r in prof.events.rows() if r["kind"] in ("begin", "end")]
    seq = [(r["kind"], r["layer_path"], r["depth"]) for r in rows]
    assert seq == [("begin", "0", 1), ("begin", "0.0", 2), ("end", "0.0", 2),
                   ("begin", "0.1", 2), ("end", "0.1", 2), ("end", "0", 1)]
    assert {(r["mode"], r["iter"]) for r in rows} == {("forward", 2)}


def test_no_events_when_sink_absent():
    model = nn.Sequential(nn.ReLU())
    prof = LayerProfiler(model)
    with torch.no_grad():
        model(torch.zeros(2))
    assert prof.events is None
