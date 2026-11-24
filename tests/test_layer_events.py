# tests/test_layer_events.py
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn

from eventlog import EventLog
from layers import LayerProfiler, NullProfiler, param_bytes


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


def test_deepcopied_module_logs_only_while_sink_set():
    model = nn.Sequential(nn.ReLU())
    prof = LayerProfiler(model)
    log = EventLog()
    prof.events = log
    clone = copy.deepcopy(model[0])
    with torch.no_grad():
        clone(torch.zeros(2))
    n = len(log.rows())
    assert n > 0                       # hooks survive deepcopy and still point at prof
    prof.events = None
    with torch.no_grad():
        clone(torch.zeros(2))
    assert len(log.rows()) == n


def test_param_bytes_is_dtype_aware():
    fp32 = nn.Linear(16, 8)
    fp16 = nn.Linear(16, 8).half()
    assert param_bytes(fp32) == (16 * 8 + 8) * 4
    assert param_bytes(fp16) * 2 == param_bytes(fp32)
    model = nn.Sequential(nn.Linear(16, 8).half())
    prof = LayerProfiler(model)
    assert prof.param_bytes("0") == param_bytes(model[0]) == prof.params("0") * 2


def test_aggregated_bytes_moved_counts_fp16_weights_at_two_bytes():
    def bm(dtype):
        m = nn.Sequential(nn.Linear(16, 8).to(dtype))
        prof = LayerProfiler(m)
        with torch.no_grad():
            m(torch.zeros(1, 16, dtype=dtype))
        return next(r for r in prof.aggregated() if r["path"] == "0")["bytes_moved"]
    # in (16) + out (8) + weights (16*8+8): everything is half-size in fp16
    assert bm(torch.float32) == (16 + 8 + 136) * 4
    assert bm(torch.float16) == (16 + 8 + 136) * 2


def test_null_profiler_registers_no_hooks_but_writes_labels():
    model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()))
    prof = NullProfiler(model)
    assert prof._modules == {} and prof.exec_order == []
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    prof.set_mode("forward")          # no sink yet: must not raise
    prof.set_frame(1)
    prof.events = EventLog()
    prof.set_mode("serve")
    prof.set_frame(7)
    assert (prof.mode, prof.frame) == ("serve", 7)
    assert prof.events.labels["mode"] == "serve" and prof.events.labels["iter"] == 7
    with torch.no_grad():
        model(torch.zeros(1, 3, 8, 8))
    assert prof.events.rows() == [] or all(r["kind"] not in ("begin", "end")
                                          for r in prof.events.rows())
    assert prof.aggregated(exclude_frame=0) == [] and prof.leaf_configs() == {}
