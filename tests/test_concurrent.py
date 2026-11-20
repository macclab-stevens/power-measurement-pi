# tests/test_concurrent.py -- run.py --concurrent: second workload thread (src/coload.py), flags,
# tagged events, per-layer events off, run_meta block.
import contextlib
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import torch
import torch.nn as nn

import coload
import run
from eventlog import EventLog
from layers import LayerProfiler
from tasks import make_task


def _err(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        run.parse_args(list(argv))
    assert e.value.code == 2
    return capsys.readouterr().err


# ---- flag -----------------------------------------------------------------------------------
def test_concurrent_flag_default_is_v1():
    a = run.parse_args([])
    assert a.concurrent is None and run.dataset_version(a) == 1


@pytest.mark.parametrize("value", ["yolo11n.pt", "yolov8s.pt", "spinner"])
def test_concurrent_marks_dataset_v2(value):
    a = run.parse_args(["--concurrent", value])
    assert a.concurrent == value and run.dataset_version(a) == 2


def test_concurrent_rejects_unknown_value_and_incompatible_flags(capsys):
    assert "<model>.pt" in _err(capsys, "--concurrent", "resnet")
    assert "--task image" in _err(capsys, "--task", "lm", "--model", "minigpt",
                                  "--concurrent", "spinner")
    assert "--soak-minutes" in _err(capsys, "--concurrent", "spinner", "--soak-minutes", "1")
    assert "--live-camera" in _err(capsys, "--concurrent", "spinner", "--live-camera")
    assert "--quant" in _err(capsys, "--concurrent", "spinner", "--quant", "fp16")
    run.parse_args(["--concurrent", "spinner", "--serve", "10"])   # serve runs after, alone


# ---- thread split -----------------------------------------------------------------------------
def test_thread_split_policy():
    assert coload.thread_split("yolo11n.pt", 4) == {"m0": 2, "m1": 2}
    assert coload.thread_split("yolo11n.pt", 5) == {"m0": 2, "m1": 3}
    assert coload.thread_split("yolo11n.pt", 1) == {"m0": 1, "m1": 1}
    assert coload.thread_split("spinner", 4) == {"m0": 3, "spin": 1}
    assert coload.thread_split("spinner", 1) == {"m0": 1, "spin": 1}
    assert coload.worker_tag("spinner") == "spin" and coload.worker_tag("yolo11n.pt") == "m1"


# ---- worker thread ----------------------------------------------------------------------------
@pytest.fixture
def small_spin(monkeypatch):
    monkeypatch.setattr(coload, "SPIN_DIM", 16)
    monkeypatch.setattr(coload, "SPIN_MM_PER_STEP", 2)


def _steps(ev, tag):
    return [(r["kind"], r["name"]) for r in ev.rows() if r["tag"] == tag]


def test_spinner_lifecycle_warms_up_idles_then_loops_until_stopped(small_spin):
    ev = EventLog()
    w = coload.SecondWorkload("spin", 1, ev.step_for("spin"))
    w.start()
    w.wait_warm()
    n_warm = len(ev.rows())
    assert n_warm == 2 * (1 + coload.SPIN_WARMUP_STEPS)   # first call + warm-up, begin/end each
    assert w.first_call_ms is not None and w.iterations == 0 and w.alive()
    threading.Event().wait(0.05)
    assert len(ev.rows()) == n_warm and w.iterations == 0      # idles until released
    w.release()
    while w.iterations < 3:
        threading.Event().wait(0.005)
    w.stop()
    assert not w.alive() and w.iterations == len(w.lat_ms) >= 3
    assert {n for _, n in _steps(ev, "spin")} == {"spin"}
    assert {r["tag"] for r in ev.rows()} == {"spin"}
    w.check()
    w.stop()   # idempotent


def test_stop_before_release_or_start_is_safe(small_spin):
    w = coload.SecondWorkload("spin", 1, EventLog().step_for("spin"))
    w.stop()                                   # never started
    w2 = coload.SecondWorkload("spin", 1, EventLog().step_for("spin"))
    w2.start()
    w2.wait_warm()
    w2.stop()                                  # warm but never released
    assert not w2.alive() and w2.iterations == 0


def test_model_worker_runs_its_task_with_tagged_steps():
    opt = SimpleNamespace(imgsz=64, conf=0.25, check_image=None, live_camera=False, seq=16,
                          batch=1, optimizer="sgd")
    task = make_task("image", "torchvision:mobilenet_v3_small", opt)
    ev = EventLog()
    w = coload.SecondWorkload("m1", 1, ev.step_for("m1"), task=task, warmup=1)
    w.start()
    w.wait_warm()
    w.release()
    while w.iterations < 2:
        threading.Event().wait(0.01)
    w.stop()
    names = [n for k, n in _steps(ev, "m1") if k == "step_begin"]
    assert names[:3] == ["preprocess", "forward", "postprocess"]
    assert {r["tag"] for r in ev.rows()} == {"m1"}
    assert w.iterations == len(w.lat_ms) >= 2


def test_torch_thread_count_is_per_thread(small_spin):
    """The worker sets its own torch thread count without touching the main thread's."""
    before = torch.get_num_threads()
    seen = {}

    class Probe(coload.SecondWorkload):
        def _one(self):
            seen["n"] = torch.get_num_threads()
            super()._one()
    w = Probe("spin", 1, EventLog().step_for("spin"))
    w.start()
    w.wait_warm()
    w.stop()
    assert seen["n"] == 1 and torch.get_num_threads() == before


class _Boom:
    step = None

    def run_one(self, mode):
        raise ValueError("boom")


def test_worker_failure_surfaces_in_the_main_thread():
    w = coload.SecondWorkload("m1", 1, EventLog().step_for("m1"), task=_Boom(), warmup=0)
    w.start()
    with pytest.raises(RuntimeError, match="concurrent worker 'm1' failed: ValueError"):
        w.wait_warm()
    with pytest.raises(RuntimeError, match="boom"):
        w.check()
    w.stop()


def test_wait_warm_calls_the_thermal_check_while_waiting():
    gate = threading.Event()

    class Slow:
        step = None

        def run_one(self, mode):
            gate.wait(5)
    w = coload.SecondWorkload("m1", 1, EventLog().step_for("m1"), task=Slow(), warmup=0)
    w.start()
    calls = []

    def check():
        calls.append(1)
        gate.set()                              # let the worker finish after the first poll
    w.wait_warm(check, poll_s=0.01)
    assert calls
    w.stop()


def test_stop_raises_when_the_worker_will_not_stop_but_not_on_the_abort_path():
    gate = threading.Event()
    state = {"n": 0}

    class Hang:
        step = None

        def run_one(self, mode):
            state["n"] += 1
            if state["n"] > 1:               # first call passes (warm-up), then block when looping
                gate.wait(10)
    w = coload.SecondWorkload("m1", 1, EventLog().step_for("m1"), task=Hang(), warmup=0,
                              join_timeout_s=0.05)
    w.start()
    w.wait_warm()
    w.release()
    while state["n"] < 2:
        threading.Event().wait(0.005)
    with pytest.raises(RuntimeError, match="did not stop within 0.05 s"):
        w.stop()
    assert w.alive() and w.stop_join_s >= 0.05
    w.stop(strict=False)                      # abort/finally path: no raise, caller logs alive()
    assert w.alive()
    gate.set()
    w.stop()                                  # now it joins cleanly
    assert not w.alive()


def test_stop_records_join_time_and_default_timeout(small_spin):
    assert coload.JOIN_TIMEOUT_S == 30.0
    w = coload.SecondWorkload("spin", 1, EventLog().step_for("spin"))
    assert w.stop_join_s is None
    w.start()
    w.wait_warm()
    w.release()
    w.stop()
    assert w.stop_join_s is not None and w.stop_join_s < 5


def test_end_concurrent_marks_the_drain_before_stopping_the_worker():
    order = []

    class W:
        tag = "m1"

        def stop(self):
            order.append("stop")

        def check(self):
            order.append("check")
    run.end_concurrent(W(), lambda name: order.append(f"marker:{name}"))
    assert order == ["marker:concurrent_drain", "stop", "check"]

    class Hung(W):
        def stop(self):
            raise RuntimeError("did not stop")
    with pytest.raises(RuntimeError, match="did not stop"):
        run.end_concurrent(Hung(), lambda name: None)


# ---- measured loop hook ------------------------------------------------------------------------
class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


class _Task:
    def __init__(self):
        self.calls = 0

    @contextlib.contextmanager
    def step(self, name):
        yield

    def run_one(self, mode):
        self.calls += 1
        torch.ones(2) + 1


class _Prof:
    def set_frame(self, f):
        pass


def test_on_first_frame_runs_once_right_after_frame_zero():
    task, seen = _Task(), []
    run.measured_loop(task, _Prof(), "forward", 4, None, lambda: None, clock=_Clock(),
                      on_first_frame=lambda: seen.append(task.calls))
    assert seen == [1]


# ---- per-layer events off, labels kept ----------------------------------------------------------
def test_layer_events_off_keeps_labels_and_drops_begin_end():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
    prof = LayerProfiler(model)
    prof.events = EventLog()
    prof.layer_events = False
    prof.set_mode("forward")
    prof.set_frame(3)
    with torch.no_grad():
        model(torch.zeros(1, 3, 8, 8))
    assert prof.events.rows() == []
    assert prof.events.labels == {"mode": "forward", "iter": 3}
    assert len(prof.times["0"]) == 1                     # timings are still recorded
    prof.layer_events = True
    with torch.no_grad():
        model(torch.zeros(1, 3, 8, 8))
    assert {r["kind"] for r in prof.events.rows()} == {"begin", "end"}


# ---- run_meta block ----------------------------------------------------------------------------
def test_concurrent_block_records_split_counts_and_latency():
    w = SimpleNamespace(tag="m1", iterations=40, lat_ms=[100.0, 110.0, 120.0], first_call_ms=900.0,
                        stop_join_s=0.31)
    a = run.parse_args(["--concurrent", "yolo11n.pt"])
    b = run.concurrent_block(a, {"m0": 2, "m1": 2}, w, {"forward": 30}, {"mean": 98.0}, 1.25,
                             serve_concurrent=False)
    assert b["mode"] == "model" and b["model"] == "yolo11n.pt" and b["tags"] == ["m0", "m1"]
    assert b["threads_split"] == {"m0": 2, "m1": 2} and b["sync_envelopes"] == 1
    assert b["layer_events"] is False and b["serve_concurrent"] is False
    assert b["stop_join_s"] == 0.31
    assert b["iterations"] == {"m0": 30, "m1": 40} and b["overlapped_primary_frames"] == 29
    assert b["latency_ms"]["m0"] == {"mean": 98.0}
    assert b["latency_ms"]["m1"]["n"] == 3 and b["first_call_ms"] == {"m1": 900.0}
    assert b["load_weights_s"] == {"m1": 1.25} and b["spinner"] is None
    s = run.concurrent_block(run.parse_args(["--concurrent", "spinner"]), {"m0": 3, "spin": 1},
                             SimpleNamespace(tag="spin", iterations=5, lat_ms=[], first_call_ms=1.0,
                                             stop_join_s=None),
                             {"forward": 30}, None, None, serve_concurrent=False)
    assert s["mode"] == "spinner" and s["model"] is None and s["spinner"]["dim"] == coload.SPIN_DIM
    assert s["latency_ms"]["spin"] is None and s["load_weights_s"] == {"spin": None}
