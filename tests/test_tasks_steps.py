# tests/test_tasks_steps.py
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import torch

from eventlog import EventLog
from layers import LayerProfiler
from tasks import TRAIN_STEPS, make_task


def _opt(**kw):
    base = dict(imgsz=64, conf=0.25, check_image=None, live_camera=False, seq=16,
                batch=2, optimizer="sgd")
    base.update(kw)
    return SimpleNamespace(**base)


def _steps(ev):
    return [(r["kind"], r["name"]) for r in ev.rows() if r["kind"].startswith("step_")]


def _flat(names):
    out = []
    for n in names:
        out += [("step_begin", n), ("step_end", n)]
    return out


def test_image_classification_steps():
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt())
    ev = EventLog()
    task.step = ev.step
    out = task.run_one("forward")
    assert out.shape == (1, 1000)
    assert _steps(ev) == _flat(["preprocess", "forward", "postprocess"])


def test_image_preprocess_letterbox_shape_for_detection_layout():
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt(imgsz=96))
    task._is_detection = True                       # exercise the letterbox path
    x = task.preprocess(task._source_frame())
    assert x.shape == (1, 3, 96, 96) and x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert torch.allclose(x[0, :, 0, 0], torch.full((3,), 114 / 255.0))   # pad row (4:3 frame)


def test_lm_steps_both_modes():
    task = make_task("lm", "minigpt-explicit", _opt())
    ev = EventLog()
    task.step = ev.step
    task.run_one("prefill")
    task.run_one("decode")
    assert _steps(ev) == _flat(["preprocess", "forward", "postprocess"]) * 2


def test_train_image_steps_order_and_params_move():
    task = make_task("train", "torchvision:mobilenet_v3_small", _opt(imgsz=32))
    ev = EventLog()
    task.step = ev.step
    before = [p.detach().clone() for p in task.module.parameters()]
    task.run_one("train")
    assert _steps(ev) == _flat(TRAIN_STEPS)
    assert any(not torch.equal(a, b) for a, b in zip(before, task.module.parameters()))
    assert task.oracle()


def test_train_layer_events_only_inside_forward_step():
    task = make_task("train", "torchvision:resnet18", _opt(imgsz=32, optimizer="adam"))
    prof = LayerProfiler(task.module)
    ev = EventLog()
    prof.events = ev
    task.step = ev.step
    task.run_one("train")
    cur, seen = None, set()
    for r in ev.rows():
        if r["kind"] == "step_begin":
            cur = r["name"]
        elif r["kind"] == "step_end":
            cur = None
        elif r["kind"] == "begin":
            seen.add(cur)
    assert seen == {"forward"}


def test_train_lm_adam_loss_decreases():
    task = make_task("train", "minigpt-explicit", _opt(optimizer="adam", batch=4))
    for _ in range(24):
        task.run_one("train")
    assert task.oracle()
    assert sum(task.losses[-4:]) / 4 < task.losses[0]


def test_train_rejects_batch_one_and_unknown_optimizer():
    with pytest.raises(ValueError, match="batch"):
        make_task("train", "torchvision:resnet18", _opt(batch=1))
    with pytest.raises(ValueError, match="optimizer"):
        make_task("train", "minigpt-explicit", _opt(optimizer="lion"))


def test_topk_bounded_by_number_of_classes():
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt())
    vals, idx = task.postprocess(torch.randn(1, 2))
    assert idx.shape == (1, 2)


def test_live_frames_use_shared_preprocess(monkeypatch):
    import numpy as np
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt(imgsz=96, live_camera=True))
    frames = iter([np.zeros((480, 640, 3), dtype=np.uint8)] * 3)
    monkeypatch.setattr(task, "_camera_frame", lambda: next(frames))
    seen = []
    real = task.preprocess
    monkeypatch.setattr(task, "preprocess", lambda f: seen.append(f.shape) or real(f))
    ev = EventLog()
    task.step = ev.step
    task.run_one("forward")
    assert seen == [(480, 640, 3)]
    assert _steps(ev) == _flat(["camera_capture", "preprocess", "forward", "postprocess"])
