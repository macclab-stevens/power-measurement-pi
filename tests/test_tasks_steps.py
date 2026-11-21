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
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt(batch=1))
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
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt(imgsz=96, live_camera=True, batch=1))
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


# ---- dataset v2 inputs: batch stacking, stream frames, --input-image ---------------------------
def _write_jpgs(d, n=3):
    import cv2
    import numpy as np
    paths = []
    for i in range(n):
        img = np.full((24, 32, 3), 40 * (i + 1), dtype=np.uint8)
        p = os.path.join(str(d), f"{i:03d}.jpg")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def test_image_batch_stacks_copies_and_runs_whole_batch():
    task = make_task("image", "torchvision:mobilenet_v3_small", _opt(imgsz=64, batch=3))
    ev = EventLog()
    task.step = ev.step
    seen = []
    real = task._module.forward
    task._module.forward = lambda x: seen.append(tuple(x.shape)) or real(x)
    out = task.run_one("forward")
    assert seen == [(3, 3, 64, 64)] and out.shape == (3, 1000)
    assert _steps(ev) == _flat(["preprocess", "forward", "postprocess"])


def test_image_batch_none_or_one_is_single_frame():
    for b in (None, 1):
        task = make_task("image", "torchvision:mobilenet_v3_small", _opt(imgsz=64, batch=b))
        assert task.run_one("forward").shape == (1, 1000)


def test_load_stream_frames_sorted_decoded_once(tmp_path):
    from tasks import load_stream_frames
    paths = _write_jpgs(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored")
    frames = load_stream_frames(str(tmp_path))
    assert [p for p, _ in frames] == sorted(paths)
    assert all(f.shape == (24, 32, 3) and f.dtype.name == "uint8" for _, f in frames)


def test_load_stream_frames_errors(tmp_path):
    from tasks import load_stream_frames
    with pytest.raises(ValueError, match="no \\*.jpg"):
        load_stream_frames(str(tmp_path))
    with pytest.raises(ValueError, match="not a directory"):
        load_stream_frames(str(tmp_path / "missing"))


def test_stream_batch_uses_consecutive_images_and_cycles(tmp_path):
    _write_jpgs(tmp_path, 3)
    task = make_task("image", "torchvision:mobilenet_v3_small",
                     _opt(imgsz=64, batch=2, frames_source="stream", stream_dir=str(tmp_path)))
    xs = [task._batch_input() for _ in range(3)]
    assert all(x.shape == (2, 3, 64, 64) for x in xs)
    pre = [task.preprocess(f) for _, f in task._stream]
    # batches: [0,1] [2,0] [1,2]
    for x, idx in zip(xs, [(0, 1), (2, 0), (1, 2)]):
        assert torch.equal(x, torch.cat([pre[i] for i in idx]))


def test_input_identity(tmp_path):
    from tasks import sha256_file
    paths = _write_jpgs(tmp_path, 3)
    stream = make_task("image", "torchvision:mobilenet_v3_small",
                       _opt(frames_source="stream", stream_dir=str(tmp_path)))
    ident = stream.input_identity()
    assert ident["source"] == "stream"
    assert ident["files"] == [{"path": p, "sha256": sha256_file(p)} for p in sorted(paths)]
    rep = make_task("image", "torchvision:mobilenet_v3_small", _opt(input_image=paths[1]))
    assert rep.input_identity() == {"source": "repeat",
                                    "files": [{"path": paths[1], "sha256": sha256_file(paths[1])}]}
    assert make_task("image", "torchvision:mobilenet_v3_small", _opt()).input_identity() \
        == {"source": "repeat", "files": []}   # synthetic frame: no file
    live = make_task("image", "torchvision:mobilenet_v3_small", _opt(live_camera=True))
    assert live.input_identity() == {"source": "camera", "files": []}


def test_input_image_feeds_model_and_check_image_stays_the_oracle(tmp_path):
    paths = _write_jpgs(tmp_path, 2)
    task = make_task("image", "torchvision:mobilenet_v3_small",
                     _opt(input_image=paths[1], check_image=paths[0]))
    from tasks import read_rgb
    assert (task._source_frame() == read_rgb(paths[1])).all()
    assert task._check_image == paths[0]
    assert task.input_identity()["files"][0]["path"] == paths[1]


def test_unreadable_input_image_is_an_error(tmp_path):
    task = make_task("image", "torchvision:mobilenet_v3_small",
                     _opt(input_image=str(tmp_path / "nope.jpg")))
    with pytest.raises(ValueError, match="cannot read"):
        task._source_frame()


def test_live_camera_batch_captures_b_frames(monkeypatch):
    import numpy as np
    task = make_task("image", "torchvision:mobilenet_v3_small",
                     _opt(imgsz=64, live_camera=True, batch=3))
    n = []
    monkeypatch.setattr(task, "_camera_frame",
                        lambda: n.append(1) or np.zeros((48, 64, 3), dtype=np.uint8))
    assert task._batch_input().shape == (3, 3, 64, 64) and len(n) == 3
