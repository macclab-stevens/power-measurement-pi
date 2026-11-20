"""Tests for tools/eval_map.py: metric wrapper, keys/merge, letterbox inverse, plan, tiny end-to-end loop."""
import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

import eval_map as em


# ---- metric wrapper ---------------------------------------------------------------------------
def box(*xyxy):
    return torch.tensor([xyxy], dtype=torch.float32)


def test_one_perfect_box_is_map_one():
    acc = em.MapAccumulator()
    acc.add_image(pred=torch.tensor([[10., 10., 50., 60., 0.9, 3.]]), gt_boxes=box(10, 10, 50, 60),
                  gt_cls=torch.tensor([3.]))
    r = acc.result()
    # ultralytics' 101-point interpolated AP tops out at 0.995 for a single perfect detection
    assert r["map50_95"] == pytest.approx(1.0, abs=0.006)
    assert r["map50"] == pytest.approx(1.0, abs=0.006)
    assert r["n_images"] == 1


def test_wrong_class_is_map_zero():
    acc = em.MapAccumulator()
    acc.add_image(pred=torch.tensor([[10., 10., 50., 60., 0.9, 4.]]), gt_boxes=box(10, 10, 50, 60),
                  gt_cls=torch.tensor([3.]))
    r = acc.result()
    assert r["map50_95"] == 0.0 and r["map50"] == 0.0


def test_loose_box_scores_between_iou_thresholds():
    # IoU 0.8 -> true positive at thresholds 0.50..0.80 (7 of 10), so mAP50-95 = 0.7, mAP50 = 1
    acc = em.MapAccumulator()
    acc.add_image(pred=torch.tensor([[0., 0., 100., 80., 0.9, 0.]]), gt_boxes=box(0, 0, 100, 100),
                  gt_cls=torch.tensor([0.]))
    r = acc.result()
    assert r["map50"] == pytest.approx(1.0, abs=0.006)
    assert r["map50_95"] == pytest.approx(0.7, abs=0.01)


def test_no_predictions_and_empty_images():
    acc = em.MapAccumulator()
    acc.add_image(pred=torch.zeros((0, 6)), gt_boxes=box(0, 0, 10, 10), gt_cls=torch.tensor([1.]))
    acc.add_image(pred=torch.tensor([[0., 0., 5., 5., 0.5, 1.]]), gt_boxes=torch.zeros((0, 4)),
                  gt_cls=torch.zeros(0))   # a false positive on a label-free image
    assert acc.result()["map50_95"] == 0.0
    assert acc.result()["n_images"] == 2


def test_no_labels_at_all_is_zero_not_a_crash():
    acc = em.MapAccumulator()
    acc.add_image(pred=torch.zeros((0, 6)), gt_boxes=torch.zeros((0, 4)), gt_cls=torch.zeros(0))
    assert acc.result()["map50_95"] == 0.0


# ---- keys, entries, merge ---------------------------------------------------------------------
PI = {"hostname": "pi", "machine": "aarch64", "torch": "2.14.0", "quantized_engine": "qnnpack"}


def test_measured_on_describes_this_process():
    import platform
    m = em.measured_on()
    assert set(m) == {"hostname", "machine", "torch", "quantized_engine"}
    assert m["machine"] == platform.machine() and m["torch"] == torch.__version__
    assert m["quantized_engine"] == torch.backends.quantized.engine


def test_artifact_entry_records_sha256_and_file_sha256_is_correct(tmp_path):
    import hashlib
    f = tmp_path / "a.pt"
    f.write_bytes(b"x" * 3_000_000)
    sha = em.file_sha256(str(f))
    assert sha == hashlib.sha256(b"x" * 3_000_000).hexdigest()
    e = em.make_entry(0.4, 0.5, 500, "abc", "2026-11-20", PI, artifact_sha256=sha)
    assert e["artifact_sha256"] == sha
    assert "artifact_sha256" not in em.make_entry(0.4, 0.5, 500, "abc", "2026-11-20", PI)


def test_already_measured_needs_aarch64_and_a_matching_artifact_hash():
    pi = em.make_entry(0.4, 0.5, 500, "abc", "2026-11-20", PI)
    nuc = {**pi, "measured_on": {**PI, "machine": "x86_64"}}
    assert em.already_measured(pi)
    assert not em.already_measured(nuc)
    assert not em.already_measured({"map50_95": 0.4})            # provisional NUC entry: no measured_on
    assert not em.already_measured(None)
    art = {**pi, "artifact_sha256": "aa"}
    assert em.already_measured(art, "aa")
    assert not em.already_measured(art, "bb")                    # artifact regenerated
    assert not em.already_measured(pi, "aa")                     # entry predates artifact hashing


def test_main_skip_existing_only_skips_entries_measured_on_aarch64(tmp_path, monkeypatch):
    table = tmp_path / "m.json"
    pi = em.make_entry(0.4, 0.5, 500, "abc", "2026-11-20", PI)
    table.write_text(json.dumps({"a@640": pi, "b@640": {**pi, "measured_on": {**PI, "machine": "x86_64"}},
                                 "c@640": {"map50_95": 0.1}}))
    jobs = [em.Job(v, f"/w/{v}.pt", 640) for v in "abc"]
    monkeypatch.setattr(em, "all_jobs", lambda root, artifacts: jobs)
    ran = []
    monkeypatch.setattr(em, "run_job", lambda job, *a, **k: ran.append((job.variant, a[-1])) or {})
    assert em.main(["--all", "--skip-existing", "--out", str(table), "--tool-version", "v9"]) == 0
    assert ran == [("b", "v9"), ("c", "v9")]       # a is a Pi entry; the NUC/provisional ones are redone


def test_key_format_matches_what_run_py_looks_up(tmp_path):
    import yolo_variants as yv
    assert em.entry_key("yolo11n", 640) == "yolo11n@640"
    entry = em.make_entry(0.4, 0.55, 500, tool_version="abc1234", measured="2026-11-19",
                          measured_on=PI)
    assert set(entry) == set(yv.MAP_REFERENCE_FIELDS)
    assert entry["measured_on"] == PI
    assert entry["dataset"] == "coco-val2017-first500"
    f = tmp_path / "m.json"
    em.merge_entry(str(f), em.entry_key("yolo11n_int8s", 480), entry)
    assert yv.lookup_map_reference(str(f), "yolo11n_int8s", 480) == entry


def test_merge_preserves_other_keys_and_overwrites_same_key(tmp_path):
    f = tmp_path / "m.json"
    assert not f.exists()
    em.merge_entry(str(f), "a@640", {"map50_95": 0.1})
    em.merge_entry(str(f), "b@320", {"map50_95": 0.2})
    em.merge_entry(str(f), "a@640", {"map50_95": 0.3})
    assert json.loads(f.read_text()) == {"a@640": {"map50_95": 0.3}, "b@320": {"map50_95": 0.2}}


def test_merge_refuses_to_clobber_a_corrupt_file(tmp_path):
    f = tmp_path / "m.json"
    f.write_text("{not json")
    with pytest.raises(ValueError):
        em.merge_entry(str(f), "a@640", {})
    assert f.read_text() == "{not json"


def _fake_git(head="abc1234\n", porcelain=""):
    def run(cmd, **k):
        out = head if "rev-parse" in cmd else porcelain
        return type("R", (), {"stdout": out, "returncode": 0})()
    return run


def test_tool_version_is_git_short_hash_or_unknown():
    assert em.tool_version(run=_fake_git()) == "abc1234"

    def boom(*a, **k):
        raise OSError("no git")
    assert em.tool_version(run=boom) == "unknown"


def test_tool_version_marks_uncommitted_tracked_changes_dirty():
    assert em.tool_version(run=_fake_git(porcelain=" M tools/eval_map.py\n")) == "abc1234-dirty"
    assert em.tool_version(run=_fake_git(porcelain="")) == "abc1234"


def test_tool_version_asks_git_for_tracked_files_only():
    seen = []

    def run(cmd, **k):
        seen.append(cmd)
        return type("R", (), {"stdout": "abc1234\n", "returncode": 0})()
    em.tool_version(run=run)
    status = next(c for c in seen if "status" in c)
    assert "--porcelain" in status and "-uno" in status      # untracked files do not make it dirty
    assert ":(exclude)data/map_reference.json" in status     # nor does the table this tool writes


# ---- letterbox shared with the Pi -------------------------------------------------------------
@pytest.mark.parametrize("h,w,S", [(480, 640, 640), (427, 640, 320), (640, 480, 960), (333, 500, 480)])
def test_letterbox_geometry_matches_image_task_preprocess(h, w, S):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    x = em.make_preprocess(S, half=False)(frame)
    assert x.shape == (1, 3, S, S)
    nw, nh, left, top = em.letterbox_geometry(h, w, S)
    canvas = (x[0].permute(1, 2, 0) * 255).round().numpy().astype(np.uint8)
    inner = canvas[top:top + nh, left:left + nw]
    assert np.array_equal(inner, cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR))
    mask = np.ones((S, S), bool)
    mask[top:top + nh, left:left + nw] = False
    assert (canvas[mask] == 114).all()


def test_unletterbox_roundtrip_and_clip():
    h, w, S = 427, 640, 320
    nw, nh, left, top = em.letterbox_geometry(h, w, S)
    orig = torch.tensor([[100., 50., 300., 200.]])
    lb = torch.stack([orig[:, 0] * nw / w + left, orig[:, 1] * nh / h + top,
                      orig[:, 2] * nw / w + left, orig[:, 3] * nh / h + top], dim=1)
    assert torch.allclose(em.unletterbox(lb, h, w, S), orig, atol=1e-3)
    out = em.unletterbox(torch.tensor([[-5., -5., S + 9., S + 9.]]), h, w, S)
    assert out.min() >= 0 and out[0, 2] <= w and out[0, 3] <= h


def test_half_preprocess_is_float16():
    assert em.make_preprocess(320, half=True)(np.zeros((100, 100, 3), np.uint8)).dtype == torch.float16


# ---- labels -----------------------------------------------------------------------------------
def test_read_labels_converts_normalised_xywh(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("3 0.5 0.5 0.2 0.4\n7 0.25 0.5 0.5 1.0\n")
    boxes, cls = em.read_labels(str(f), h=100, w=200)
    assert cls.tolist() == [3.0, 7.0]
    assert boxes[0].tolist() == pytest.approx([80., 30., 120., 70.])
    assert boxes[1].tolist() == pytest.approx([0., 0., 100., 100.])
    b, c = em.read_labels(str(tmp_path / "missing.txt"), h=10, w=10)
    assert b.shape == (0, 4) and c.shape == (0,)


# ---- the plan and artifact identity -----------------------------------------------------------
def test_all_jobs_is_the_plans_set_without_int8_dynamic(tmp_path):
    jobs = em.all_jobs("ROOT", artifacts=[])
    keys = [em.entry_key(j.variant, j.imgsz) for j in jobs]
    assert len(keys) == len(set(keys))
    for m in ("yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s", "yolov8m"):
        assert f"{m}@640" in keys
    for m in ("yolo11n", "yolo11s", "yolo11m"):
        for s in (320, 480, 960):
            assert f"{m}@{s}" in keys
    assert "yolo11n_fp16@640" in keys
    assert not any("int8d" in k for k in keys)
    assert len(keys) == 16
    assert next(j for j in jobs if j.variant == "yolo11n_fp16").quant == "fp16"
    assert next(j for j in jobs if j.variant == "yolo11s" and j.imgsz == 320).model == "ROOT/yolo11s.pt"


def _save_artifact(tmp_path, name, **fields):
    p = str(tmp_path / name)
    torch.save({"model": nn.Sequential(nn.Conv2d(3, 4, 3)), **fields}, p)
    return p


def test_artifact_job_identity_comes_from_the_artifact(tmp_path):
    p8 = _save_artifact(tmp_path, "yolo11n_int8s.pt", format="pi-power-int8s/1", base_model="yolo11n.pt")
    job = em.artifact_job(p8, "ROOT", imgsz=640)
    assert (job.variant, job.quant, job.imgsz, job.artifact) == ("yolo11n_int8s", "int8-static", 640, p8)
    pp = _save_artifact(tmp_path, "yolo11s_p30.pt", format="pi-power-pruned/1", base_model="yolo11s.pt",
                        ratio=0.3)
    job = em.artifact_job(pp, "ROOT", imgsz=640)
    assert job.variant == "yolo11s_p30" and job.prune_ratio == 0.3
    bad = _save_artifact(tmp_path, "x.pt", format="other/9", base_model="yolo11n.pt")
    with pytest.raises(ValueError):
        em.artifact_job(bad, "ROOT", imgsz=640)


def test_discover_artifacts_lists_models_dir(tmp_path):
    (tmp_path / "models").mkdir()
    for n in ("b.pt", "a.pt", "readme.txt"):
        (tmp_path / "models" / n).write_text("x")
    got = em.discover_artifacts(str(tmp_path))
    assert [g.split("/")[-1] for g in got] == ["a.pt", "b.pt"]
    assert em.discover_artifacts(str(tmp_path / "nope")) == []


# ---- the custom loop end to end on a fake detector --------------------------------------------
class FakeDetector(nn.Module):
    """Returns the ground-truth box (in letterbox space) as a (1, 4+nc, 1) ultralytics head output."""
    def __init__(self, xywh_lb, cls, nc=80):
        super().__init__()
        self.t = torch.zeros(1, 4 + nc, 1)
        self.t[0, :4, 0] = torch.tensor(xywh_lb)
        self.t[0, 4 + cls, 0] = 0.95

    def forward(self, x):
        return (self.t.to(x.dtype), [])


def test_evaluate_loop_with_a_perfect_fake_detector(tmp_path):
    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir(), lab_dir.mkdir()
    h, w, S = 60, 100, 320
    cv2.imwrite(str(img_dir / "000001.jpg"), np.full((h, w, 3), 90, np.uint8))
    (lab_dir / "000001.txt").write_text("5 0.5 0.5 0.4 0.5\n")      # box x 30..70, y 15..45
    nw, nh, left, top = em.letterbox_geometry(h, w, S)
    xywh = [(0.5 * w) * nw / w + left, (0.5 * h) * nh / h + top, 0.4 * w * nw / w, 0.5 * h * nh / h]
    res = em.evaluate(FakeDetector(xywh, 5), imgsz=S, half=False, images_dir=str(img_dir),
                      labels_dir=str(lab_dir), log=lambda *_: None)
    assert res["n_images"] == 1
    assert res["map50_95"] > 0.95 and res["map50"] > 0.99


def test_evaluate_stops_when_projected_time_exceeds_budget(tmp_path):
    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir(), lab_dir.mkdir()
    for i in range(6):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), np.zeros((20, 20, 3), np.uint8))
    clock = iter(range(0, 10000, 1000))          # every call to the clock: +1000 s
    with pytest.raises(em.TooSlow) as e:
        em.evaluate(FakeDetector([10, 10, 5, 5], 0), imgsz=64, half=False, images_dir=str(img_dir),
                    labels_dir=str(lab_dir), log=lambda *_: None, budget_s=60, probe_images=2,
                    clock=lambda: next(clock))
    assert e.value.s_per_image == pytest.approx(1000.0)
