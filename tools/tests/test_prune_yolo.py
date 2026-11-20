"""Tests for tools/prune_yolo.py and the split-free blocks in src/yolo_variants.py (tiny random YOLO11)."""
import copy

import pytest
import torch
import torch.nn as nn

import prune_yolo as py
import yolo_variants as yv

pytestmark = pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
TOL = 1e-5


def _randomise_bn(model: nn.Module, seed: int = 0) -> nn.Module:
    """Non-trivial BN statistics/affine params, so a wrong channel slice cannot pass by accident."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.weight.data = 0.5 + torch.rand(n, generator=g)
            m.bias.data = torch.randn(n, generator=g) * 0.1
            m.running_mean = torch.randn(n, generator=g) * 0.1
            m.running_var = 0.5 + torch.rand(n, generator=g)
    return model.eval()


@pytest.fixture(scope="module")
def tiny_yolo():
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    return _randomise_bn(DetectionModel("yolo11n.yaml", nc=5, verbose=False))


# ---- split-free blocks equal the originals ----------------------------------------------------
def _same(a: nn.Module, b: nn.Module, x: torch.Tensor) -> float:
    with torch.inference_mode():
        return (a(x) - b(x)).abs().max().item()


@pytest.mark.parametrize("kw", [dict(), dict(c3k=True), dict(attn=True), dict(n=3, shortcut=False)])
def test_c3k2_split_free_equals_original(kw):
    from ultralytics.nn.modules.block import C3k2
    torch.manual_seed(1)
    orig = _randomise_bn(C3k2(32, 64, **{"n": 2, **kw}))
    new = yv.C2fSplitFree.from_c2f(orig).eval()
    assert _same(orig, new, torch.randn(2, 32, 16, 16)) <= TOL
    assert new.cv1a.conv.out_channels == new.cv1b.conv.out_channels == orig.c


def test_c2f_split_free_equals_original():
    from ultralytics.nn.modules.block import C2f
    torch.manual_seed(2)
    orig = _randomise_bn(C2f(32, 48, n=2))
    assert _same(orig, yv.C2fSplitFree.from_c2f(orig).eval(), torch.randn(1, 32, 8, 8)) <= TOL


def test_c2psa_split_free_equals_original():
    from ultralytics.nn.modules.block import C2PSA
    torch.manual_seed(3)
    orig = _randomise_bn(C2PSA(64, 64, n=2))
    assert _same(orig, yv.C2PSASplitFree.from_c2psa(orig).eval(), torch.randn(1, 64, 8, 8)) <= TOL


def test_split_free_does_not_alias_or_change_the_original_weights():
    from ultralytics.nn.modules.block import C2f
    orig = _randomise_bn(C2f(16, 16))
    w = orig.cv1.conv.weight.detach().clone()
    new = yv.C2fSplitFree.from_c2f(orig)
    new.cv1a.conv.weight.data.zero_()
    assert torch.equal(orig.cv1.conv.weight, w)


def test_make_split_free_replaces_every_split_block_and_keeps_the_output(tiny_yolo):
    sf, counts = py.make_split_free(tiny_yolo)
    names = {type(m).__name__ for m in sf.modules()}
    assert not names & {"C2f", "C3k2", "C2PSA"} and {"C2fSplitFree", "C2PSASplitFree"} <= names
    assert counts == {"C3k2": 8, "C2PSA": 1}
    assert "C3k2" in {type(m).__name__ for m in tiny_yolo.modules()}     # the input is not modified
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        # box coordinates reach ~60 here: a few float32 ulps (2e-5 abs) differ with the conv blocking
        assert torch.allclose(sf(x)[0], tiny_yolo(x)[0], atol=TOL, rtol=TOL)
    assert [(l.f, l.i, l.type) for l in sf.model] == [(l.f, l.i, l.type) for l in tiny_yolo.model]


def test_block_classes_resolve_from_the_module_for_pickle_and_import_stays_light():
    for name in yv._BLOCK_CLASSES:
        cls = getattr(yv, name)
        assert cls.__module__ == "yolo_variants" and cls.__qualname__ == name
        assert getattr(yv, name) is cls
    with pytest.raises(AttributeError):
        yv.NoSuchBlock


# ---- pruning ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pruned_30(tiny_yolo):
    sf, _ = py.make_split_free(tiny_yolo)
    return py.prune_model(sf, 0.3, imgsz=64)


def test_prune_model_shrinks_the_net_and_keeps_the_prediction_layout(tiny_yolo, pruned_30):
    assert py.count_params(pruned_30) < 0.8 * py.count_params(tiny_yolo)
    assert py.count_macs(pruned_30, 64) < py.count_macs(tiny_yolo, 64)
    with torch.inference_mode():
        out = pruned_30(torch.randn(1, 3, 64, 64))[0]
    assert out.shape == (1, 4 + 5, 8 * 8 + 4 * 4 + 2 * 2) and torch.isfinite(out).all()
    assert not pruned_30.training
    detect = pruned_30.model[-1]
    assert [b[-1].out_channels for b in detect.cv2] == [64] * 3        # 4 * reg_max: not pruned
    assert [b[-1].out_channels for b in detect.cv3] == [5] * 3         # nc: not pruned
    assert detect.dfl.conv.out_channels == 1


def test_prune_model_leaves_the_input_untouched(tiny_yolo):
    before = py.count_params(tiny_yolo)
    sf, _ = py.make_split_free(tiny_yolo)
    py.prune_model(sf, 0.5, imgsz=64)
    assert py.count_params(sf) == before == py.count_params(tiny_yolo)


def test_higher_ratio_prunes_more(tiny_yolo, pruned_30):
    sf, _ = py.make_split_free(tiny_yolo)
    assert py.count_params(py.prune_model(sf, 0.5, imgsz=64)) < py.count_params(pruned_30)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_prune_model_rejects_ratio_outside_0_1(tiny_yolo, ratio):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        py.prune_model(tiny_yolo, ratio)


# ---- artifact ---------------------------------------------------------------------------------
def test_artifact_path_names_the_variant():
    assert py.artifact_path("../yolo11n.pt", 0.30, "/m") == "/m/yolo11n_p30.pt"
    assert py.artifact_path("yolo11n.pt", 0.5, "/m") == "/m/yolo11n_p50.pt"


def test_build_artifact_has_the_task5_fields_and_measured_counts(tiny_yolo, pruned_30):
    art = py.build_artifact(pruned_30, "../yolo11n.pt", 0.3, py.count_params(tiny_yolo),
                            py.count_macs(tiny_yolo, 64), imgsz=64, split_counts={"C3k2": 8})
    assert art["format"] == "pi-power-pruned/1" and art["base_model"] == "yolo11n.pt"
    assert art["ratio"] == 0.3 and art["params_after"] == py.count_params(pruned_30)
    assert art["params_after"] < art["params_before"] and art["macs_after"] < art["macs_before"]
    assert art["method"]["fine_tuned"] is False and art["method"]["split_free_replaced"] == {"C3k2": 8}
    assert art["method"]["bn_recalibration"] == {"images": 0, "subset": "calib200",
                                                 "momentum": "cumulative", "batch": 20}
    assert "GroupMagnitudeImportance" in art["method"]["importance"]
    art = py.build_artifact(pruned_30, "yolo11n.pt", 0.3, 1, 1, imgsz=64, bn_images=200)
    assert art["method"]["bn_recalibration"]["images"] == 200
    assert yv.validate_artifact(art, yv.PRUNED_FORMAT)["ratio"] == 0.3


@pytest.mark.parametrize("ratio", [0, 1, 0.0, 2.0])
def test_build_artifact_rejects_ratio_outside_0_1(pruned_30, ratio):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        py.build_artifact(pruned_30, "yolo11n.pt", ratio, 1, 1, imgsz=64)


# ---- BN recalibration -------------------------------------------------------------------------
def _bn_buffers(model):
    return {n: b.clone() for n, b in model.named_buffers()}


def test_recalibrate_bn_changes_only_bn_buffers_and_no_weight(pruned_30):
    model = _randomise_bn(copy.deepcopy(pruned_30))
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = _bn_buffers(model)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in bns]
    torch.manual_seed(5)
    n = py.recalibrate_bn(model, [torch.rand(4, 3, 64, 64) for _ in range(3)])
    assert n == 12
    for name, p in model.named_parameters():                      # conv + BN weights and biases
        assert torch.equal(p, params[name]), name
    assert [m.momentum for m in bns] == momenta                   # restored
    assert not any(m.training for m in model.modules())           # whole model back in eval
    changed = {name for name, b in model.named_buffers() if not torch.equal(b, buffers[name])}
    bn_buffers = {f"{name}.{b}" for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)
                  for b in ("running_mean", "running_var", "num_batches_tracked")}
    assert changed and changed <= bn_buffers
    assert all(m.num_batches_tracked.item() == 3 for m in bns)    # reset, then one per batch


def test_recalibrate_bn_is_the_cumulative_average_of_the_batch_statistics():
    torch.manual_seed(6)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4, momentum=0.1)).eval()
    net[1].running_mean.fill_(99.0)                               # must be forgotten
    batches = [torch.randn(5, 3, 8, 8) * (i + 1) for i in range(3)]
    py.recalibrate_bn(net, batches)
    ys = [net[0](b) for b in batches]
    mean = torch.stack([y.mean((0, 2, 3)) for y in ys]).mean(0)
    var = torch.stack([y.var((0, 2, 3), unbiased=True) for y in ys]).mean(0)
    assert torch.allclose(net[1].running_mean, mean, atol=1e-6)
    assert torch.allclose(net[1].running_var, var, rtol=1e-5)
    assert net[1].momentum == 0.1 and not net.training


def test_recalibrate_bn_restores_eval_and_momentum_when_a_forward_fails():
    net = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2)).eval()
    with pytest.raises(RuntimeError):
        py.recalibrate_bn(net, [torch.zeros(2, 5, 4, 4)])         # wrong channel count
    assert net[1].momentum == 0.1 and not any(m.training for m in net.modules())


def test_recalibrate_bn_rejects_no_images():
    with pytest.raises(ValueError, match="at least one image"):
        py.recalibrate_bn(nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2)), [])


def test_calibration_batches_are_letterboxed_batches_of_the_sorted_images(tmp_path):
    import cv2
    import numpy as np
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.full((40, 60, 3), 40 * i, np.uint8))
    shapes = [tuple(b.shape) for b in py.calibration_batches(str(tmp_path), imgsz=64, batch=2)]
    assert shapes == [(2, 3, 64, 64), (2, 3, 64, 64), (1, 3, 64, 64)]


def test_round_trip_and_fresh_process_load_with_only_src_on_the_path(tiny_yolo, pruned_30, tmp_path):
    path = str(tmp_path / "t_p30.pt")
    torch.save(py.build_artifact(pruned_30, "yolo11n.pt", 0.3, py.count_params(tiny_yolo),
                                 py.count_macs(tiny_yolo, 64), imgsz=64), path)
    back = yv.load_artifact(path, yv.PRUNED_FORMAT)
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(back["model"].eval()(x)[0], pruned_30(x)[0])
    info = py.verify_artifact(path, py.fp32_output_shape(tiny_yolo, 64), imgsz=64)
    assert info["output_shape"] == [1, 9, 84] and info["ratio"] == 0.3
    assert info["leaked_modules"] == [] and info["params_after"] == py.count_params(pruned_30)


def test_verify_artifact_rejects_a_wrong_output_shape(tiny_yolo, pruned_30, tmp_path):
    path = str(tmp_path / "t_p30.pt")
    torch.save(py.build_artifact(pruned_30, "yolo11n.pt", 0.3, 10, 10, imgsz=64), path)
    with pytest.raises(AssertionError, match="output shape"):
        py.verify_artifact(path, [1, 84, 8400], imgsz=64)


def test_load_artifact_rejects_an_int8_artifact_for_a_pruned_run(tiny_yolo, tmp_path):
    path = str(tmp_path / "x.pt")
    torch.save({"format": yv.INT8S_FORMAT, "model": tiny_yolo}, path)
    with pytest.raises(ValueError, match="format"):
        yv.load_artifact(path, yv.PRUNED_FORMAT)
