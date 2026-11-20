"""Tests for tools/quantize_yolo.py: artifact building, module census, FX pipeline on a tiny YOLO."""

import pytest
import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn

import quantize_yolo as qy

pytestmark = [pytest.mark.filterwarnings("ignore:TypedStorage is deprecated"),
              pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor")]


def _toy_quantized(with_float_conv=True):
    torch.backends.quantized.engine = "qnnpack"
    qconv = nnq.Conv2d(3, 4, 3)
    layers = [nnq.Quantize(0.02, 0, torch.quint8), qconv, nnq.DeQuantize()]
    if with_float_conv:
        layers.append(nn.Conv2d(4, 4, 1))
    return nn.Sequential(*layers).eval()


# ---- pure pieces ------------------------------------------------------------------------------
def test_artifact_path_is_models_stem_int8s():
    assert qy.artifact_path("../yolo11n.pt", "/m") == "/m/yolo11n_int8s.pt"
    assert qy.artifact_path("yolo11s", "/m") == "/m/yolo11s_int8s.pt"


def test_summarize_modules_separates_quantized_stubs_and_float_compute():
    c = qy.summarize_modules(_toy_quantized())
    assert c == {"quantized": {"Conv2d": 1}, "float_compute": {"Conv2d": 1}, "stubs": 2}


def test_fp32_module_records_name_class_and_reason_of_float_compute():
    net = _toy_quantized()
    recs = qy.fp32_module_records(net, {"3": "closing conv kept float"})
    assert recs == [{"name": "3", "class": "Conv2d", "reason": "closing conv kept float"}]
    assert qy.fp32_module_records(_toy_quantized(with_float_conv=False), {}) == []


def test_build_artifact_has_the_task5_fields_and_the_honest_count():
    art = qy.build_artifact(_toy_quantized(), "../yolo11n.pt", [{"name": "x"}], "m")
    assert art["format"] == "pi-power-int8s/1" and art["base_model"] == "yolo11n.pt"
    assert art["quantized_compute_modules"] == 1
    assert art["calibration"] == {"n_images": 200, "subset": "calib200", "engine": "qnnpack", "method": "m"}
    assert art["torch_version"] == torch.__version__ and art["fp32_modules"] == [{"name": "x"}]


def test_build_artifact_refuses_stubs_without_quantized_compute():
    stubs_only = nn.Sequential(nnq.Quantize(0.02, 0, torch.quint8), nn.Conv2d(3, 4, 3), nnq.DeQuantize())
    with pytest.raises(ValueError, match="0 quantized compute modules"):
        qy.build_artifact(stubs_only, "yolo11n.pt", [], "m")


# ---- the FX pipeline on a tiny randomly initialised YOLO11 (no weights, no downloads) ---------
@pytest.fixture(scope="module")
def tiny_yolo():
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    m = DetectionModel("yolo11n.yaml", nc=5, verbose=False).eval()
    return m


def test_fx_pipeline_quantizes_backbone_and_head_and_keeps_detect_decode_and_attention_fp32(tiny_yolo, tmp_path):
    calib = [torch.rand(1, 3, 64, 64) for _ in range(3)]
    q, reasons = qy.quantize_fx_model(tiny_yolo, calib, imgsz=64, log=lambda *_: None)
    census = qy.summarize_modules(q)
    assert census["quantized"]["Conv2d"] > 50                       # backbone + neck + head branches
    recs = qy.fp32_module_records(q, reasons)
    assert {r["class"] for r in recs} == {"Detect", "C2PSA"} and all(r["reason"] for r in recs)
    silu = qy.silu_record(q)
    assert silu["class"] == "SiLU" and silu["call_sites"] > 40      # one shared nn.SiLU, many calls
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        out, ref = q(x)[0], tiny_yolo(x)[0]
    assert out.shape == ref.shape == (1, 4 + 5, (8 * 8) + (4 * 4) + (2 * 2))
    assert torch.isfinite(out).all()
    assert tiny_yolo.model[0].conv.weight.dtype == torch.float32    # the input model is untouched

    # artifact round trip through the Pi's loader (yolo_variants.load_artifact)
    import yolo_variants as yv
    path = str(tmp_path / "t_int8s.pt")
    torch.save(qy.build_artifact(q, "yolo11n.pt", recs, "test"), path)
    back = yv.load_artifact(path, yv.INT8S_FORMAT)
    assert back["quantized_compute_modules"] == yv.count_quantized_modules(back["model"].eval())
    with torch.inference_mode():
        assert torch.allclose(back["model"](x)[0], out, atol=1e-5)


def test_traceable_ultralytics_restores_the_stock_forwards():
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import block
    before = (block.C2f.forward, tasks.DetectionModel.forward)
    with qy.traceable_ultralytics():
        assert block.C2f.forward is not before[0]
    assert (block.C2f.forward, tasks.DetectionModel.forward) == before


def test_c2f_traceable_forward_matches_the_stock_forward():
    from ultralytics.nn.modules import block
    torch.manual_seed(0)
    m = block.C2f(8, 8, n=2, shortcut=True).eval()
    x = torch.randn(1, 8, 16, 16)
    stock = m(x)
    assert torch.allclose(qy._c2f_forward_traceable(m, x), stock, atol=1e-6)
