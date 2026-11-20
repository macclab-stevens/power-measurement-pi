# tests/test_variants.py -- precision / compression variants (src/yolo_variants.py, ImageTask, run.py)
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import torch
import torch.nn as nn

import run
import yolo_variants as yv
from layers import LayerProfiler
from tasks import ImageTask


def _parse(*argv):
    return run.parse_args(list(argv))


def _err(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        run.parse_args(list(argv))
    assert e.value.code == 2
    return capsys.readouterr().err


# ---- variant_id -----------------------------------------------------------------------------
@pytest.mark.parametrize("model,quant,ratio,want", [
    ("yolo11n.pt", "fp32", None, "yolo11n"),
    ("yolo11n.pt", "int8-dynamic", None, "yolo11n_int8d"),
    ("yolo11n.pt", "int8-static", None, "yolo11n_int8s"),
    ("yolo11n.pt", "fp16", None, "yolo11n_fp16"),
    ("yolo11n.pt", "fp32", 0.3, "yolo11n_p30"),
    ("yolov8s.pt", "fp32", 0.5, "yolov8s_p50"),
    ("/home/pi/models/yolo11m.pt", "fp16", None, "yolo11m_fp16"),
    ("yolo11n", "fp32", None, "yolo11n"),
    ("minigpt", "fp32", None, "minigpt"),
])
def test_variant_id_table(model, quant, ratio, want):
    assert yv.variant_id(model, quant, ratio) == want


def test_variant_id_defaults_and_errors():
    assert yv.variant_id("yolo11n.pt") == "yolo11n"
    with pytest.raises(ValueError, match="cannot be combined"):
        yv.variant_id("yolo11n.pt", "int8-static", 0.3)
    with pytest.raises(ValueError, match="unknown quant"):
        yv.variant_id("yolo11n.pt", "int4")


# ---- int8-dynamic: replaced-module count and the 0-module acceptance failure -------------------
def test_quantize_dynamic_replaces_linear_and_leaves_original_alone():
    m = nn.Sequential(nn.Linear(8, 4))
    q, n = yv.quantize_dynamic_counted(m)
    assert n == 1
    assert ".quantized" in type(q[0]).__module__
    assert isinstance(m[0], nn.Linear)                 # the FP32 module is not modified
    assert q(torch.randn(2, 8)).shape == (2, 4)        # and the replacement really runs
    assert torch.backends.quantized.engine == "qnnpack"


def test_quantize_dynamic_on_conv_only_model_is_a_noop_with_count_zero():
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
    q, n = yv.quantize_dynamic_counted(m)
    assert n == 0
    assert isinstance(q[0], nn.Conv2d)


def test_stubs_alone_are_not_quantized_compute():
    """R18: Quantize/DeQuantize stubs around float convs must not pass the int8-static gate."""
    import torch.ao.nn.quantized as nnq
    m = nn.Sequential(nnq.Quantize(1.0, 0, torch.quint8), nn.Conv2d(3, 4, 3), nnq.DeQuantize())
    assert yv.count_quantized_modules(m) == 0
    assert "int8-static quantized 0 modules" in yv.quant_count_errors(
        "int8-static", yv.count_quantized_modules(m))[0]


def test_real_quantized_conv_linear_and_fused_variants_count():
    import torch.ao.nn.intrinsic.quantized as nniq
    import torch.ao.nn.quantized as nnq
    torch.backends.quantized.engine = "qnnpack"
    qc = nnq.Conv2d(3, 4, 3)
    qcr = nniq.ConvReLU2d(3, 4, 3)
    ql = nnq.Linear(4, 2)
    m = nn.Sequential(nnq.Quantize(1.0, 0, torch.quint8), qc, qcr, ql, nnq.DeQuantize(), nn.Conv2d(3, 4, 3))
    assert yv.count_quantized_modules(m) == 3


def test_zero_quantized_modules_fails_acceptance_with_clear_message():
    _, n = yv.quantize_dynamic_counted(nn.Sequential(nn.Conv2d(3, 4, 3)))
    errs = yv.quant_count_errors("int8-dynamic", n)
    assert len(errs) == 1 and "int8-dynamic quantized 0 modules" in errs[0]
    assert "float32" in errs[0] and "INT8" in errs[0]


def test_quant_count_errors_only_for_int8_variants_with_nothing_quantized():
    assert yv.quant_count_errors("int8-dynamic", 3) == []
    assert yv.quant_count_errors("fp32", 0) == []
    assert yv.quant_count_errors("fp16", 0) == []
    assert yv.quant_count_errors("fp32", None) == []
    assert "int8-static quantized 0 modules" in yv.quant_count_errors("int8-static", 0)[0]


# ---- fp16 executed-dtype check on synthetic layer_rows --------------------------------------
def _row(cls="Conv2d", dtypes=("torch.float16",), leaf=True):
    return {"class": cls, "leaf": leaf, "input_dtypes": json.dumps(sorted(dtypes))}


def test_fp16_check_passes_when_all_conv_leaves_ran_float16():
    rows = [_row(), _row(), _row("Linear", ("torch.float32",)),
            _row(dtypes=("torch.float32",), leaf=False)]   # non-leaf rows are not the kernels
    assert yv.fp16_errors(rows) == []
    assert yv.executed_conv_dtype(rows) == "float16"


def test_fp16_check_fails_when_any_conv_leaf_ran_float32():
    rows = [_row(), _row(dtypes=("torch.float32",))]
    errs = yv.fp16_errors(rows)
    assert len(errs) == 1 and "fp16 requested but conv ran in float32" in errs[0]
    assert yv.executed_conv_dtype(rows) == "float16+float32"


def test_fp16_check_fails_closed_without_conv_leaves():
    assert "no Conv2d leaf" in yv.fp16_errors([_row("Linear")])[0]
    assert yv.fp16_errors([])
    assert yv.executed_conv_dtype([_row("Linear")]) is None


def test_fp16_check_on_real_profiler_rows():
    # end to end through LayerProfiler: a half module fed half input reports float16
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()).half().eval()
    prof = LayerProfiler(net)
    with torch.no_grad():
        net(torch.randn(1, 3, 8, 8).half())
    rows = prof.aggregated(exclude_frame=-1)
    assert yv.fp16_errors(rows) == [] and yv.executed_conv_dtype(rows) == "float16"
    # the failure mode: half weights cannot run float32 input, so a silent float32 fallback is
    # modelled by a float32 net under the same check
    net32 = nn.Sequential(nn.Conv2d(3, 4, 3)).eval()
    prof32 = LayerProfiler(net32)
    with torch.no_grad():
        net32(torch.randn(1, 3, 8, 8))
    assert "conv ran in float32" in yv.fp16_errors(prof32.aggregated(exclude_frame=-1))[0]


# ---- mAP reference lookup ---------------------------------------------------------------------
def test_map_reference_hit_and_miss(tmp_path):
    entry = {"map50_95": 0.39, "map50": 0.55, "n_images": 500, "dataset": "coco-val-500",
             "tool_version": "1", "measured": "2026-11-20"}
    f = tmp_path / "map_reference.json"
    f.write_text(json.dumps({"yolo11n@640": entry}))
    assert yv.lookup_map_reference(str(f), "yolo11n", 640) == entry
    assert yv.lookup_map_reference(str(f), "yolo11n", 480) is None          # other imgsz
    assert yv.lookup_map_reference(str(f), "yolo11n_int8s", 640) is None    # other variant


def test_map_reference_never_raises(tmp_path):
    assert yv.lookup_map_reference(str(tmp_path / "missing.json"), "yolo11n", 640) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert yv.lookup_map_reference(str(bad), "yolo11n", 640) is None
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]")
    assert yv.lookup_map_reference(str(lst), "yolo11n", 640) is None


def test_committed_map_reference_is_valid_json_object():
    with open(run.MAP_REFERENCE_PATH) as f:
        assert isinstance(json.load(f), dict)


# ---- artifact loading / format validation -----------------------------------------------------
def _save(tmp_path, name, obj):
    p = str(tmp_path / name)
    torch.save(obj, p)
    return p


def test_load_artifact_accepts_expected_format(tmp_path):
    net = nn.Sequential(nn.Linear(2, 2))
    p = _save(tmp_path, "a.pt", {"format": yv.PRUNED_FORMAT, "model": net, "base_model": "yolo11n.pt"})
    art = yv.load_artifact(p, yv.PRUNED_FORMAT)
    assert isinstance(art["model"], nn.Sequential)


def test_load_artifact_wrong_format_string(tmp_path):
    p = _save(tmp_path, "a.pt", {"format": "pi-power-pruned/2", "model": nn.Linear(1, 1)})
    with pytest.raises(ValueError, match="pi-power-pruned/2.*pi-power-pruned/1"):
        yv.load_artifact(p, yv.PRUNED_FORMAT)
    p = _save(tmp_path, "b.pt", {"format": yv.PRUNED_FORMAT, "model": nn.Linear(1, 1)})
    with pytest.raises(ValueError, match="pi-power-pruned/1.*pi-power-int8s/1"):
        yv.load_artifact(p, yv.INT8S_FORMAT)


def test_load_artifact_rejects_non_dict_missing_format_and_non_module(tmp_path):
    with pytest.raises(ValueError, match="expected a"):
        yv.load_artifact(_save(tmp_path, "m.pt", nn.Linear(1, 1)), yv.INT8S_FORMAT)
    with pytest.raises(ValueError, match="format is None"):
        yv.load_artifact(_save(tmp_path, "n.pt", {"model": nn.Linear(1, 1)}), yv.INT8S_FORMAT)
    with pytest.raises(ValueError, match="not an nn.Module"):
        yv.load_artifact(_save(tmp_path, "o.pt", {"format": yv.INT8S_FORMAT, "model": "x"}),
                         yv.INT8S_FORMAT)


# ---- ImageTask._apply_variant -----------------------------------------------------------------
def _task(module, **opt):
    t = ImageTask()
    t._module = module
    t._is_detection = True
    t._imgsz = 32
    t._conf = 0.25
    t._apply_variant("yolo11n.pt", SimpleNamespace(**opt))
    return t


def test_default_options_change_nothing():
    net = nn.Sequential(nn.Conv2d(3, 4, 3))
    t = _task(net)
    assert t.module is net and not t._half and not t._variant_oracle
    assert t.variant["quant"] == "fp32" and t.variant["prune"] is None
    assert t.variant["quantized_modules"] is None


def test_fp16_variant_halves_module_and_preprocess_input():
    t = _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="fp16")
    assert next(t.module.parameters()).dtype == torch.float16
    assert t._half and t._variant_oracle
    x = t.preprocess(t._source_frame())
    assert x.dtype == torch.float16 and x.shape == (1, 3, 32, 32)


def test_int8_dynamic_variant_records_quantized_module_count():
    t = _task(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)), quant="int8-dynamic")
    assert t.variant["quantized_modules"] == 2 and t._variant_oracle
    t0 = _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-dynamic")
    assert t0.variant["quantized_modules"] == 0


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
def test_int8_static_loads_artifact_and_hashes_it(tmp_path):
    net = nn.Sequential(nn.Linear(4, 4))
    q, _ = yv.quantize_dynamic_counted(net)       # stands in for a statically quantized module
    p = _save(tmp_path, "q.pt", {"format": yv.INT8S_FORMAT, "model": q, "base_model": "yolo11n.pt",
                                 "fp32_modules": [], "calibration": {}})
    t = _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static", quant_weights=p)
    assert t.module is not None and ".quantized" in type(t.module[0]).__module__
    assert t.variant["quantized_modules"] == 1
    import hashlib
    assert t.variant["quant_weights_sha256"] == hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_int8_static_artifact_declared_count_must_match_loaded_count(tmp_path):
    q, _ = yv.quantize_dynamic_counted(nn.Sequential(nn.Linear(4, 4)))
    ok = _save(tmp_path, "ok.pt", {"format": yv.INT8S_FORMAT, "model": q, "base_model": "yolo11n.pt",
                                   "quantized_compute_modules": 1})
    assert _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static",
                 quant_weights=ok).variant["quantized_modules"] == 1
    bad = _save(tmp_path, "bad.pt", {"format": yv.INT8S_FORMAT, "model": q, "base_model": "yolo11n.pt",
                                     "quantized_compute_modules": 99})
    with pytest.raises(ValueError, match="declares 99 quantized compute modules but 1"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static", quant_weights=bad)


def test_int8_static_errors(tmp_path):
    with pytest.raises(ValueError, match="needs --quant-weights"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static")
    p = _save(tmp_path, "w.pt", {"format": "wrong/1", "model": nn.Linear(1, 1)})
    with pytest.raises(ValueError, match="expected 'pi-power-int8s/1'"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static", quant_weights=p)
    p = _save(tmp_path, "x.pt", {"format": yv.INT8S_FORMAT, "model": nn.Linear(1, 1),
                                 "base_model": "yolov8n.pt"})
    with pytest.raises(ValueError, match="base_model"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="int8-static", quant_weights=p)


def test_prune_variant_records_path_ratio_sha(tmp_path):
    pruned = nn.Sequential(nn.Conv2d(3, 2, 3))
    p = _save(tmp_path, "p.pt", {"format": yv.PRUNED_FORMAT, "model": pruned, "base_model": "yolo11n.pt",
                                 "ratio": 0.3, "params_before": 10, "params_after": 7})
    t = _task(nn.Sequential(nn.Conv2d(3, 4, 3)), prune=p)
    assert t.module[0].out_channels == 2 and t._variant_oracle
    assert t.variant["prune"]["path"] == p and t.variant["prune"]["ratio"] == 0.3
    assert len(t.variant["prune"]["sha256"]) == 64
    assert yv.variant_id("yolo11n.pt", "fp32", t.variant["prune_ratio"]) == "yolo11n_p30"


def test_prune_artifact_errors(tmp_path):
    base = {"format": yv.PRUNED_FORMAT, "model": nn.Linear(1, 1), "base_model": "yolo11n.pt"}
    for ratio in (None, 0, 1.5, "0.3"):
        p = _save(tmp_path, "r.pt", {**base, "ratio": ratio})
        with pytest.raises(ValueError, match="ratio"):
            _task(nn.Sequential(nn.Conv2d(3, 4, 3)), prune=p)
    p = _save(tmp_path, "f.pt", {**base, "format": "pi-power-int8s/1", "ratio": 0.3})
    with pytest.raises(ValueError, match="expected 'pi-power-pruned/1'"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), prune=p)


def test_quant_and_prune_flags_conflict_at_task_level():
    with pytest.raises(ValueError, match="cannot be combined"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="fp16", prune="x.pt")
    with pytest.raises(ValueError, match="only used with --quant int8-static"):
        _task(nn.Sequential(nn.Conv2d(3, 4, 3)), quant="fp16", quant_weights="x.pt")


# ---- oracle: the executed module, not yolo.predict ----------------------------------------------
class _FixedDetector(nn.Module):
    """Ignores pixel values; emits a (1, 4+2, 5) YOLO-style prediction with one box of `conf`."""

    def __init__(self, conf):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        pred = torch.zeros(1, 6, 5)
        pred[0, :4, 0] = torch.tensor([16.0, 16.0, 10.0, 10.0])
        pred[0, 4, 0] = conf
        self.register_buffer("pred", pred)

    def forward(self, x):
        self.conv(x)
        return (self.pred.to(x.dtype), None)


class _ExplodingYolo:
    def predict(self, *a, **k):
        raise AssertionError("v1 yolo.predict oracle must not run for a variant")


def _oracle_task(conf, **opt):
    t = _task(_FixedDetector(conf), **opt)
    t._check_image = os.path.join(os.path.dirname(__file__), "..", "bus.jpg")
    t._yolo = _ExplodingYolo()
    return t


@pytest.mark.parametrize("opt", [{"quant": "fp16"}, {"quant": "int8-dynamic"}])
def test_variant_oracle_uses_executed_module(opt):
    t = _oracle_task(0.9, **opt)
    assert t.oracle() is True
    assert t._checked == pytest.approx(0.9, abs=1e-3)


def test_variant_oracle_fails_below_conf_and_with_no_detection():
    assert _oracle_task(0.1, quant="fp16").oracle() is False
    t = _oracle_task(0.0, quant="fp16")
    assert t.oracle() is False and t._checked == 0.0


def test_fp32_keeps_v1_oracle_path():
    t = _oracle_task(0.9)            # default quant: yolo.predict is used (and here explodes)
    with pytest.raises(AssertionError, match="v1 yolo.predict"):
        t.oracle()


def test_fp16_run_one_casts_output_to_float32_before_nms():
    t = _oracle_task(0.9, quant="fp16")
    seen = {}
    real = t.postprocess

    def spy(out):
        seen["in"] = out[0].dtype
        return real(out)
    t.postprocess = spy
    out = t.run_one("forward")
    assert out[0].dtype == torch.float16          # the module really executed in half
    assert seen["in"] == torch.float16            # postprocess received half ...
    assert t.postprocess(out)[0].dtype == torch.float32    # ... and NMS output is float32


# ---- argparse / dataset_version -----------------------------------------------------------------
def test_v1_defaults_for_variant_flags():
    a = _parse()
    assert (a.quant, a.quant_weights, a.prune) == ("fp32", None, None)
    assert run.dataset_version(a) == 1
    assert run.dataset_version(_parse("--quant", "fp32")) == 1


@pytest.mark.parametrize("argv", [
    ["--quant", "fp16"], ["--quant", "int8-dynamic"],
    ["--quant", "int8-static", "--quant-weights", "q.pt"], ["--prune", "p.pt"]])
def test_variant_flags_make_the_run_v2(argv):
    assert run.dataset_version(_parse(*argv)) == 2


def test_variant_flag_conflicts(capsys):
    assert "--quant-weights" in _err(capsys, "--quant", "int8-static")
    assert "--quant-weights" in _err(capsys, "--quant", "fp16", "--quant-weights", "q.pt")
    assert "--quant-weights" in _err(capsys, "--quant-weights", "q.pt")
    assert "--prune" in _err(capsys, "--prune", "p.pt", "--quant", "fp16")
    assert "--task image" in _err(capsys, "--task", "lm", "--model", "minigpt", "--quant", "fp16")
    assert "--quant" in _err(capsys, "--quant", "int4")
    assert _parse("--prune", "p.pt").prune == "p.pt"
