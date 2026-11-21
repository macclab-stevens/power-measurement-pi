"""INT8 static post-training quantization of a YOLO model for the Pi's qnnpack engine.

  cd tools
  uv run python quantize_yolo.py --model ../yolo11n.pt          # -> ../models/yolo11n_int8s.pt
  uv run python quantize_yolo.py --model ../yolo11s.pt
  uv run python quantize_yolo.py --model ../yolo11n.pt --verify-only   # reload + forward in a fresh process

The artifact is the Task 5 dict (src/yolo_variants.py INT8S_FORMAT 'pi-power-int8s/1') that
`src/run.py --quant int8-static --quant-weights models/<stem>_int8s.pt` loads:
    format, model (the converted nn.Module), base_model, fp32_modules, calibration, torch_version,
    quantized_compute_modules (real quantized Conv/Linear count; stubs do not count).

Method: FX graph mode (torch.ao.quantization.quantize_fx prepare_fx / convert_fx, qnnpack qconfig
mapping, activations quint8, weights qint8), Conv+BN fused first (ultralytics `.fuse()`), calibrated
on tools/data/calib200 with the Pi's own letterbox preprocess at 640 (eval_map.make_preprocess).

What traces and what does not (the honest part, recorded in the artifact's `fp32_modules`):
  * backbone/neck (Conv, C2f/C3k2, C3k, Bottleneck, SPPF, Upsample, Concat) traces once the
    ultralytics C2f forward is swapped for a slice-based equivalent while tracing (`list(x.chunk())`
    iterates a Proxy). The swap only affects tracing; the converted graph is self-contained.
  * Detect and C2PSA (attention: `x.shape` unpacking, split-unpacking) cannot be traced, so they stay
    FP32 as modules. Detect's per-level conv branches (cv2 box, cv3 class) are quantized as
    separate float-in/float-out subgraphs; its decode (DFL, anchors, dist2bbox, sigmoid) stays FP32.
  * SiLU has no quantized kernel: it runs FP32 between quantized convs (dequantize -> SiLU -> quantize).
  * Measured on yolo11n/coco500 (see the Task 8 report): keeping the closing 1x1 box/class convs of the
    head FP32 as well bought +0.2 mAP50-95 points, so every head conv is quantized.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import warnings

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import yolo_variants as yv  # noqa: E402  (artifact format, honest quantized-module count)

CALIB_DIR = os.path.join(HERE, "data", "calib200", "images")
MODELS_DIR = os.path.join(REPO, "models")
IMGSZ = 640
ENGINE = "qnnpack"
CALIB_SUBSET = "calib200"
CALIB_N_IMAGES = 200
UNTRACEABLE = ("Detect", "C2PSA")            # class names kept as FP32 leaf modules in the main graph
OBSERVERS = ("histogram", "minmax")
DETECT_REASON = ("Detect decode (DFL, anchor grid from tensor shapes, dist2bbox, sigmoid) does not trace "
                 "under FX and is cheap: kept FP32; its cv2/cv3 conv branches are quantized")
C2PSA_REASON = ("C2PSA attention: Attention.forward unpacks tensor shapes/splits (`B, C, H, W = x.shape`), "
                "not traceable by FX; 20x20 map at 640, kept FP32")
SILU_REASON = "no quantized SiLU kernel: runs FP32 between quantized convs (dequantize -> SiLU -> quantize)"


# ---- pure pieces (unit tested) ------------------------------------------------------------------
def artifact_path(model_path: str, out_dir: str = MODELS_DIR) -> str:
    """models/<stem>_int8s.pt for ../yolo11n.pt (the name run.py's --quant-weights is given)."""
    stem = os.path.basename(model_path)
    stem = stem[:-3] if stem.endswith(".pt") else stem
    return os.path.join(out_dir, f"{stem}_int8s.pt")


def module_kind(module: nn.Module) -> str:
    """'quantized' for a real quantized compute module, 'stub' for a quantize/dequantize/functional
    helper, 'float' for everything else (plain FP32 module)."""
    t = type(module)
    if ".quantized" in t.__module__:
        return "quantized" if yv.count_quantized_modules(module) else "stub"
    return "float"


def summarize_modules(model: nn.Module) -> dict:
    """Leaf-level census of a converted model, by module class:
    {'quantized': {cls: n}, 'float_compute': {cls: n}, 'stubs': n}.

    A 'leaf' is a module that has no children, or a quantized module (whose own helpers such as
    LinearPackedParams are not walked). Float compute = Conv2d / Linear / ConvTranspose2d leaves that
    still run in float; activations, pooling, reshapes are not counted."""
    q: dict[str, int] = {}
    f: dict[str, int] = {}
    stubs = 0

    def walk(m):
        nonlocal stubs
        kind = module_kind(m)
        if kind == "quantized":
            q[type(m).__name__] = q.get(type(m).__name__, 0) + 1
            return
        if kind == "stub":
            stubs += 1
            return
        kids = list(m.children())
        if not kids:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                f[type(m).__name__] = f.get(type(m).__name__, 0) + 1
            return
        for c in kids:
            walk(c)

    walk(model)
    return {"quantized": q, "float_compute": f, "stubs": stubs}


def fp32_module_records(model: nn.Module, reasons: dict[str, str]) -> list[dict]:
    """[{name, class, reason}] for every module left in float that still holds compute.

    `reasons` maps a module-name prefix to the reason it stayed FP32; the longest prefix wins. The
    walk stops at the first module that is a float compute leaf or one of the UNTRACEABLE classes, so
    each reported entry is a unit a reader can find in the model (e.g. 'model.23' Detect)."""
    out: list[dict] = []

    def reason_for(name: str) -> str:
        best = ""
        for pfx in reasons:
            if (name == pfx or name.startswith(pfx + ".")) and len(pfx) >= len(best):
                best = pfx
        return reasons.get(best, "left in float by the quantization config")

    def has_float_compute(m) -> bool:
        return any(isinstance(x, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))
                   for x in m.modules() if module_kind(x) == "float")

    def walk(name, m):
        if module_kind(m) != "float":
            return
        if type(m).__name__ in UNTRACEABLE or not list(m.children()):
            if has_float_compute(m):
                out.append({"name": name, "class": type(m).__name__, "reason": reason_for(name)})
            return
        # a container (GraphModule, Sequential, ...) - report children, not the container
        for cn, c in m.named_children():
            walk(f"{name}.{cn}" if name else cn, c)

    walk("", model)
    return out


def build_artifact(model: nn.Module, base_model: str, fp32_modules: list, method: str,
                   n_images: int = CALIB_N_IMAGES, subset: str = CALIB_SUBSET,
                   extra_calibration: dict | None = None) -> dict:
    """The pi-power-int8s/1 dict. Refuses a model without real quantized compute modules."""
    n = yv.count_quantized_modules(model)
    if n == 0:
        raise ValueError("refusing to build an INT8 artifact: the model has 0 quantized compute "
                         "modules (Quantize/DeQuantize stubs alone are still FP32 compute)")
    calibration = {"n_images": n_images, "subset": subset, "engine": ENGINE, "method": method,
                   **(extra_calibration or {})}
    return {"format": yv.INT8S_FORMAT, "model": model, "base_model": os.path.basename(base_model),
            "fp32_modules": fp32_modules, "calibration": calibration,
            "torch_version": torch.__version__, "quantized_compute_modules": n}


# ---- tracing helpers --------------------------------------------------------------------------
def _c2f_forward_traceable(self, x):
    """C2f.forward without iterating a Proxy: same maths, slices instead of `.chunk(2, 1)`."""
    y = self.cv1(x)
    ys = [y[:, :self.c], y[:, self.c:]]
    ys.extend(m(ys[-1]) for m in self.m)
    return self.cv2(torch.cat(ys, 1))


@contextlib.contextmanager
def traceable_ultralytics():
    """While active, ultralytics modules trace under torch.fx: C2f (and C3k2, a subclass) uses a
    slice-based forward and DetectionModel.forward is `_predict_once(x)` (the stock forward takes
    *args/**kwargs, which FX cannot unpack). Restored on exit."""
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import block
    saved = (block.C2f.forward, tasks.DetectionModel.forward)
    block.C2f.forward = _c2f_forward_traceable
    tasks.DetectionModel.forward = lambda self, x: self._predict_once(x)
    try:
        yield
    finally:
        block.C2f.forward, tasks.DetectionModel.forward = saved


def make_qconfig_mapping(per_channel: bool, observer: str):
    """qnnpack QConfigMapping: quint8 activations (reduce_range), qint8 symmetric weights, per-channel
    by default (qnnpack supports per-channel conv weights); `observer` is histogram or minmax."""
    from torch.ao.quantization import (HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver,
                                       QConfig, QConfigMapping)
    act = (HistogramObserver if observer == "histogram" else MinMaxObserver).with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
    weight = (PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
              if per_channel else
              MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    return QConfigMapping().set_global(QConfig(activation=act, weight=weight))


# ---- the PTQ pipeline -------------------------------------------------------------------------
def calibration_inputs(images_dir: str, imgsz: int = IMGSZ, limit: int | None = None):
    """Yield 1x3xSxS float tensors for every *.jpg (sorted) through the Pi's own preprocess."""
    from eval_map import make_preprocess
    from tasks import read_rgb
    pre = make_preprocess(imgsz, half=False)
    names = sorted(n for n in os.listdir(images_dir) if n.lower().endswith(".jpg"))[:limit]
    for n in names:
        yield pre(read_rgb(os.path.join(images_dir, n)))


def quantize_fx_model(yolo_module: nn.Module, calib, imgsz: int = IMGSZ, per_channel: bool = True,
                      observer: str = "histogram", log=print):
    """FP32 DetectionModel -> (converted model, {module name: reason it stayed FP32}).

    `calib` is an iterable of preprocessed 1x3xSxS tensors. The input module is not modified."""
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from ultralytics.nn.modules import block, head
    torch.backends.quantized.engine = ENGINE
    model = copy.deepcopy(yolo_module).eval().fuse(verbose=False)
    example = torch.randn(1, 3, imgsz, imgsz)
    detect_name, detect = next((n, m) for n, m in model.named_modules() if isinstance(m, head.Detect))
    reasons = {detect_name: DETECT_REASON}
    reasons.update({n: C2PSA_REASON for n, m in model.named_modules() if isinstance(m, block.C2PSA)})
    mapping = make_qconfig_mapping(per_channel, observer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # torch.ao.quantization deprecation noise
        # 1. Detect's conv branches: each cv2[i] / cv3[i] is its own float-in/float-out subgraph
        feats = detect_inputs(model, example)
        with traceable_ultralytics():
            for branches in (detect.cv2, detect.cv3):
                for i, level_input in enumerate(feats):
                    branches[i] = prepare_fx(branches[i], mapping, (level_input,))
        # 2. the rest of the network; Detect and C2PSA stay FP32 leaves (not traced into)
        cfg = PrepareCustomConfig().set_non_traceable_module_classes([head.Detect, block.C2PSA])
        with traceable_ultralytics():
            prepared = prepare_fx(model, mapping, (example,), prepare_custom_config=cfg)
        # 3. calibrate (activation ranges of the main graph and of the head branches together)
        n = 0
        with torch.inference_mode():
            for x in calib:
                prepared(x)
                n += 1
        log(f"calibrated on {n} images")
        # 4. convert: the head branches (inside the Detect leaf), then the main graph
        prepared_detect = prepared.get_submodule(detect_name)
        for branches in (prepared_detect.cv2, prepared_detect.cv3):
            for i in range(len(branches)):
                branches[i] = convert_fx(branches[i])
        converted = convert_fx(prepared)
    return converted.eval(), reasons


def detect_inputs(model: nn.Module, example: torch.Tensor) -> list[torch.Tensor]:
    """The three feature maps Detect receives for `example` (one per level), FP32."""
    from ultralytics.nn.modules import head
    detect = next(m for m in model.modules() if isinstance(m, head.Detect))
    got: list = []
    h = detect.register_forward_pre_hook(lambda mod, args: got.append([a.detach() for a in args[0]]))
    try:
        with torch.inference_mode():
            model(example)
    finally:
        h.remove()
    return got[0]


# ---- verification and command line -------------------------------------------------------------
def verify_artifact(path: str, fp32_path: str, imgsz: int = IMGSZ) -> dict:
    """Reload `path` the way the Pi does (yolo_variants.load_artifact: torch.load weights_only=False)
    and check it: quantized compute modules > 0 and equal to the declared count, a forward at `imgsz`
    on qnnpack gives the FP32 output shape. Meant to run in a fresh process (see main)."""
    from ultralytics import YOLO
    torch.backends.quantized.engine = ENGINE
    art = yv.load_artifact(path, yv.INT8S_FORMAT)
    model = art["model"].eval()
    n = yv.count_quantized_modules(model)
    if n == 0 or n != art.get("quantized_compute_modules"):
        raise AssertionError(f"{path}: {n} quantized compute modules, artifact declares "
                             f"{art.get('quantized_compute_modules')}")
    x = next(calibration_inputs(CALIB_DIR, imgsz, 1))
    with torch.inference_mode():
        out = model(x)[0]
        ref = YOLO(fp32_path).model.eval()(x)[0]
    if out.shape != ref.shape:
        raise AssertionError(f"{path}: output shape {tuple(out.shape)} != FP32 {tuple(ref.shape)}")
    return {"quantized_compute_modules": n, "output_shape": list(out.shape), "engine":
            torch.backends.quantized.engine, "size_bytes": os.path.getsize(path)}


def silu_record(model: nn.Module) -> dict | None:
    """One record for the SiLU activations that run FP32 between quantized convs. ultralytics shares a
    single nn.SiLU instance across all Conv blocks, so count the call sites in the traced graphs."""
    import torch.fx as fx
    calls = 0
    for m in model.modules():
        if isinstance(m, fx.GraphModule):
            calls += sum(1 for n in m.graph.nodes if n.op == "call_module"
                         and isinstance(m.get_submodule(n.target), nn.SiLU))
    return {"name": "*.act", "class": "SiLU", "call_sites": calls, "reason": SILU_REASON} if calls else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base ultralytics weights, e.g. ../yolo11n.pt")
    ap.add_argument("--out-dir", default=MODELS_DIR)
    ap.add_argument("--calib-dir", default=CALIB_DIR)
    ap.add_argument("--calib-limit", type=int, default=None, help="use only the first N images (debug)")
    ap.add_argument("--per-tensor", action="store_true", help="per-tensor weights (default per-channel)")
    ap.add_argument("--observer", choices=OBSERVERS, default="histogram")
    ap.add_argument("--verify-only", action="store_true",
                    help="only reload the existing artifact and check it (what main runs in a subprocess)")
    args = ap.parse_args(argv)
    path = artifact_path(args.model, args.out_dir)

    if args.verify_only:
        print(json.dumps(verify_artifact(path, args.model)))
        return 0

    from ultralytics import YOLO
    fp32 = YOLO(args.model).model.eval()
    calib = list(calibration_inputs(args.calib_dir, IMGSZ, args.calib_limit))
    print(f"{len(calib)} calibration images from {args.calib_dir}", flush=True)
    q, reasons = quantize_fx_model(fp32, calib, per_channel=not args.per_tensor, observer=args.observer)
    census = summarize_modules(q)
    records = fp32_module_records(q, reasons)
    if (rec := silu_record(q)):
        records.append(rec)
    print("module census:", json.dumps(census))
    for r in records:
        print("  stays FP32:", json.dumps(r))
    method = (f"torch.ao FX graph mode (prepare_fx/convert_fx), qnnpack, {args.observer} activation "
              f"observer, {'per-tensor' if args.per_tensor else 'per-channel'} qint8 weights, "
              f"Conv+BN fused, {len(calib)} calibration images")
    art = build_artifact(q, args.model, records, method, n_images=len(calib),
                         extra_calibration={"census": census})
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(art, path)
    print(f"saved {path} ({os.path.getsize(path)} bytes, {art['quantized_compute_modules']} quantized "
          "compute modules)", flush=True)
    # a fresh process: what the Pi does with the file
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", args.model,
                        "--out-dir", args.out_dir, "--verify-only"], capture_output=True, text=True)
    print("verify (fresh process):", (r.stdout.strip().splitlines() or [r.stderr[-400:]])[-1])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
