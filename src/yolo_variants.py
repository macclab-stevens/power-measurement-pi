"""Precision / compression variants of a YOLO model: ids, artifact loading, honesty checks.

Used by src/run.py (--quant, --quant-weights, --prune) and, for `variant_id`, by the NUC tools
(tools/quantize.py, prune.py, map_eval.py), so the top level imports only the standard library;
torch is imported inside the functions that need it.

  variant_id(model_path, quant, prune_ratio)   yolo11n | yolo11n_int8d | _int8s | _fp16 | _p30
  load_artifact(path, expected_format)         torch.load + format validation (dict artifacts)
  quantize_dynamic_counted(module)             int8-dynamic at load, with the replaced-module count
  count_quantized_modules(module)              quantized compute modules (Conv/Linear, fused too), no stubs
  quant_count_errors(quant, n)                 acceptance: never label FP32 data INT8
  executed_conv_dtype(layer_rows)              what the Conv2d leaves really ran in
  fp16_errors(layer_rows)                      acceptance: --quant fp16 must really run float16
  lookup_map_reference(path, variant, imgsz)   data/map_reference.json entry or None

  C2fSplitFree / C2PSASplitFree                split-free replacements of ultralytics C2f+C3k2 / C2PSA

Artifacts are pickled nn.Modules (torch.load weights_only=False). Any custom module class a
pruned/quantized artifact needs (the split-free blocks below, which pruned artifacts contain) must
live in THIS module so the pickle resolves on the Pi; `register()` is the hook run.py calls before
torch.load. The block classes are built on first attribute access (module `__getattr__`), so
importing this module still does not import torch.
"""
from __future__ import annotations

import json
import os
import re
import sys

QUANT_CHOICES = ("fp32", "int8-dynamic", "int8-static", "fp16")
QUANT_SUFFIX = {"fp32": "", "int8-dynamic": "int8d", "int8-static": "int8s", "fp16": "fp16"}
INT8S_FORMAT = "pi-power-int8s/1"
PRUNED_FORMAT = "pi-power-pruned/1"
# data/map_reference.json entry. `measured_on` = {hostname, machine (platform.machine()), torch,
# quantized_engine}: host/verify_cell.py warns unless machine == "aarch64" (the Pi). Entries for a
# weights artifact (models/*.pt) also carry `artifact_sha256`, so a regenerated artifact is detected.
MAP_REFERENCE_FIELDS = ("map50_95", "map50", "n_images", "dataset", "tool_version", "measured",
                        "measured_on")
MAP_REFERENCE_ARTIFACT_FIELDS = ("artifact_sha256",)


def _repair_quantized_conv_unpickling() -> None:
    """torch 2.14: the quantized conv `_ConvNd.__setstate__` restores its attributes without ever
    running nn.Module.__init__, so an unpickled quantized conv has no `_modules`/`_parameters`/hook
    dicts and `.eval()`, `.modules()` or unpickling an FX GraphModule that holds one raises
    AttributeError. Initialise the Module first. Idempotent; touches only unpickling."""
    import torch.nn as nn
    from torch.ao.nn.quantized.modules.conv import _ConvNd
    orig = _ConvNd.__setstate__
    if getattr(orig, "_pi_power_repaired", False):
        return

    def __setstate__(self, state):
        nn.Module.__init__(self)
        orig(self, state)

    __setstate__._pi_power_repaired = True
    _ConvNd.__setstate__ = __setstate__


def _split_conv(block, sizes):
    """One ultralytics Conv block (conv [+ bn] + act) -> one block per output-channel slice of
    `sizes`, each a deep copy holding that slice of the weights (and of the BN statistics), so
    cat(split_blocks(x)) == block(x) exactly: the 'split' becomes two convs instead of a `chunk`."""
    import copy
    import torch
    import torch.nn as nn
    assert sum(sizes) == block.conv.out_channels, (sizes, block.conv.out_channels)
    out, lo = [], 0
    for n in sizes:
        b = copy.deepcopy(block)
        sl = slice(lo, lo + n)
        b.conv.weight = nn.Parameter(block.conv.weight.detach()[sl].clone())
        if block.conv.bias is not None:
            b.conv.bias = nn.Parameter(block.conv.bias.detach()[sl].clone())
        b.conv.out_channels = n
        bn = getattr(b, "bn", None)           # fused blocks have no bn
        if bn is not None:
            src = block.bn
            for name in ("weight", "bias"):
                if getattr(src, name) is not None:
                    setattr(bn, name, nn.Parameter(getattr(src, name).detach()[sl].clone()))
            bn.running_mean = src.running_mean[sl].clone()
            bn.running_var = src.running_var[sl].clone()
            bn.num_features = n
        out.append(b)
        lo += n
    return out


_BLOCK_CLASSES = ("C2fSplitFree", "C2PSASplitFree")


def _keep_layer_attrs(new, old) -> None:
    """The attributes ultralytics' DetectionModel._predict_once reads off every layer."""
    for k in ("f", "i", "type", "np"):
        if hasattr(old, k):
            setattr(new, k, getattr(old, k))


def _build_block_classes() -> dict:
    import torch
    import torch.nn as nn

    class C2fSplitFree(nn.Module):
        """Ultralytics C2f / C3k2 without the fixed-width `chunk`: cv1 is two convs (cv1a, cv1b), one
        per half, so every tensor's width is a property of a layer and channel pruning can resize it.

        C3k2 is C2f with different bottlenecks in `m`, so one class covers both. `from_c2f` gives
        bit-identical outputs (same weights, sliced)."""

        def __init__(self, cv1a, cv1b, m, cv2):
            super().__init__()
            self.cv1a, self.cv1b, self.m, self.cv2 = cv1a, cv1b, m, cv2

        @classmethod
        def from_c2f(cls, c2f):
            half = c2f.c
            a, b = _split_conv(c2f.cv1, [half, half])
            new = cls(a, b, c2f.m, c2f.cv2)
            _keep_layer_attrs(new, c2f)
            return new

        def forward(self, x):
            y = [self.cv1a(x), self.cv1b(x)]
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    class C2PSASplitFree(nn.Module):
        """Ultralytics C2PSA with cv1 split into two convs instead of `.split((c, c), 1)`."""

        def __init__(self, cv1a, cv1b, m, cv2):
            super().__init__()
            self.cv1a, self.cv1b, self.m, self.cv2 = cv1a, cv1b, m, cv2

        @classmethod
        def from_c2psa(cls, c2psa):
            a, b = _split_conv(c2psa.cv1, [c2psa.c, c2psa.c])
            new = cls(a, b, c2psa.m, c2psa.cv2)
            _keep_layer_attrs(new, c2psa)
            return new

        def forward(self, x):
            return self.cv2(torch.cat((self.cv1a(x), self.m(self.cv1b(x))), 1))

    classes = {c.__name__: c for c in (C2fSplitFree, C2PSASplitFree)}
    for c in classes.values():
        c.__module__, c.__qualname__ = __name__, c.__name__
    return classes


def __getattr__(name):
    """PEP 562: build the split-free block classes on first use (torch is imported only then) and
    cache them as real module attributes, which is what pickle resolves on load."""
    if name in _BLOCK_CLASSES:
        globals().update(_build_block_classes())
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register() -> None:
    """Make artifacts loadable: run by load_artifact before torch.load (and by src/run.py).

    Builds the split-free block classes a pruned artifact pickles (they must resolve as
    `yolo_variants.<Class>`) and installs the quantized-conv unpickling repair (FX graph modules
    pickle as code and need nothing else)."""
    for name in _BLOCK_CLASSES:
        getattr(sys.modules[__name__], name)
    _repair_quantized_conv_unpickling()


def variant_id(model_path: str, quant: str = "fp32", prune_ratio: float | None = None) -> str:
    """Stable id of a (base model, precision, pruning) variant: `yolo11n`, `yolo11n_int8s`, `yolo11n_p30`.

    Quantization and pruning together are out of scope (ValueError)."""
    if quant not in QUANT_SUFFIX:
        raise ValueError(f"unknown quant {quant!r}; valid: {list(QUANT_CHOICES)}")
    if prune_ratio is not None and quant != "fp32":
        raise ValueError("quant and prune cannot be combined")
    base = os.path.basename(model_path)
    base = base[:-3] if base.endswith(".pt") else base
    if prune_ratio is not None:
        return f"{base}_p{round(prune_ratio * 100):d}"
    return f"{base}_{QUANT_SUFFIX[quant]}" if QUANT_SUFFIX[quant] else base


def validate_artifact(obj, expected_format: str, path: str = "<artifact>") -> dict:
    """The artifact dict if it is a dict of the expected format holding an nn.Module, else ValueError."""
    import torch.nn as nn
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a {expected_format!r} dict, got {type(obj).__name__}")
    fmt = obj.get("format")
    if fmt != expected_format:
        raise ValueError(f"{path}: format is {fmt!r}, expected {expected_format!r}")
    if not isinstance(obj.get("model"), nn.Module):
        raise ValueError(f"{path}: artifact 'model' is not an nn.Module")
    return obj


def load_artifact(path: str, expected_format: str) -> dict:
    """torch.load (weights_only=False: the artifacts are pickled modules) and validate the format."""
    import torch
    register()
    return validate_artifact(torch.load(path, map_location="cpu", weights_only=False),
                             expected_format, path)


_QUANT_COMPUTE_CLASSES = re.compile(
    r"^(Conv(Transpose)?[123]d|ConvReLU[123]d|ConvAdd(ReLU)?2d|Linear(ReLU|LeakyReLU|Tanh)?)$")


def count_quantized_modules(module) -> int:
    """Quantized COMPUTE modules: quantized Conv/ConvTranspose/Linear (and fused ConvReLU, ConvAdd,
    LinearReLU, ...) living in a torch quantized namespace (torch.ao.nn.quantized[.dynamic],
    torch.ao.nn.intrinsic.quantized). Quantize/DeQuantize/QFunctional stubs, quantized pooling and
    activations, a quantized module's own helpers (LinearPackedParams) and plain float modules do
    not count, nor do torch.ao.nn.quantized.reference modules (they compute in float): a model with
    stubs but float convs is FP32 and must not be labelled INT8."""
    t = type(module)
    if ".quantized.reference" in t.__module__:   # float compute wrapped in quant/dequant: not INT8
        return 0
    if ".quantized" in t.__module__:
        return 1 if _QUANT_COMPUTE_CLASSES.match(t.__name__) else 0
    return sum(count_quantized_modules(c) for c in module.children())


def quantize_dynamic_counted(module):
    """(quantized copy, replaced-module count): quantize_dynamic({nn.Linear}, qint8) on qnnpack.

    quantize_dynamic only covers Linear-type layers, so on a conv-heavy YOLO the count can be 0
    (the copy is then plain FP32; quant_count_errors turns that into an acceptance failure)."""
    import warnings
    import torch
    import torch.nn as nn
    torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():   # torch.ao.quantization deprecation noise, once per load
        warnings.simplefilter("ignore")
        q = torch.ao.quantization.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)
    return q, count_quantized_modules(q)


def quant_count_errors(quant: str, quantized_modules: int | None) -> list[str]:
    """Acceptance errors when an int8 variant actually executes no quantized module."""
    if quant not in ("int8-dynamic", "int8-static") or quantized_modules:
        return []
    return [f"{quant} quantized 0 modules: the model runs in float32 and must not be labelled INT8 "
            "(dynamic quantization only replaces Linear-type layers)"]


def _short_dtype(name: str) -> str:
    return name[len("torch."):] if name.startswith("torch.") else name


def _conv_leaf_dtypes(layer_rows) -> set[str]:
    dts: set[str] = set()
    for r in layer_rows:
        if r.get("class") == "Conv2d" and r.get("leaf"):
            dts.update(_short_dtype(d) for d in json.loads(r.get("input_dtypes") or "[]"))
    return dts


def executed_conv_dtype(layer_rows) -> str | None:
    """Input dtype(s) the Conv2d leaves executed in, e.g. 'float16', 'float32+float16'; None when
    there is no Conv2d leaf row. Read from the LayerProfiler rows (what really ran)."""
    dts = _conv_leaf_dtypes(layer_rows)
    return "+".join(sorted(dts)) if dts else None


def fp16_errors(layer_rows) -> list[str]:
    """Acceptance errors for --quant fp16: every Conv2d leaf must have executed in float16."""
    dts = _conv_leaf_dtypes(layer_rows)
    if not dts:
        return ["fp16 requested but no Conv2d leaf row to verify the executed dtype"]
    if dts != {"float16"}:
        return [f"fp16 requested but conv ran in {'+'.join(sorted(dts - {'float16'}))} "
                f"(Conv2d input dtypes seen: {sorted(dts)})"]
    return []


def lookup_map_reference(path: str, variant: str, imgsz: int) -> dict | None:
    """The data/map_reference.json entry for '<variant>@<imgsz>', or None (missing file, unreadable
    JSON or missing key): the Pi never fails a run over it, host/verify_cell.py warns."""
    try:
        with open(path) as f:
            table = json.load(f)
    except (OSError, ValueError):
        return None
    entry = table.get(f"{variant}@{imgsz}") if isinstance(table, dict) else None
    return entry if isinstance(entry, dict) else None
