"""Per-layer and per-op capture via forward hooks on every module of the model.

Provides:
- layer_table rows: module path, config_id, params, MACs, shapes, per-frame
  latency stats (profiler frame 0 excluded).
- leaf_configs: the atomic (leaf) module instances keyed by config_id, ready
  for microbenchmarking. config_id identity includes in_channels AND input
  spatial shape so configurations with different compute are never collapsed.
- Stable hashed config_id across runs (a join key for training).
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time

import torch.nn as nn

_ATTRS = (
    "in_channels", "out_channels", "kernel_size", "stride", "padding",
    "dilation", "groups", "dim", "reg_max", "nc",
)


def _ser(v):
    if isinstance(v, (tuple, list)):
        return [_ser(x) for x in v]
    if isinstance(v, slice):
        return [v.start, v.stop, v.step]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


def module_config(mod: nn.Module) -> dict:
    cfg = {"class": type(mod).__name__}
    for n in _ATTRS:
        if hasattr(mod, n):
            cfg[n] = _ser(getattr(mod, n))
    return cfg


def config_id(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _tensor_shape(t):
    return list(t.shape)


def _input_record(inp) -> dict:
    """Return (kind, shapes) for a module's forward input tuple."""
    if len(inp) == 1 and isinstance(inp[0], (list, tuple)):
        return {"kind": "list", "shapes": [_tensor_shape(t) for t in inp[0]]}
    if len(inp) == 1 and hasattr(inp[0], "shape"):
        return {"kind": "tensor", "shapes": [_tensor_shape(inp[0])]}
    return {"kind": "other", "shapes": [_tensor_shape(x) for x in inp if hasattr(x, "shape")]}


def _output_record(out) -> dict:
    if isinstance(out, (list, tuple)):
        return {"kind": "list", "shapes": [_tensor_shape(x) for x in out if hasattr(x, "shape")]}
    if hasattr(out, "shape"):
        return {"kind": "tensor", "shapes": [_tensor_shape(out)]}
    return {"kind": "other", "shapes": []}


def _iter_inference_modules(model):
    """Yield (path, module) over the inference graph of any nn.Module.

    ultralytics YOLO exposes the DetectionModel at `.model`; raw nn.Module
    instances are walked directly. Paths are the named_modules paths.
    """
    base = getattr(model, "model", None)
    root = base if isinstance(base, nn.Module) else model
    for name, mod in root.named_modules():
        if not name:  # skip the root container itself
            continue
        yield name, mod


class LayerProfiler:
    def __init__(self, model) -> None:
        self.model = model
        self._start: dict[str, float] = {}
        self.frame = 0
        self.times: dict[str, dict[int, float]] = {}
        self.config: dict[str, dict] = {}        # path -> cfg
        self.config_id_map: dict[str, str] = {}   # path -> config_id
        self.input_rec: dict[str, dict] = {}
        self.output_rec: dict[str, dict] = {}
        self._modules: dict[str, nn.Module] = {}
        self._register()

    def _register(self) -> None:
        for name, mod in _iter_inference_modules(self.model):
            self._modules[name] = mod
            self.times.setdefault(name, {})
            mod.register_forward_pre_hook(self._make_pre(name))
            mod.register_forward_hook(self._make_post(name))

    def _make_pre(self, path: str):
        def pre(mod, inp):  # noqa: ARG001
            self._start[path] = time.perf_counter()
            if path not in self.input_rec:
                self.input_rec[path] = _input_record(inp)
                self.config[path] = module_config(mod)
                self.config_id_map[path] = config_id(self.config[path])
        return pre

    def _make_post(self, path: str):
        def post(mod, inp, out):  # noqa: ARG001
            dt = (time.perf_counter() - self._start.pop(path, time.perf_counter())) * 1000.0
            self.times[path][self.frame] = dt
            if path not in self.output_rec:
                self.output_rec[path] = _output_record(out)
        return post

    def set_frame(self, frame: int) -> None:
        self.frame = frame

    def is_leaf(self, path: str) -> bool:
        if path not in self.config:  # never executed (e.g. fused-away BN)
            return False
        prefix = path + "."
        return not any(p.startswith(prefix) for p in self._modules if p in self.config)

    def params(self, path: str) -> int:
        return sum(p.numel() for p in self._modules[path].parameters())

    def macs(self, path: str) -> int | None:
        mod = self._modules[path]
        if not isinstance(mod, nn.Conv2d):
            return None
        out = self.output_rec.get(path, {}).get("shapes")
        if not out:
            return None
        o = out[0]
        out_c, out_h, out_w = o[-3], o[-2], o[-1]
        k = mod.kernel_size[0]
        return int(out_h * out_w * out_c * k * k * mod.in_channels / mod.groups)

    # ---- aggregation ----------------------------------------------------------
    def aggregated(self, exclude_frame: int = 0):
        rows = []
        for path in self._modules:
            if path not in self.config:  # never executed (fused away / rebuilt graph)
                continue
            cfg = self.config[path]
            times = {f: t for f, t in self.times[path].items() if f != exclude_frame}
            ms = list(times.values())
            if not ms:
                continue
            row = {
                "path": path,
                "config_id": self.config_id_map[path],
                "class": cfg["class"],
                "config_json": json.dumps(cfg, sort_keys=True),
                "leaf": self.is_leaf(path),
                "params": self.params(path),
                "macs": self.macs(path),
                "input_kind": self.input_rec[path]["kind"],
                "input_shapes": json.dumps(self.input_rec[path]["shapes"]),
                "out_shapes": json.dumps(self.output_rec[path]["shapes"]),
                "count": len(ms),
                "lat_mean_ms": round(statistics.mean(ms), 4),
                "lat_median_ms": round(statistics.median(ms), 4),
                "lat_std_ms": round(statistics.pstdev(ms) if len(ms) > 1 else 0.0, 4),
            }
            rows.append(row)
        return rows

    def leaf_configs(self) -> dict[str, dict]:
        """config_id -> entry for atomic (leaf) modules that executed."""
        out = {}
        for path in self._modules:
            if not self.is_leaf(path):
                continue
            if path not in self.config:
                continue
            cid = self.config_id_map[path]
            entry = out.setdefault(cid, {
                "config_id": cid,
                "class": self.config[path]["class"],
                "config_json": json.dumps(self.config[path], sort_keys=True),
                "path": path,
                "module": None,
                "input_kind": None,
                "input_shapes": None,
                "calls": 0,
            })
            entry["module"] = self._modules[path]
            entry["input_kind"] = self.input_rec[path]["kind"]
            entry["input_shapes"] = self.input_rec[path]["shapes"]
            entry["calls"] += len([f for f in self.times[path] if f != 0])
        return out