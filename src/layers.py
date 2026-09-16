"""Per-layer and per-op capture via forward hooks on ANY nn.Module.

Key contracts (post adversarial review):

* MEASUREMENT IDENTITY (config_id) = sha256(class, structural attrs, input
  shapes, input dtypes). It NEVER includes path/callsite, so two sites with the
  identical (op, attrs, shape, dtype) share one measurement and are bench-deduped.
  path / callsite / mode are SEPARATE structural columns for the dataset.

* CALL-SITE-AWARE TIMING: each path may fire multiple times per forward
  (shared/tied modules, sequential re-invocation, nesting). A per-path stack
  tags every invocation, so latency is never corrupted by the old single-slot
  `_start[path]` bug. Every call is recorded with its frame + mode + rec_idx.

* GENERALIZED INPUT CAPTURE: per forward argument we keep kind, shape, dtype and
  a replay spec (agnostic / const / captured), so multi-arg leaves (e.g.
  nn.MultiheadAttention q,k,v), int64 embedding indices and bool masks are all
  replayable by microbench instead of being SKIPPED-fatal.

* GENERALIZED MACs + BYTES: MACs for Conv2d/Conv1d/ConvTranspose2d/Linear from
  recorded shapes (+ None for exotic leaves, flagged), plus bytes_moved and
  arithmetic_intensity for every site (power tracks memory bandwidth).
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time

import torch
import torch.nn as nn

# Structural attrs snapshot per op kind (generalized beyond conv). Captured when
# present; absent attrs are simply omitted from the identity.
_ATTRS = (
    "in_channels", "out_channels", "kernel_size", "stride", "padding",
    "dilation", "groups", "dim", "reg_max", "nc",
    "in_features", "out_features", "bias",
    "num_embeddings", "embedding_dim", "padding_idx",
    "normalized_shape", "eps", "elementwise_affine",
    "num_heads", "embed_dim", "batch_first", "add_bias_kv",
    "max_len", "head_dim", "dropout",
)

DTYPE_BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
               torch.float64: 8, torch.int8: 1, torch.uint8: 1,
               torch.int16: 2, torch.int32: 4, torch.int64: 8,
               torch.bool: 1}


def _ser(v):
    if isinstance(v, (tuple, list)):
        return [_ser(x) for x in v]
    if isinstance(v, slice):
        return [v.start, v.stop, v.step]
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


def module_config(mod: nn.Module) -> dict:
    """Structural attrs for an op (class + present attrs only)."""
    cfg = {"class": type(mod).__name__}
    for n in _ATTRS:
        try:
            if hasattr(mod, n):
                cfg[n] = _ser(getattr(mod, n))
        except Exception:  # noqa: BLE001 - some modules raise in getattr
            pass
    return cfg


def config_id(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# input capture
# --------------------------------------------------------------------------
def _dt(t):
    return str(t.dtype)


def _tensor_spec(t):
    """Replay spec for a single tensor arg."""
    dt = t.dtype
    # int indices (embedding) and bool masks: kernels identical, values must be
    # in-range/legal -> allocate zeros, not empty.
    value = "zeros" if dt in (torch.int64, torch.int32, torch.bool) else "empty"
    return {"shape": list(t.shape), "dtype": _dt(t), "value": value,
            "bytes": int(t.numel()) * DTYPE_BYTES.get(dt, 0)}


def _const_spec(v):
    if isinstance(v, slice):
        return {"const": True, "py": "slice", "v": _ser(v)}
    if isinstance(v, (int, float, bool)) or v is None:
        return {"const": True, "py": type(v).__name__, "v": v}
    if isinstance(v, str):
        return {"const": True, "py": "str", "v": v}
    return None  # unknown object -> drop (won't be replayed)


def _input_record(inp) -> dict:
    """Distinct-input record for a module's forward args (positional + kwargs)."""
    args, kwargs = [], {}
    if inp is None:
        inp = ()
    if isinstance(inp, (tuple, list)):
        it = list(inp)
    else:
        it = [inp]
    for x in it:
        if hasattr(x, "shape"):            # tensor
            args.append({"k": "tensor", **{k: v for k, v in _tensor_spec(x).items()}})
        elif isinstance(x, (list, tuple)):  # nested list (e.g. Concat inputs)
            sub = []
            for y in _flatten(x):
                sub.append(dict(k="tensor", **{k2: v for k2, v in _tensor_spec(y).items()})
                           if hasattr(y, "shape")
                           else {"k": "const", "py": type(y).__name__, "v": _const_or(y)})
            args.append({"k": "list", "sub": sub})
        else:
            c = _const_spec(x)
            if c is not None:
                args.append({"k": "const", **c})
    return {"args": args, "kwargs": kwargs}


def _flatten(x):
    for item in x:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _const_or(v):
    return _ser(v) if not isinstance(v, (int, float, bool)) and v is not None else v


def _input_signature(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# module walk (any nn.Module; ultralytics exposes net at .model)
# --------------------------------------------------------------------------
def _iter_inference_modules(model):
    base = getattr(model, "model", None)
    root = base if isinstance(base, nn.Module) else model
    for name, mod in root.named_modules():
        if not name:
            continue
        yield name, mod


# --------------------------------------------------------------------------
# MACs / bytes
# --------------------------------------------------------------------------
def _macs(mod: nn.Module, in_shapes, out_shapes):
    """MACs from recorded shapes, per op class. None when unknown (flagged)."""
    if not in_shapes or not out_shapes:
        return None
    i0 = in_shapes[0]
    o0 = out_shapes[0]
    if isinstance(mod, (nn.Conv2d,)):
        try:
            _, oc, oh, ow = o0
            k = mod.kernel_size[0]
            g = mod.groups
            return int(oh * ow * oc * k * k * mod.in_channels / g)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(mod, (nn.Conv1d,)):
        try:
            _, oc, ol = o0
            k = mod.kernel_size[0]
            g = mod.groups
            return int(ol * oc * k * mod.in_channels / g)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(mod, (nn.ConvTranspose2d,)):
        try:
            _, oc, oh, ow = o0
            k = mod.kernel_size[0]
            g = mod.groups
            return int(oh * ow * oc * k * k * mod.in_channels / g)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(mod, nn.Linear) or (hasattr(mod, "in_features") and hasattr(mod, "out_features")):
        try:
            in_f = mod.in_features
            out_f = mod.out_features
            # output numel drives cost; drop spatial dims
            per = int(torch.tensor(o0, dtype=torch.int64).prod().item())
            tok = per // out_f if out_f else per
            return int(tok * in_f * out_f)
        except Exception:  # noqa: BLE001
            return None
    return None  # Embedding, MultiheadAttention internals, norms, exotic


def _bytes_moved(in_args, out_shapes):
    b = sum(a.get("bytes", 0) for a in _arg_tensors(in_args))
    for s in out_shapes or []:
        b += int(torch.tensor(s, dtype=torch.int64).prod().item()) * (4 if len(s) else 0)
    return b


def _arg_tensors(rec):
    for a in rec.get("args", []):
        if a["k"] == "tensor":
            yield a
        elif a["k"] == "list":
            for s in a.get("sub", []):
                if s.get("k") == "tensor":
                    yield s


class LayerProfiler:
    def __init__(self, model) -> None:
        self.model = model
        self.frame = 0
        self.mode = "forward"
        self._stack: dict[str, list] = {}           # path -> [(t0, rec_idx)]
        self.times: dict[str, list] = {}            # path -> [(frame, mode, rec_idx, dt_ms)]
        self.config: dict[str, dict] = {}           # path -> cfg
        self.input_recs: dict[str, list] = {}       # path -> [rec] (distinct)
        self.rec_idx: dict[str, dict] = {}          # path -> {sig: idx}
        self.calls: dict[str, list] = {}            # path -> [count per rec_idx]
        self.outputs: dict[str, list] = {}          # path -> [out_rec per rec_idx]
        self._modules: dict[str, nn.Module] = {}
        self.exec_order: list[tuple] = []           # (mode, path) first-execution order
        self._register()

    def _register(self) -> None:
        for name, mod in _iter_inference_modules(self.model):
            self._modules[name] = mod
            self._stack.setdefault(name, [])
            self.times[name] = []
            self.input_recs[name] = []
            self.rec_idx[name] = {}
            self.calls[name] = []
            self.outputs[name] = []
            mod.register_forward_pre_hook(self._make_pre(name))
            mod.register_forward_hook(self._make_post(name))

    def _match_rec(self, path, rec) -> int:
        sig = _input_signature(rec)
        m = self.rec_idx[path]
        idx = m.get(sig)
        if idx is None:
            idx = len(self.input_recs[path])
            m[sig] = idx
            self.input_recs[path].append(rec)
            self.outputs[path].append(None)
            self.calls[path].append(0)
        return idx

    def set_frame(self, frame: int) -> None:
        self.frame = frame

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _make_pre(self, path: str):
        def pre(mod, inp):  # noqa: ARG001
            if not (self.exec_order[-1] == (self.mode, path) if self.exec_order else False):
                self.exec_order.append((self.mode, path))
            rec = _input_record(inp)
            idx = self._match_rec(path, rec)
            if path not in self.config:
                self.config[path] = module_config(mod)
            self.calls[path][idx] += 1
            self._stack[path].append((time.perf_counter(), idx))
        return pre

    def _make_post(self, path: str):
        def post(mod, inp, out):  # noqa: ARG001
            if not self._stack[path]:
                return
            t0, idx = self._stack[path].pop()
            dt = (time.perf_counter() - t0) * 1000.0
            self.times[path].append((self.frame, self.mode, idx, dt))
            if self.outputs[path][idx] is None:
                self.outputs[path][idx] = _out_shapes(out)
        return post

    def is_leaf(self, path: str) -> bool:
        prefix = path + "."
        return not any(p.startswith(prefix) for p in self._modules)

    def params(self, path: str) -> int:
        return sum(p.numel() for p in self._modules[path].parameters())

    def in_shapes(self, path, idx):
        return [a["shape"] for a in self.input_recs[path][idx].get("args", [])
                if a["k"] == "tensor"] + \
               [s["shape"] for a in self.input_recs[path][idx].get("args", [])
                if a["k"] == "list" for s in a.get("sub", []) if s.get("k") == "tensor"]

    def out_shapes(self, path, idx):
        return self.outputs[path][idx] or []

    # ---------------------------------------------------------------- agg ----
    def aggregated(self, exclude_frame: int = 0):
        rows = []
        for path in self._modules:
            if path not in self.config:
                continue
            cfg = self.config[path]
            for idx, rec in enumerate(self.input_recs[path]):
                times = [t for (fr, mo, ri, t) in self.times[path]
                         if ri == idx and fr != exclude_frame]
                if not times and not self.calls[path][idx]:
                    continue
                meas_cfg = {**cfg,
                            "shapes": self.in_shapes(path, idx),
                            "dtypes": sorted({a.get("dtype") for a in
                                              _arg_tensors(rec) if a.get("dtype")})}
                mid = config_id(meas_cfg)
                out_s = self.out_shapes(path, idx)
                macs = _macs(self._modules[path], self.in_shapes(path, idx), out_s)
                bm = _bytes_moved(rec, out_s)
                rows.append({
                    "path": path,
                    # structural columns (NOT part of measurement identity)
                    "config_id": mid,
                    "class": cfg["class"],
                    "config_json": json.dumps(cfg, sort_keys=True),
                    "input_shapes": json.dumps(self.in_shapes(path, idx)),
                    "input_dtypes": json.dumps(sorted(
                        {a.get("dtype") for a in _arg_tensors(rec) if a.get("dtype")})),
                    "out_shapes": json.dumps(out_s),
                    "callsite": path, "mode": ",".join(sorted({mo for (fr, mo, ri, t)
                                                     in self.times[path]
                                                     if ri == idx and fr != exclude_frame})),
                    "leaf": self.is_leaf(path),
                    "params": self.params(path),
                    "macs": macs,
                    "bytes_moved": bm,
                    "arith_intensity": round(macs / bm, 4) if macs and bm else None,
                    "count": self.calls[path][idx],
                    "lat_mean_ms": round(statistics.mean(times), 4) if times else None,
                    "lat_median_ms": round(statistics.median(times), 4) if times else None,
                    "lat_std_ms": round(statistics.pstdev(times), 4) if len(times) > 1 else 0.0,
                })
        return rows

    def leaf_configs(self) -> dict:
        """measurement config_id -> bench entry (deduped by op+shape+dtype)."""
        out = {}
        for path in self._modules:
            if not self.is_leaf(path) or path not in self.config:
                continue
            mod = self._modules[path]
            cfg = self.config[path]
            for idx, rec in enumerate(self.input_recs[path]):
                if not self.calls[path][idx]:
                    continue
                in_s = self.in_shapes(path, idx)
                out_s = self.out_shapes(path, idx)
                meas_cfg = {**cfg, "shapes": in_s,
                            "dtypes": sorted({a.get("dtype") for a in
                                              _arg_tensors(rec) if a.get("dtype")})}
                mid = config_id(meas_cfg)
                entry = out.setdefault(mid, {
                    "config_id": mid,
                    "class": cfg["class"],
                    "config_json": json.dumps(cfg, sort_keys=True),
                    "input_kind": rec,
                    "input_shapes": in_s,
                    "input_dtypes": json.dumps(sorted(
                        {a.get("dtype") for a in _arg_tensors(rec) if a.get("dtype")})),
                    "module": mod,
                    "calls": 0,
                    "sites": [],
                    "macs": _macs(mod, in_s, out_s),
                    "bytes_moved": _bytes_moved(rec, out_s),
                })
                entry["calls"] += self.calls[path][idx]
                entry["sites"].append(path)
        return out


def _out_shapes(out) -> list:
    if isinstance(out, (list, tuple)):
        return [list(x.shape) for x in out if hasattr(x, "shape")]
    if hasattr(out, "shape"):
        return [list(out.shape)]
    return []