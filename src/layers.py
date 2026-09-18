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

SCHEMA_VERSION = 3
_ALLOW = {
    "Conv2d": ("in_channels","out_channels","kernel_size","stride","padding","dilation","groups"),
    "Conv1d": ("in_channels","out_channels","kernel_size","stride","padding","dilation","groups"),
    "ConvTranspose2d": ("in_channels","out_channels","kernel_size","stride","padding","groups"),
    "Linear": ("in_features","out_features"),
    "Embedding": ("num_embeddings","embedding_dim","padding_idx"),
    "LayerNorm": ("normalized_shape","eps","elementwise_affine"),
    "BatchNorm2d": ("eps","momentum"),
    "MultiheadAttention": ("embed_dim","num_heads","batch_first","dropout"),
}

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


def module_config(mod):
    import torch.nn as nn
    cls = type(mod).__name__
    cfg = {"class": cls}
    for k in _ALLOW.get(cls, ()):
        if hasattr(mod, k):
            v = getattr(mod, k)
            cfg[k] = _ser(v)
    # booleans only, never values:
    if hasattr(mod, "bias"):
        b = getattr(mod, "bias")
        cfg["has_bias"] = b is not None
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
def _macs(mod, in_shapes, out_shapes):
    import torch.nn as nn
    if not in_shapes or not out_shapes:
        return None
    try:
        if isinstance(mod, nn.Conv2d):
            _, oc, oh, ow = out_shapes[0]
            return int(oh*ow*oc*mod.kernel_size[0]*mod.kernel_size[1]*mod.in_channels/max(1,mod.groups))
        # preserved pre-Task3 support (not in brief snippet): Conv1d/ConvTranspose2d
        if isinstance(mod, nn.Conv1d):
            _, oc, ol = out_shapes[0]
            return int(ol*oc*mod.kernel_size[0]*mod.in_channels/max(1,mod.groups))
        if isinstance(mod, nn.ConvTranspose2d):
            _, oc, oh, ow = out_shapes[0]
            return int(oh*ow*oc*mod.kernel_size[0]*mod.kernel_size[1]*mod.in_channels/max(1,mod.groups))
        if isinstance(mod, nn.Linear):
            import torch
            per = int(torch.tensor(out_shapes[0]).prod().item())
            return int(per*mod.in_features)
        if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d)):
            import torch
            return int(2*torch.tensor(out_shapes[0]).prod().item())
        if isinstance(mod, nn.Embedding):
            return int(in_shapes[0][0]*in_shapes[0][1]*mod.embedding_dim)  # gather cost proxy
        # SiLU/ReLU/GELU/Upsample/Concat/Pool: 1-2 FLOP per elem
        if type(mod).__name__ in ("SiLU","ReLU","GELU","Upsample","Concat","MaxPool2d","Identity"):
            import torch
            return int(torch.tensor(out_shapes[0]).prod().item())
    except Exception:
        return None
    return None


def _bytes_moved(in_args, out_shapes, out_dtype="torch.float32"):
    import torch
    b = sum(a.get("bytes",0) for a in _arg_tensors(in_args))
    per_elem = {"torch.float32":4,"torch.float16":2,"torch.bfloat16":2,"torch.float64":8,
                "torch.int8":1,"torch.uint8":1,"torch.int16":2,
                "torch.int64":8,"torch.int32":4,"torch.bool":1}.get(str(out_dtype),4)
    for s in out_shapes or []:
        n = 1
        for d in s: n *= d
        b += n*per_elem
    # add weight reads (cold):
    # caller adds params*4 for fp32
    return int(b)


def _out_dtype_for(rec, class_name):
    """Output dtype for byte accounting.

    The recorder only stores *input* dtypes, so the single-input-dtype case
    uses that dtype for outputs; Embedding (int64 idx in -> float out) and
    mixed-dtype leaves fall back to fp32.
    """
    if class_name == "Embedding":
        return "torch.float32"
    dtypes = sorted({a.get("dtype") for a in _arg_tensors(rec) if a.get("dtype")})
    if len(dtypes) == 1:
        return dtypes[0]
    return "torch.float32"


def _t_eff(class_name, in_shapes):
    # effective sequence/spatial length for scaling laws
    try:
        s = in_shapes[0]
        if class_name in ("Linear","LayerNorm","Embedding","MultiheadAttention"):
            return int(s[-2]) if len(s) >= 2 else 1  # T for [B,T,C]
        if class_name in ("Conv2d","BatchNorm2d","SiLU"):
            return int(s[-2]*s[-1]) if len(s) >= 4 else 1  # HW for [B,C,H,W]
    except Exception:
        pass
    return 1


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
                in_s = self.in_shapes(path, idx)
                out_s = self.out_shapes(path, idx)
                macs = _macs(self._modules[path], in_s, out_s)
                bm = _bytes_moved(rec, out_s,
                                  out_dtype=_out_dtype_for(rec, cfg["class"])) + self.params(path) * 4
                te = _t_eff(cfg["class"], in_s)
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
                    "t_eff": te,
                    "seq_len": te,
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
                out_dtype = _out_dtype_for(rec, cfg["class"])
                te = _t_eff(cfg["class"], in_s)
                wbytes = sum(p.numel() for p in mod.parameters()) * 4
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
                    "bytes_moved": _bytes_moved(rec, out_s, out_dtype=out_dtype) + wbytes,
                    "t_eff": te,
                    "seq_len": te,
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