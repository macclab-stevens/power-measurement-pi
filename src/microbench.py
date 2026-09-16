"""Per-operation steady-state power via isolated microbenchmarks.

For each unique leaf MEASUREMENT config (config_id = op + attrs + input
shapes + dtypes) found by LayerProfiler:
- replay inputs ONCE outside the timed loop (cache-hot; dtype-aware; zeros for
  int-index/bool-mask args so values are legal, empty for floats),
- deepcopy the actual module instance (identical semantics),
- loop it for ~bench_s, sampling VDD_CORE power the whole time,
- P_delta = P_mean - idle baseline; E_per_call = P_delta * t_per_call.

Windows below `require_samples` are retried (+1 s, up to 2 times). Any
config that cannot be benchmarked is a SKIPPED row — run.py treats SKIPPED as
fatal so a dataset is never silently missing its highest-value ops.

Input reconstruction supports tensor / list (Concat) / const / multi-tensor
args and int64/bool dtypes, so LM embedding indices, boolean masks and
nn.MultiheadAttention-style multi-tensor leaves replay instead of skipping.
"""
from __future__ import annotations

import copy
import time

import torch


def _alloc(shape, dtype, value):
    # dtype is stored as str(t.dtype) = "torch.float32" -> take the tail
    dt = getattr(torch, str(dtype).rsplit(".", 1)[-1]) if dtype else torch.float32
    if value == "zeros":
        return torch.zeros(tuple(shape), dtype=dt)
    return torch.empty(tuple(shape), dtype=dt)


def _replay_args(rec: dict) -> list:
    """Rebuild the exact positional arg list for one forward call."""
    out = []
    for a in rec.get("args", []):
        k = a["k"]
        if k == "tensor":
            out.append(_alloc(a["shape"], a.get("dtype"), a.get("value", "empty")))
        elif k == "const":
            out.append(a["v"])
        elif k == "list":
            sub = []
            for s in a.get("sub", []):
                if s.get("k") == "tensor":
                    sub.append(_alloc(s["shape"], s.get("dtype"), s.get("value", "empty")))
                else:
                    sub.append(s.get("v"))
            out.append(sub)
        elif k == "dict":
            out.append(a.get("v", {}))
        else:
            raise ValueError(f"unhandled arg kind {k}")
    return out


def _forward(mod, args: list):
    """list-kind modules (Concat) take one list arg, not splatted tensors."""
    if len(args) == 1 and isinstance(args[0], list):
        return mod(args[0])
    return mod(*args)


def _skip(entry: dict, reason: str, err: Exception) -> dict:
    return {
        "config_id": entry["config_id"],
        "class": entry["class"],
        "config_json": entry["config_json"],
        "input_kind": "multi",
        "input_shapes": entry["input_shapes"],
        "macs": entry.get("macs"), "bytes_moved": entry.get("bytes_moved"),
        "window_s": None, "N_calls": 0, "t_per_call_ms": None,
        "P_mean_W": None, "P_delta_W": None, "E_per_call_mJ": None,
        "temp_start_C": None, "temp_end_C": None,
        "freq_start_MHz": None, "freq_end_MHz": None,
        "samples_used": 0, "SKIPPED": True, "skip_reason": reason,
        "error": f"{type(err).__name__}: {err}",
    }


def bench_configs(profiler, sampler, baseline_W, bench_s, require_samples=30,
                  max_retries=2, log=None):
    """Benchmark every leaf config; returns list of perop_power rows."""
    rows = []
    items = list(profiler.leaf_configs().items())
    total = len(items)
    for i, (cid, entry) in enumerate(items, 1):
        if log:
            log(f"bench [{i}/{total}] {entry['class']} {entry['config_id']} "
                f"shapes={entry['input_shapes']}")
        rows.append(_bench_one(entry, sampler, baseline_W, bench_s,
                               require_samples, max_retries, log))
    return rows


def _bench_one(entry, sampler, baseline_W, bench_s, require_samples, max_retries, log=None):
    try:
        args = _replay_args(entry["input_kind"])
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"    input alloc failed: {e}")
        return _skip(entry, "input_alloc", e)

    try:
        mod = copy.deepcopy(entry["module"]).eval()
        with torch.no_grad():
            for _ in range(5):  # warmup (also primes lazy internal buffers)
                _forward(mod, args)
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"    SKIPPED (warmup): {e}")
        return _skip(entry, "warmup", e)

    target_ms = bench_s * 1000.0
    msec, retries = bench_s * 1000.0, 0
    sampler.marker(f"bench:{entry['config_id']}")
    w0 = sampler.now_ms()
    env0 = sampler.snapshot_env()
    try:
        while True:
            t_start = time.perf_counter()
            n = 0
            while sampler.now_ms() - w0 < msec:
                _forward(mod, args)
                n += 1
            t_end = time.perf_counter()
            stats = sampler.window(w0, sampler.now_ms())
            if stats["count"] >= require_samples or retries >= max_retries:
                break
            msec += 1000.0
            retries += 1
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"    SKIPPED (forward): {type(e).__name__}: {e}")
        return _skip(entry, "forward", e)
    finally:
        sampler.marker(None)
    env1 = sampler.snapshot_env()

    t_per_call = (t_end - t_start) * 1000.0 / n if n else None
    p_mean = stats["mean_P"]
    p_delta = (p_mean - baseline_W) if p_mean is not None else None
    if log:
        t0c, f0 = (env0[0], env0[1]) if env0[0] else (None, None)
        log(f"    done t={t_per_call:.3f}ms | P_delta={p_delta:.4f}W | "
            f"E={p_delta * t_per_call:.4f}mJ | N={n} | samples={stats['count']} | "
            f"temp {t0c}->{env1[0]}C | freq {f0}->{env1[1]}MHz" if t_per_call else
            f"    done (no calls)")
    return {
        "config_id": entry["config_id"],
        "class": entry["class"],
        "config_json": entry["config_json"],
        "input_kind": "multi",
        "input_shapes": entry["input_shapes"],
        "macs": (entry.get("macs") if entry.get("macs") is not None else None),
        "bytes_moved": entry.get("bytes_moved"),
        "window_s": round((sampler.now_ms() - w0) / 1000.0, 3),
        "N_calls": n,
        "t_per_call_ms": round(t_per_call, 4) if t_per_call else None,
        "P_mean_W": round(p_mean, 6) if p_mean is not None else None,
        "P_delta_W": round(p_delta, 6) if p_delta is not None else None,
        "E_per_call_mJ": round(p_delta * t_per_call, 6) if p_delta is not None and t_per_call else None,
        "temp_start_C": env0[0], "temp_end_C": env1[0],
        "freq_start_MHz": env0[1], "freq_end_MHz": env1[1],
        "samples_used": stats["count"],
        "SKIPPED": False, "skip_reason": "", "error": "",
    }