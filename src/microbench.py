"""Per-operation steady-state power via isolated microbenchmarks.

For each unique leaf-module config (config_id) found by LayerProfiler:
- allocate inputs ONCE outside the timed loop (cache-hot steady state),
- deepcopy the actual module instance (identical semantics),
- loop it for ~bench_s, sampling VDD_CORE power the whole time,
- P_delta = P_mean - idle baseline; E_per_call = P_delta * t_per_call.

Windows below `require_samples` are retried (+1 s, up to 2 times). Any
module that cannot be benchmarked (input alloc or forward failure after
warmup) is emitted as a SKIPPED row — run.py treats SKIPPED as fatal so a
dataset is never silently missing its highest-value ops.
"""
from __future__ import annotations

import copy
import time

import torch


def _alloc_inputs(entry: dict) -> list:
    """Build fresh tensors for one forward call, matching recorded shapes/kind."""
    if entry["input_kind"] == "list":
        return [torch.empty(tuple(s), dtype=torch.float32) for s in entry["input_shapes"]]
    if entry["input_kind"] == "tensor":
        (s,) = entry["input_shapes"]
        return [torch.empty(tuple(s), dtype=torch.float32)]
    raise ValueError(f"unsupported input_kind={entry['input_kind']} for {entry['class']}")


def _forward(mod, entry: dict, ins: list):
    """Call the module with the input shape it actually receives:
    list-kind modules (e.g. Concat) take one list arg, not splatted tensors."""
    if entry["input_kind"] == "list":
        return mod(ins)
    return mod(*ins)


def _skip(entry: dict, reason: str, err: Exception) -> dict:
    return {
        "config_id": entry["config_id"],
        "class": entry["class"],
        "config_json": entry["config_json"],
        "input_kind": entry["input_kind"],
        "input_shapes": entry["input_shapes"],
        "window_s": None, "N_calls": 0, "t_per_call_ms": None,
        "P_mean_W": None, "P_delta_W": None, "E_per_call_mJ": None,
        "temp_start_C": None, "temp_end_C": None,
        "freq_start_MHz": None, "freq_end_MHz": None,
        "samples_used": 0, "SKIPPED": True, "skip_reason": reason,
        "error": f"{type(err).__name__}: {err}",
    }


def bench_configs(profiler, sampler, baseline_W, bench_s, require_samples=30,
                  max_retries=2, log=None):
    """Benchmark every leaf config; returns list of perop_power rows.

    `log(message)` (if given) emits one human-readable line per op so the run
    visibly reports which layer/op is being worked on and its measured power.
    """
    rows = []
    items = list(profiler.leaf_configs().items())
    total = len(items)
    for i, (cid, entry) in enumerate(items, 1):
        if log:
            log(f"bench [{i}/{total}] {entry['class']} {entry['config_id']} "
                f"shapes={entry['input_shapes']} kind={entry['input_kind']}")
        rows.append(_bench_one(entry, sampler, baseline_W, bench_s,
                               require_samples, max_retries, log))
    return rows


def _bench_one(entry, sampler, baseline_W, bench_s, require_samples, max_retries, log=None):
    try:
        ins = _alloc_inputs(entry)
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"    input alloc failed: {e}")
        return _skip(entry, "input_alloc", e)

    try:
        mod = copy.deepcopy(entry["module"]).eval()
        with torch.no_grad():
            for _ in range(5):  # warmup (also primes any lazy internal buffers)
                _forward(mod, entry, ins)
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"    SKIPPED (warmup): {e}")
        return _skip(entry, "warmup", e)

    # extend window until we have enough samples (degraded-Hz safety net)
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
                _forward(mod, entry, ins)
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
        "input_kind": entry["input_kind"],
        "input_shapes": entry["input_shapes"],
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