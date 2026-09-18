"""Task-agnostic measurement core.

Loads a TaskAdapter (any PyTorch model family), warms up, captures an idle
baseline, then for every work mode runs N iterations while recording:
  - per-layer + per-op latency (hooks on every module; call-site-aware),
  - aten-op timing via torch.profiler on frame 0,
  - per-op steady-state power/energy via isolated microbenchmarks on the PMIC.

Outputs (runs/<ts>/): layer_table.csv, perop_power.csv, op_table.csv,
samples.csv, context.csv, run_meta.json. Same join contract as before, now
with generalized measurement identity (config_id = op+attrs+shapes+dtype) and
structural columns (path/callsite, mode) kept separate.

Usage:
  uv run src/run.py --task image --model yolo11n.pt --check-image bus.jpg
  uv run src/run.py --task lm --model minigpt --seq 64
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import sys
import time

import torch
import torch.profiler as profiler

from layers import LayerProfiler
from microbench import bench_configs
from powersampler import PowerSampler
from tasks import make_task

BASELINE_W_MIN, BASELINE_W_MAX = 0.5, 8.0
ACC_MIN_PD = 0.01   # per-op P_delta sanity (dominant op class)
MIN_SAMPLES = 30


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _op_rows(prof, mode: str) -> list[dict]:
    rows = []
    for e in prof.key_averages():
        c = e.count
        if c <= 0:
            continue
        rows.append({
            "op": e.key, "mode": mode, "count": c,
            "self_total_us": round(e.self_cpu_time_total, 2),
            "self_mean_us": round(e.self_cpu_time_total / c, 2),
            "input_shapes": json.dumps(getattr(e, "input_shapes", []) or []),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="image", help="image | lm (default image)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--bench", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seq", type=int, default=64, help="LM context length (prefill)")
    ap.add_argument("--check-image", default=None,
                    help="detection oracle: require object conf>=--conf on this image file")
    ap.add_argument("--live-camera", action="store_true",
                    help="feed real camera frames (cam.py) instead of synthetic fixtures")
    ap.add_argument("--pin-freq", action="store_true",
                    help="pin CPU to performance governor for consistent power measurements")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    outdir = args.out or f"runs/{ts}"
    os.makedirs(outdir, exist_ok=True)

    logger = logging.getLogger("powcap")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    _fh = logging.FileHandler(os.path.join(outdir, "run.log"))
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)
    logger.info(f"run {ts} -> {outdir}")

    if args.pin_freq:
        os.system("sudo cpupower frequency-set -g performance 2>/dev/null || "
                  "sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")

    task = make_task(args.task, args.model, args)
    prof = LayerProfiler(task.module)
    modes = task.modes()
    n_params = sum(p.numel() for p in task.module.parameters())
    logger.info(f"task='{args.task}' model='{args.model}' modes={modes} | "
                f"torch={torch.__version__} | hooked={len(prof._modules)} modules | "
                f"params={n_params/1e6:.2f}M")

    sampler = PowerSampler()
    sampler.start()
    try:
        # ---- warm-up: sustained inference over ALL modes so temp/DVFS settle
        # before the idle baseline and timed measurements.
        if args.warmup > 0:
            sampler.marker("warmup")
            env0 = sampler.snapshot_env()
            prof.set_mode("warmup")
            for _ in range(args.warmup):
                for m in modes:
                    task.run_one(m)
            env1 = sampler.snapshot_env()
            prof.exec_order.clear()  # op_sequence reflects measured modes only
            logger.info(f"warm-up: {args.warmup}x {modes} passes | "
                        f"temp {env0[0]}->{env1[0]}C | arm {env0[1]}->{env1[1]}MHz")

        # ABAB global baseline, slice A (2 s) — slice B follows the forward loops
        sampler.marker("baseline_A")
        tA0 = sampler.now_ms()
        time.sleep(2.0)
        baseA = sampler.window(tA0, sampler.now_ms())
        logger.info(f"baseline_A = {baseA['mean_P']} W ({baseA['count']} samples)")

        # ---- per-mode measurement -------------------------------------------
        op_rows = []
        for mode in modes:
            logger.info(f"== mode: {mode} ==")
            prof.set_mode(mode)
            sampler.marker(mode)  # tag phase so samples.csv carries measured mean-P
            for f in range(args.frames):
                prof.set_frame(f)
                if f == 0:
                    with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                                          record_shapes=True) as p:
                        task.run_one(mode)
                    op_rows += _op_rows(p, mode)
                else:
                    task.run_one(mode)
            logger.info(f"  mode {mode}: {args.frames} iterations done")

        # ABAB global baseline, slice B (2 s); the run baseline is mean(A, B)
        sampler.marker("baseline_B")
        tB0 = sampler.now_ms()
        time.sleep(2.0)
        baseB = sampler.window(tB0, sampler.now_ms())
        aP, bP = baseA["mean_P"], baseB["mean_P"]
        baseline_W = (aP + bP) / 2 if aP is not None and bP is not None else None
        baseline_drift_W = abs(aP - bP) if aP is not None and bP is not None else None
        if baseline_W is None or not (BASELINE_W_MIN <= baseline_W <= BASELINE_W_MAX):
            raise SystemExit(f"baseline_W={baseline_W} outside "
                             f"[{BASELINE_W_MIN}, {BASELINE_W_MAX}] - aborting")
        aS = f"{aP:.4f}" if aP is not None else "None"
        bS = f"{bP:.4f}" if bP is not None else "None"
        mS = f"{baseline_W:.4f}" if baseline_W is not None else "None"
        dS = f"{baseline_drift_W:.4f}" if baseline_drift_W is not None else "None"
        logger.info(f"ABAB baseline: A={aS}W B={bS}W mean={mS}W drift={dS}W")

        # ---- outputs ---------------------------------------------------------
        layer_rows = prof.aggregated(exclude_frame=0)
        _write_csv(os.path.join(outdir, "layer_table.csv"),
                   ["path", "config_id", "class", "config_json",
                    "input_shapes", "input_dtypes", "out_shapes", "callsite", "mode",
                    "leaf", "params", "macs", "bytes_moved", "arith_intensity",
                    "t_eff", "seq_len",
                    "count", "lat_mean_ms", "lat_median_ms", "lat_std_ms"], layer_rows)
        logger.info(f"layer profile: {len(layer_rows)} sites; "
                    f"{sum(1 for r in layer_rows if r['leaf'])} leaf configs")
        for r in layer_rows:
            if r["leaf"]:
                logger.debug(f"  leaf path={r['path']} {r['class']} mod={r['mode']} "
                             f"params={r['params']} macs={r['macs']} "
                             f"bytes={r['bytes_moved']} ai={r['arith_intensity']} "
                             f"lat={r['lat_mean_ms']}ms cfg={r['config_id']}")

        _write_csv(os.path.join(outdir, "op_table.csv"),
                   ["op", "mode", "count", "self_total_us", "self_mean_us",
                    "input_shapes"], op_rows)
        top_op = sorted(op_rows, key=lambda r: -float(r["self_total_us"]))[:3]
        logger.info(f"aten-op profile: {len(op_rows)} rows; top: " +
                    ", ".join(f"{o['op']}({o['mode']}) {o['count']}x {o['self_mean_us']}us"
                              for o in top_op))

        achieved_hz = sampler.achieved_hz()
        logger.info(f"per-op power benches ({achieved_hz:.0f} W-Hz sampling):")
        perop = bench_configs(prof, sampler, baseline_W, args.bench,
                              require_samples=MIN_SAMPLES, max_retries=2, log=logger.info)
        _write_csv(os.path.join(outdir, "perop_power.csv"),
                   ["config_id", "class", "config_json", "input_kind",
                    "input_shapes", "input_dtypes", "macs", "bytes_moved",
                    "t_eff", "seq_len",
                    "window_s", "N_calls", "t_per_call_ms", "P_mean_W",
                    "P_delta_W", "baseline_interp_W", "P_delta_ABAB_W",
                    "E_per_call_mJ", "temp_start_C", "temp_end_C",
                    "freq_start_MHz", "freq_end_MHz", "samples_used", "SKIPPED",
                    "skip_reason", "error"], perop)

        srows = sampler.rows()
        _write_csv(os.path.join(outdir, "samples.csv"),
                   ["t_ms", "phase", "window_id", "I_A", "V_V", "P_W"], srows)
        ctx = sampler.context()
        flat = []
        rail_keys: list[str] = []
        for c in ctx:
            for k in c.get("dump", {}):
                if f"rail_{k}" not in rail_keys:
                    rail_keys.append(f"rail_{k}")
        for c in ctx:
            row = {"t_ms": c.get("t_ms"), "temp_C": c.get("temp_C"),
                   "arm_MHz": c.get("arm_MHz"), "throttle_bits": c.get("throttle_bits")}
            for k, v in c.get("dump", {}).items():
                row[f"rail_{k}"] = v
            flat.append(row)
        _write_csv(os.path.join(outdir, "context.csv"),
                   ["t_ms", "temp_C", "arm_MHz", "throttle_bits"] + rail_keys, flat)

        # execution order (structural, per mode) -> run_meta
        op_seq: dict[str, list] = {}
        for (m, path) in prof.exec_order:
            op_seq.setdefault(m, [])
            if path not in op_seq[m]:
                op_seq[m].append(path)

        arch = {}
        for r in layer_rows:
            if r["leaf"]:
                arch[r["class"]] = arch.get(r["class"], 0) + 1

        meta = {
            "task": args.task, "model": args.model, "modes": modes,
            "imgsz": args.imgsz, "conf": args.conf, "seq_len": args.seq,
            "threads": args.threads, "frames_per_mode": args.frames,
            "warmup_passes": args.warmup, "params_M": round(n_params / 1e6, 3),
            "mean_fps": None,
            "baseline_W": round(baseline_W, 6) if baseline_W is not None else None,
            "baseline_A_W": round(aP, 6) if aP is not None else None,
            "baseline_B_W": round(bP, 6) if bP is not None else None,
            "baseline_drift_W": round(baseline_drift_W, 6) if baseline_drift_W is not None else None,
            "schema_version": 3,
            "achieved_sample_hz": round(achieved_hz, 1),
            "throttle_bits": ctx[-1].get("throttle_bits") if ctx else None,
            "arch_fingerprint": arch,
            "op_sequence": op_seq,
            "hostname": platform.node(), "kernel": platform.release(),
            "ts": ts, "outdir": outdir,
            "torch_version": torch.__version__,
            "join_contract": "measurement config_id = op+attrs+shape+dtype; "
                             "layer_table.config_id <-> perop_power.config_id; "
                             "path/callsite + mode are structural columns; "
                             "E_mJ = P_W * t_ms; P_delta = P_mean - baseline_W",
            "methodology": "steady-state isolation microbench (deepcopy leaf, "
                           "dtype-aware cache-hot inputs, VDD_CORE PMIC rail); "
                           "per-op power is attributed, not instantaneous",
        }
        with open(os.path.join(outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # ---- acceptance (task-gated, per-dominant-op) -------------------------
        errors = []
        skipped = [r for r in perop if r["SKIPPED"]]
        if skipped:
            errors.append(f"{len(skipped)} SKIPPED perop rows: "
                          + ", ".join(f"{r['class']}:{r['config_id']}({r['skip_reason']})"
                                      for r in skipped))
        # dominant op class by bench call volume
        dom = None
        vol = {}
        for r in perop:
            if r["N_calls"]:
                vol[r["class"]] = vol.get(r["class"], 0) + r["N_calls"]
        if vol:
            dom = max(vol, key=vol.get)
        logger.info(f"dominant op class for acceptance: {dom} "
                    f"(vol={vol.get(dom) if dom else 'n/a'})")
        if dom:
            for r in perop:
                if r["class"] == dom and not r["SKIPPED"]:
                    if r["P_delta_W"] is None or r["P_delta_W"] <= ACC_MIN_PD:
                        errors.append(f"{dom} {r['config_id']}: P_delta_W={r['P_delta_W']}")
                    if r["samples_used"] < MIN_SAMPLES:
                        errors.append(f"{dom} {r['config_id']}: samples={r['samples_used']}<{MIN_SAMPLES}")
        if not op_rows:
            errors.append("op_table empty")
        if achieved_hz < 50:
            errors.append(f"achieved sample Hz={achieved_hz:.1f} < 50")
        if baseline_W is None or not (BASELINE_W_MIN <= baseline_W <= BASELINE_W_MAX):
            errors.append(f"baseline_W={baseline_W} out of range")
        if baseline_drift_W is None:
            errors.append("baseline drift unknown (empty A/B window)")
        elif baseline_drift_W > 0.5:
            errors.append(f"baseline drift {baseline_drift_W:.3f}W > 0.5W")
        thr = ctx[-1].get("throttle_bits") if ctx else None
        if thr is not None and (thr & 0x1) != 0:
            errors.append(f"throttled during run (throttle_bits=0x{thr:x})")
        if not task.oracle():
            extra = getattr(task, "_checked", None)
            oracle_note = f" (best_conf={extra:.3f})" if extra is not None else ""
            errors.append(f"task oracle failed: model output invalid / no object"
                          f"{oracle_note}")

        logger.info(f"[summary] task={args.task} model={args.model} modes={modes} "
                    f"baseline_W={baseline_W:.4f} hz={achieved_hz:.0f} "
                    f"layer_rows={len(layer_rows)} op_rows={len(op_rows)} "
                    f"perop_rows={len(perop)} skipped={len(skipped)} "
                    f"dominant_op={dom}")
        if errors:
            for e in errors:
                logger.error(f"  ACCEPTANCE: {e}")
            raise SystemExit("ACCEPTANCE FAILED (see run.log)")
        logger.info(f"[ok] dataset written to {outdir}")
    finally:
        sampler.stop()
        try:
            task.stop_camera()
        except AttributeError:
            pass
        if args.pin_freq:
            os.system("sudo sh -c 'echo ondemand > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null")

if __name__ == "__main__":
    main()