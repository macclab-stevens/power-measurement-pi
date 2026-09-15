"""End-to-end capture: camera -> YOLO11n -> per-layer/per-op power dataset.

Usage: uv run src/run.py --frames 30 --bench 3
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

import cv2
import torch
import torch.profiler as profiler
import numpy as np

from cam import Camera
from detector import Detector
from layers import LayerProfiler
from microbench import bench_configs
from powersampler import PowerSampler

KEYBOARD_CLASS = 66
ACCEPT_MIN_CONF = 0.25
BASELINE_W_MIN, BASELINE_W_MAX = 0.5, 8.0


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _op_rows(prof: "profiler.profile") -> list[dict]:
    rows = []
    for e in prof.key_averages():
        c = e.count
        if c <= 0:
            continue
        rows.append({
            "op": e.key,
            "count": c,
            "self_total_us": round(e.self_cpu_time_total, 2),
            "self_mean_us": round(e.self_cpu_time_total / c, 2),
            "input_shapes": json.dumps(getattr(e, "input_shapes", []) or []),
            "module_path": "",  # aten-level table; module join is via layer_table.config_id
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--bench", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=10,
                    help="warm-up inference passes before measurement (thermal/DVFS stabilization)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pin-freq", action="store_true", help="requires sudo; restores governor (default off)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    outdir = args.out or f"runs/{ts}"
    os.makedirs(outdir, exist_ok=True)

    # human-friendly log: INFO to console, full detail to runs/<ts>/run.log
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
    get_logger = lambda: logger  # noqa: E731
    logger.info(f"run {ts} -> {outdir}")

    # ---- load model first so yolo11n.pt download/load is outside sampling ----
    det = Detector(args.weights, args.imgsz, args.conf)
    prof = LayerProfiler(det.model)
    logger.info(f"model '{args.weights}' loaded (torch={torch.__version__}, "
                f"ultralytics={__import__('ultralytics').__version__}); "
                f"hooked {len(prof._modules)} modules")

    sampler = PowerSampler()
    cam = Camera()

    if args.pin_freq:
        os.system("sudo cpupower frequency-set -g performance 2>/dev/null || sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")

    sampler.start()
    try:
        # ---- warm-up: sustained inference load so temp/DVFS reach steady
        # state BEFORE the idle baseline and timed measurements. Discards the
        # expensive first-inference transient (lazy init / graph fusion / JIT).
        if args.warmup > 0:
            sampler.marker("warmup")
            env0 = sampler.snapshot_env()
            warm_in = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(args.warmup):
                det.infer(warm_in)
            env1 = sampler.snapshot_env()
            logger.info(f"warm-up: {args.warmup} inference passes | "
                        f"temp {env0[0]}->{env1[0]}C | arm {env0[1]}->{env1[1]}MHz")

        # idle baseline (2 s), sanity-bounded
        sampler.marker("idle")
        t0 = sampler.now_ms()
        time.sleep(2.0)
        base = sampler.window(t0, sampler.now_ms())
        baseline_W = base["mean_P"]
        if baseline_W is None or not (BASELINE_W_MIN <= baseline_W <= BASELINE_W_MAX):
            raise SystemExit(
                f"baseline_W={baseline_W} outside [{BASELINE_W_MIN}, {BASELINE_W_MAX}] "
                "- PMIC parse or rail wrong; aborting"
            )
        logger.info(f"idle baseline = {baseline_W:.4f} W ({base['count']} samples)")

        cam_meta = cam.start()
        logger.info(f"camera started: {cam_meta}")

        # forward pass list + annotated frames
        lat_ms = []
        results = []
        best = {"conf": 0.0, "bgr": None, "cls": None}
        for f in range(args.frames):
            frame = next(cam.frames())
            sampler.marker("inference")
            prof.set_frame(f)
            t_in = time.perf_counter()
            if f == 0:
                with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                                      record_shapes=True) as p:
                    result = det.infer(frame)
                op_rows = _op_rows(p)
            else:
                result = det.infer(frame)
            t_out = time.perf_counter()
            lat_ms.append((t_out - t_in) * 1000.0)
            results.append(result)
            kc = Detector.max_conf(result)
            logger.info(f"  frame {f + 1}/{args.frames} detect={(t_out - t_in) * 1000:.0f} ms conf={kc:.3f}")
            if kc > best["conf"]:
                best["conf"] = kc
                best["bgr"] = Detector.plot(result)
                if result.boxes is not None and len(result.boxes.cls):
                    best["cls"] = result.names[int(result.boxes.cls[int(result.boxes.conf.argmax())])]

        cam.stop()
        sampler.marker("idle")
        fps = (args.frames - 1) / (sum(lat_ms[1:]) / 1000.0) if len(lat_ms) > 1 else 0.0
        logger.info(f"inference done: {args.frames} frames, mean fps={fps:.2f}")

        # ---- outputs --------------------------------------------------------
        cv2.imwrite(os.path.join(outdir, "annotated.jpg"), best["bgr"]) if best["bgr"] is not None else None

        layer_rows = prof.aggregated(exclude_frame=0)
        _write_csv(os.path.join(outdir, "layer_table.csv"),
                   ["path", "config_id", "class", "config_json", "leaf", "params", "macs",
                    "input_kind", "input_shapes", "out_shapes", "count",
                    "lat_mean_ms", "lat_median_ms", "lat_std_ms"], layer_rows)
        logger.info(f"layer profile: {len(layer_rows)} modules executed; "
                    f"{sum(1 for r in layer_rows if r['leaf'])} atomic (leaf) ops -> run.log for all")
        for r in layer_rows:  # full per-layer detail lands in run.log
            logger.debug(f"  layer path={r['path']} {r['class']} leaf={r['leaf']} "
                         f"params={r['params']} macs={r['macs']} "
                         f"lat_mean={r['lat_mean_ms']}ms cfg={r['config_id']}")

        _write_csv(os.path.join(outdir, "op_table.csv"),
                   ["op", "count", "self_total_us", "self_mean_us", "input_shapes",
                    "module_path"], op_rows)
        top_op = sorted(op_rows, key=lambda r: -float(r["self_total_us"]))[:3]
        logger.info(f"aten-op profile: {len(op_rows)} unique; top: " +
                    ", ".join(f"{o['op']} {o['count']}x {o['self_mean_us']}us" for o in top_op))
        for o in sorted(op_rows, key=lambda r: -float(r["self_total_us"])):
            logger.debug(f"  op {o['op']} count={o['count']} total={o['self_total_us']}us "
                         f"mean={o['self_mean_us']}us")

        # microbench (runs after inference; Detect anchors cached)
        achieved_hz = sampler.achieved_hz()
        logger.info(f"per-op power benches ({achieved_hz:.0f} W-Hz sampling) - one line per op:")
        perop = bench_configs(prof, sampler, baseline_W, args.bench,
                              require_samples=30, max_retries=2, log=logger.info)
        _write_csv(os.path.join(outdir, "perop_power.csv"),
                   ["config_id", "class", "config_json", "input_kind", "input_shapes",
                    "window_s", "N_calls", "t_per_call_ms", "P_mean_W", "P_delta_W",
                    "E_per_call_mJ", "temp_start_C", "temp_end_C",
                    "freq_start_MHz", "freq_end_MHz", "samples_used", "SKIPPED",
                    "skip_reason", "error"], perop)

        # samples.csv (raw, phase/window marked)
        srows = sampler.rows()
        _write_csv(os.path.join(outdir, "samples.csv"),
                   ["t_ms", "phase", "window_id", "I_A", "V_V", "P_W"], srows)
        # context.csv (rails + env every ~5 s)
        ctx = sampler.context()
        flat = []
        for c in ctx:
            row = {"t_ms": c.get("t_ms"), "temp_C": c.get("temp_C"),
                   "arm_MHz": c.get("arm_MHz"), "throttle_bits": c.get("throttle_bits")}
            for k, v in c.get("dump", {}).items():
                row[f"rail_{k}"] = v
            flat.append(row)
        _write_csv(os.path.join(outdir, "context.csv"),
                   ["t_ms", "temp_C", "arm_MHz", "throttle_bits"], flat)

        meta = {
            "model": args.weights, "imgsz": args.imgsz, "conf": args.conf,
            "threads": args.threads, "frames": args.frames, "warmup_frames": args.warmup,
            "mean_fps": round(fps, 3),
            "baseline_W": round(baseline_W, 6),
            "achieved_sample_hz": round(achieved_hz, 1),
            "throttle_bits": ctx[-1].get("throttle_bits") if ctx else None,
            "camera": cam_meta,
            "hostname": platform.node(), "kernel": platform.release(),
            "ts": ts, "outdir": outdir,
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
            "join_contract": "layer_table.config_id <-> perop_power.config_id; "
                             "op_table is aten-level timing (no module join); "
                             "all tables time-anchored by run ts; "
                             "E_mJ = P_W * t_ms; P_delta = P_mean - baseline_W",
            "methodology": "steady-state isolation microbench (deepcopy leaf module, "
                           "cache-hot inputs, VDD_CORE PMIC rail); per-op power is "
                           "attributed, not instantaneous",
        }
        with open(os.path.join(outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # ---- acceptance asserts ----------------------------------------------
        errors = []
        kb_best = best["conf"]
        if kb_best < ACCEPT_MIN_CONF:
            errors.append(f"no object detected with conf>={ACCEPT_MIN_CONF} (best={kb_best:.3f}) - is an object in view?")
        skipped = [r for r in perop if r["SKIPPED"]]
        if skipped:
            errors.append(f"{len(skipped)} SKIPPED perop rows: "
                          + ", ".join(f"{r['class']}:{r['config_id']}({r['skip_reason']})" for r in skipped))
        convs = [r for r in perop if r["class"] == "Conv2d"]
        for r in convs:
            if r["P_delta_W"] is None or r["P_delta_W"] <= 0.01:
                errors.append(f"Conv2d {r['config_id']}: P_delta_W={r['P_delta_W']}")
            if r["samples_used"] < 30:
                errors.append(f"Conv2d {r['config_id']}: samples_used={r['samples_used']}")
        if not op_rows:
            errors.append("op_table empty")
        if achieved_hz < 50:
            errors.append(f"achieved sample Hz={achieved_hz:.1f} < 50")
        if baseline_W is None or not (BASELINE_W_MIN <= baseline_W <= BASELINE_W_MAX):
            errors.append(f"baseline_W={baseline_W} out of range")

        logger.info(f"[summary] fps={fps:.2f} baseline_W={baseline_W:.4f} "
                    f"hz={achieved_hz:.0f} layer_rows={len(layer_rows)} op_rows={len(op_rows)} "
                    f"perop_rows={len(perop)} skipped={len(skipped)} "
                    f"best_conf={kb_best:.3f} ({best.get('cls', '-')})")
        if errors:
            for e in errors:
                logger.error(f"  ACCEPTANCE: {e}")
            raise SystemExit("ACCEPTANCE FAILED (see run.log)")
        logger.info(f"[ok] dataset written to {outdir}")
        sampler.stop()
    finally:
        sampler.stop()
        if args.pin_freq:
            os.system("sudo sh -c 'echo ondemand > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null")


if __name__ == "__main__":
    main()