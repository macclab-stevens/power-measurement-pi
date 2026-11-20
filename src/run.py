"""Task-agnostic measurement core.

Loads a TaskAdapter (any PyTorch model family), warms up, captures an idle
baseline, then for every work mode runs N iterations while recording:
  - per-layer + per-op latency (hooks on every module; call-site-aware),
  - aten-op timing via torch.profiler on frame 0,
  - per-op steady-state power/energy via isolated microbenchmarks on the PMIC.

Outputs (runs/<ts>/): layer_table.csv, perop_power.csv, op_table.csv,
samples.csv, context.csv, sysfs.csv, run_meta.json. Same join contract as before, now
with generalized measurement identity (config_id = op+attrs+shapes+dtype) and
structural columns (path/callsite, mode) kept separate.

--concurrent {<model>.pt,spinner} (image task) runs a second workload as a thread of this process
during the measured forward loop: one sync envelope, step events tagged m0 / m1 / spin
(src/coload.py), per-layer events off, run_meta["concurrent"] with the thread split and per-tag
iteration counts and latencies.

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

import coload
import perfcounters
import pinning
import yolo_variants
from layers import LayerProfiler
from microbench import bench_configs
from powersampler import make_sampler
from sysfs_sampler import SysfsSampler, ThermalAbort, thermal_exit_code
from tasks import make_task

BASELINE_W_MIN, BASELINE_W_MAX = 0.5, 8.0
ACC_MIN_PD = 0.01   # per-op P_delta sanity (dominant op class)
MIN_SAMPLES = 30
MONSOON_MIN_HZ = 4900   # recorder sample_hz floor in monsoon mode (nominal 5000)
PMIC_MIN_HZ = 50
MAP_REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                  "map_reference.json")


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


# v2 flags and their v1 (default) values. getattr is used for each so a flag added by a later
# task (--concurrent) joins the rule by existing on args; adjust the default here if that task's
# default is not None. Special cases handled in dataset_version():
# --holdout (v2 whenever given at all, even "none"), --fan (only auto/off count; "full" is the
# legacy --fan-full). --perf-counters is deliberately absent: it is on by default and a
# counters-only run is still a v1-shaped dataset. threads/batch/imgsz never count by themselves.
V2_FLAG_DEFAULTS = {
    "governor": "pinned", "serve_fps": None, "soak_minutes": None,
    "frames_source": "repeat", "input_image": None,
    "quant": "fp32", "prune": None, "concurrent": None,
}
V2_FAN_MODES = ("auto", "off")


def dataset_version(args) -> int:
    """2 if --holdout was given (any value) or any other v2 workload flag is non-default."""
    if getattr(args, "holdout", None) is not None:
        return 2
    if getattr(args, "fan", None) in V2_FAN_MODES:
        return 2
    for name, default in V2_FLAG_DEFAULTS.items():
        v = getattr(args, name, default)
        if v is not None and v != default:
            return 2
    return 1


def restore_fan_if_off(fan_mode: str, apply_mode=pinning.apply_mode, log=None) -> None:
    """A fan-off cell must never leave the fan off: back to auto, best effort (errors logged)."""
    if fan_mode != "off":
        return
    log = log or logging.getLogger("powcap").error
    try:
        apply_mode(None, "auto", "pinned")
    except Exception as e:
        log(f"fan not restored to auto: {e}")


def sync_pulse_pinned(sampler, tag: str, governor: str, cpu_mhz: int | None,
                      pin=pinning.pin, unpin=pinning.unpin, sleep=time.sleep) -> None:
    """Emit a sync-pulse train. With --governor ondemand the clock is pinned to cpu_mhz only
    around the train (pin, settle 0.5 s, pulses, unpin - unpin runs even if the pulses raise)."""
    if governor != "ondemand":
        sampler.sync_pulse(tag)
        return
    pin(cpu_mhz)
    try:
        sleep(0.5)   # settle at the pinned clock before the pulses
        sampler.sync_pulse(tag)
    finally:
        unpin()


def resolve_fan_mode(fan: str | None, fan_full: bool) -> str:
    """Fan mode for pinning.apply_mode: --fan wins; legacy --fan-full means full; else none."""
    if fan is not None:
        return fan
    return "full" if fan_full else "none"


def resolve_batch(task: str, batch: int | None) -> int:
    """Effective batch: the given value, else 8 for training and 1 for everything else."""
    if batch is not None:
        return batch
    return 8 if task == "train" else 1


def apply_threads(n: int) -> dict:
    """Make --threads drive torch, OpenCV and OpenMP; returns what each reports back."""
    import cv2
    os.environ["OMP_NUM_THREADS"] = str(n)   # before the first torch parallel op
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    return {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads(),
            "omp_env": os.environ["OMP_NUM_THREADS"]}


def serve_wait(t_serve0: float, k: int, fps: float, now: float) -> tuple[float, bool]:
    """Before request k (k >= 1) due at t_serve0 + k/fps: (seconds to idle, missed deadline).

    A request that is already late gets no idle time and counts as missed."""
    remaining = t_serve0 + k / fps - now
    return max(0.0, remaining), remaining < 0.0


def latency_stats(ms: list[float]) -> dict | None:
    """mean / median / p95 (linear interpolation) of per-iteration wall times, or None."""
    if not ms:
        return None
    xs = sorted(ms)
    pos = 0.95 * (len(xs) - 1)
    lo = int(pos)
    p95 = xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)
    mid = len(xs) // 2
    median = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
    return {"mean": round(sum(xs) / len(xs), 3), "median": round(median, 3),
            "p95": round(p95, 3), "n": len(xs)}


def measured_loop(task, prof, mode, frames, soak_minutes, check_thermal,
                  clock=time.perf_counter, on_first_frame=None):
    """One measured mode: frame 0 under the op profiler, then frames 1.. timed one by one.

    Runs `frames` iterations, or (soak_minutes given) until that many minutes have elapsed.
    `on_first_frame()` (--concurrent: release the second workload) runs once, right after
    frame 0 so the profiled frame is the primary's alone.
    Returns (frames_done, per-iteration ms for frames >= 1, aten op rows)."""
    op_rows: list[dict] = []
    lat_ms: list[float] = []
    deadline = None if soak_minutes is None else clock() + soak_minutes * 60.0
    f = 0
    while f < frames if deadline is None else (f == 0 or clock() < deadline):
        check_thermal()
        prof.set_frame(f)
        if f == 0:
            # torch.profiler start/stop takes ~1 s on the first mode; label it so it is
            # not mistaken for unexplained time (the task's own steps nest inside)
            with task.step("profiler"):
                with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                                      record_shapes=True) as p:
                    task.run_one(mode)
                op_rows += _op_rows(p, mode)
            if on_first_frame is not None:
                on_first_frame()
        else:
            t = clock()
            task.run_one(mode)
            lat_ms.append((clock() - t) * 1000.0)
        f += 1
    return f, lat_ms, op_rows


def end_concurrent(worker, marker) -> None:
    """The primary's measured loop is over: label the drain (the worker finishing its current
    iteration) as its own phase so it is not charged to the primary's last frame, then stop the
    worker (raises if it will not stop) and re-raise a worker failure."""
    marker("concurrent_drain")
    worker.stop()
    worker.check()


def concurrent_block(args, split, worker, frames_done, primary_lat, load2_s,
                     serve_concurrent: bool) -> dict:
    """run_meta["concurrent"]: how the second workload ran and what each tag did.

    m0 frame 0 runs alone (op profiler); m0 frames 1.. overlap with the worker."""
    tag = worker.tag
    n0 = frames_done.get("forward", 0)
    return {
        "mode": "spinner" if args.concurrent == coload.SPINNER else "model",
        "model": None if args.concurrent == coload.SPINNER else args.concurrent,
        "tags": [coload.PRIMARY_TAG, tag],
        "threads_split": split,
        "worker": "thread of the run.py process (counted by perf -p: created and warmed up "
                  "before perf attaches)",
        "sync_envelopes": 1,
        "layer_events": False,
        "released": "after primary frame 0 (the profiled frame runs alone); worker stops when "
                    "the primary's measured loop ends",
        "serve_concurrent": serve_concurrent,
        "stop_join_s": worker.stop_join_s,
        "iterations": {coload.PRIMARY_TAG: n0, tag: worker.iterations},
        "overlapped_primary_frames": max(0, n0 - 1),
        "latency_ms": {coload.PRIMARY_TAG: primary_lat,
                       tag: latency_stats(worker.lat_ms)},
        "first_call_ms": {tag: worker.first_call_ms},
        "load_weights_s": {tag: round(load2_s, 4) if load2_s is not None else None},
        "spinner": ({"dim": coload.SPIN_DIM, "mm_per_step": coload.SPIN_MM_PER_STEP}
                    if args.concurrent == coload.SPINNER else None),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="image", help="image | lm | train (default image)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--bench", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4,
                    help="sets torch, OpenCV and OMP_NUM_THREADS (default 4)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seq", type=int, default=64, help="LM context length (prefill)")
    ap.add_argument("--batch", type=int, default=None,
                    help="batch size: train default 8; image default 1 (B stacked frames per "
                         "forward); ignored by lm")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                    help="training optimizer (--task train)")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the per-op isolation benches (perop_power.csv header only)")
    ap.add_argument("--serve", type=int, default=0,
                    help="after the modes, N single requests with idle gaps (phase 'serve')")
    ap.add_argument("--serve-gap-ms", type=float, default=200.0,
                    help="idle time between serve requests (step 'idle_between_requests')")
    ap.add_argument("--serve-fps", type=float, default=None,
                    help="with --serve: request k is due at t0 + k/F (the gap is the time left; "
                         "a late request starts at once and counts as a missed deadline); "
                         "replaces --serve-gap-ms")
    ap.add_argument("--soak-minutes", type=float, default=None,
                    help="image task: loop the measured forward mode for M minutes instead of "
                         "--frames (forces --no-bench)")
    ap.add_argument("--check-image", default=None,
                    help="detection oracle: require object conf>=--conf on this image file")
    ap.add_argument("--input-image", default=None,
                    help="frame fed to the model (default: --check-image); the oracle still "
                         "uses --check-image")
    ap.add_argument("--frames-source", default="repeat", choices=["repeat", "stream"],
                    help="repeat one frame, or cycle through the sorted *.jpg of --stream-dir "
                         "(decoded once at load)")
    ap.add_argument("--stream-dir", default="inputs/stream50",
                    help="image folder for --frames-source stream")
    ap.add_argument("--live-camera", action="store_true",
                    help="feed real camera frames (cam.py) instead of synthetic fixtures")
    ap.add_argument("--quant", default="fp32", choices=list(yolo_variants.QUANT_CHOICES),
                    help="precision variant of a YOLO model: fp32 (default) | int8-dynamic "
                         "(Linear layers only, qnnpack) | int8-static (needs --quant-weights) | "
                         "fp16 (module and input .half(); executed dtype verified)")
    ap.add_argument("--quant-weights", default=None,
                    help="int8-static artifact (pi-power-int8s/1 dict from tools/quantize_yolo.py)")
    ap.add_argument("--prune", default=None,
                    help="pruned model artifact (pi-power-pruned/1 dict from tools/prune_yolo.py); "
                         "cannot be combined with --quant other than fp32")
    ap.add_argument("--concurrent", default=None, metavar="MODEL.pt|spinner",
                    help="image task: run a second workload in the same process during the "
                         "measured forward loop - a second model of the same imgsz/batch/input "
                         "(tag m1, threads split --threads//2 each) or 'spinner' (one thread "
                         "of matrix multiplies, tag spin; the primary keeps --threads-1). One "
                         "sync envelope, step events tagged m0/m1/spin, no per-layer events. "
                         "Marks the run dataset_version 2")
    ap.add_argument("--cpu-mhz", type=int, default=None,
                    help="pin all cores to this clock via pi-power-pin (1500..2400 step 100)")
    ap.add_argument("--fan-full", action="store_true",
                    help="force the fan to max state (constant fan power); same as --fan full")
    ap.add_argument("--fan", default=None, choices=["full", "auto", "off"],
                    help="fan mode via pi-power-pin: full | auto (stock trips) | off (85 C "
                         "backstop). Default: untouched unless --fan-full")
    ap.add_argument("--governor", default="pinned", choices=["pinned", "ondemand"],
                    help="pinned: --cpu-mhz held all run. ondemand: DVFS for the workload, "
                         "--cpu-mhz (required) pinned only around the sync-pulse trains")
    ap.add_argument("--holdout", default=None, choices=["primary-s", "secondary-m", "none"],
                    help="holdout tag recorded in run_meta; giving --holdout at all (even "
                         "'none') marks the run dataset_version 2")
    ap.add_argument("--sampler", default="pmic", choices=["pmic", "monsoon"],
                    help="power source: on-board PMIC (legacy) or EliteBook Monsoon recorder")
    ap.add_argument("--recorder-url", default="http://192.168.0.11:8051",
                    help="Monsoon recorder HTTP base URL (used with --sampler monsoon)")
    ap.add_argument("--perf-counters", default="on", choices=["on", "off"],
                    help="hardware counters via perf stat -I (counters.csv); forced off with "
                         "--sampler pmic")
    ap.add_argument("--perf-interval-ms", type=int, default=100, choices=[100, 200],
                    help="perf stat sampling interval (100 = 10 Hz, 200 = 5 Hz)")
    return ap


def parse_args(argv=None):
    """build_parser().parse_args(argv) plus the cross-flag checks (argparse errors, exit 2)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.fan_full and args.fan not in (None, "full"):
        ap.error(f"--fan-full conflicts with --fan {args.fan}")
    if args.governor == "ondemand":
        if args.cpu_mhz is None:
            ap.error("--governor ondemand needs --cpu-mhz (the clock pinned around sync pulses)")
        if args.cpu_mhz not in pinning.ALLOWED_MHZ:
            ap.error(f"--cpu-mhz {args.cpu_mhz} not in "
                     f"{pinning.ALLOWED_MHZ[0]}..{pinning.ALLOWED_MHZ[-1]} step 100")
    if args.serve_fps is not None:
        if args.serve_fps <= 0:
            ap.error("--serve-fps must be > 0")
        if args.serve <= 0:
            ap.error("--serve-fps needs --serve N")
    if args.soak_minutes is not None:
        if args.soak_minutes <= 0:
            ap.error("--soak-minutes must be > 0")
        if args.task != "image":
            ap.error("--soak-minutes is only supported with --task image")
    if args.batch is not None and args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.quant == "int8-static" and not args.quant_weights:
        ap.error("--quant int8-static needs --quant-weights PATH")
    if args.quant_weights and args.quant != "int8-static":
        ap.error("--quant-weights is only used with --quant int8-static")
    if args.prune and args.quant != "fp32":
        ap.error("--prune cannot be combined with --quant other than fp32")
    if (args.prune or args.quant != "fp32") and args.task != "image":
        ap.error("--quant / --prune are only supported with --task image")
    if args.frames_source == "stream" and (args.live_camera or args.input_image):
        ap.error("--frames-source stream conflicts with --live-camera / --input-image")
    if args.concurrent is not None:
        if args.concurrent != coload.SPINNER and not args.concurrent.endswith(".pt"):
            ap.error(f"--concurrent {args.concurrent!r}: expected a <model>.pt or 'spinner'")
        if args.task != "image":
            ap.error("--concurrent is only supported with --task image")
        if args.soak_minutes is not None:
            ap.error("--concurrent cannot be combined with --soak-minutes")
        if args.live_camera:
            ap.error("--concurrent cannot be combined with --live-camera")
        if args.prune or args.quant != "fp32":
            ap.error("--concurrent cannot be combined with --quant / --prune")
    return args


def main():
    args = parse_args()
    bmin, bmax = (1.5, 8.0) if args.sampler == "monsoon" else (BASELINE_W_MIN, BASELINE_W_MAX)

    split = coload.thread_split(args.concurrent, args.threads) if args.concurrent else None
    # concurrent: torch's thread count is per calling thread; this is the primary's (main thread)
    # share, the worker thread sets its own
    threads_detail = apply_threads(split[coload.PRIMARY_TAG] if split else args.threads)
    args.batch = resolve_batch(args.task, args.batch)
    fan_mode = resolve_fan_mode(args.fan, args.fan_full)
    soak = args.soak_minutes is not None
    if soak:
        args.no_bench = True   # benches would make a soak cell unbounded
    bench_forced_off = None
    if args.quant != "fp32":
        args.no_bench = True   # isolation benches deep-copy leaves and synthesize fp32 inputs
        bench_forced_off = "quant"
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

    pin_status_start = pinning.status()   # before applying: exposes a stale pin from a prior run
    logger.info(f"pin status at start: {pin_status_start}")
    if soak:
        logger.info(f"--soak-minutes {args.soak_minutes}: per-op benches forced off")
    if bench_forced_off:
        logger.info(f"--quant {args.quant}: per-op benches forced off")

    try:
        pin_status = pinning.apply_mode(args.cpu_mhz, fan_mode, args.governor)
        if fan_mode == "full":
            pinning.wait_fan_full()
        if pin_status:
            logger.info(f"pinned: {pin_status}")

        sampler = make_sampler(args.sampler, args.recorder_url)
        monsoon = args.sampler == "monsoon"
        sampler.start()
        sysfs = SysfsSampler()
        perf_on = args.perf_counters == "on" and monsoon
        if args.perf_counters == "on" and not monsoon:
            logger.info("perf counters forced off (--sampler pmic)")
        perf = perfcounters.PerfStat(os.getpid(), interval_ms=args.perf_interval_ms,
                                     out_path=os.path.join(outdir, "counters.raw.txt"))
    except BaseException:
        restore_fan_if_off(fan_mode)   # the main try/finally below is not entered yet
        raise
    phase = {"name": "init"}   # current phase, recorded in thermal_abort.json

    def _check_thermal():
        """Frame/pass/request-boundary check: raise if the 1 Hz watchdog saw >= 85 C."""
        if sysfs.thermal_abort is not None:
            raise ThermalAbort(sysfs.thermal_abort)

    def _thermal_exit(details):
        """Record thermal_abort.json and return the exit code (3 abort / 4 hard stop)."""
        rec = dict(details, phase=phase["name"])
        with open(os.path.join(outdir, "thermal_abort.json"), "w") as f:
            json.dump(rec, f, indent=2)
        logger.error(f"THERMAL {rec['level'].upper()}: {rec['temp_C']} C in phase "
                     f"'{rec['phase']}' (limits 85 abort / 90 hard stop)")
        return thermal_exit_code(details)

    def _check_loop():
        """_check_thermal plus, in a concurrent run, a dead-worker check."""
        _check_thermal()
        if worker is not None:
            worker.check()

    thr0 = sampler.snapshot_env()[2]
    task = None
    worker = None   # --concurrent second workload (coload.SecondWorkload)
    worker_task = None
    load2_s = None
    try:
        sysfs.start()   # first statement in the try: a hard stop on the very first read is handled
        # ---- lifecycle: model load and the first call of each mode are measured phases
        phase["name"] = "load_weights"
        sampler.marker("load_weights")
        t_load = time.perf_counter()
        task = make_task(args.task, args.model, args)
        load_s = time.perf_counter() - t_load
        variant = getattr(task, "variant", None) or {}
        if variant.get("quantized_modules") == 0:
            logger.error(f"--quant {args.quant}: 0 quantized modules, this run will fail "
                         "acceptance (float32 must not be labelled INT8)")
        prof = LayerProfiler(task.module)
        modes = task.modes()
        n_params = sum(p.numel() for p in task.module.parameters())
        logger.info(f"task='{args.task}' model='{args.model}' modes={modes} | "
                    f"torch={torch.__version__} | hooked={len(prof._modules)} modules | "
                    f"params={n_params/1e6:.2f}M | load {load_s:.2f}s")
        if monsoon:
            prof.events = sampler.events
            if split:
                # two threads share one event log: step events are tagged and depth-free, the
                # per-layer begin/end events (shared depth counter) are off; labels stay on
                prof.layer_events = False
                task.step = sampler.events.step_for(coload.PRIMARY_TAG)
            else:
                task.step = sampler.events.step
        if split:
            wtag = coload.worker_tag(args.concurrent)
            if args.concurrent != coload.SPINNER:
                t_load2 = time.perf_counter()
                worker_task = make_task("image", args.concurrent, args)
                load2_s = time.perf_counter() - t_load2
                logger.info(f"concurrent model '{args.concurrent}' loaded in {load2_s:.2f}s")
            worker = coload.SecondWorkload(
                wtag, split[wtag], sampler.events.step_for(wtag) if monsoon else task.step,
                task=worker_task, warmup=args.warmup)
            logger.info(f"concurrent: {args.concurrent} as tag {wtag}; torch threads {split}")
        first_ms = {}
        for m in modes:
            phase["name"] = f"first:{m}"
            _check_thermal()
            sampler.marker(f"first:{m}")
            prof.set_mode(f"first:{m}")
            t1 = time.perf_counter()
            task.run_one(m)
            first_ms[m] = round((time.perf_counter() - t1) * 1000, 3)
        logger.info(f"first call per mode (ms): {first_ms}")

        # ---- warm-up: sustained inference over ALL modes so temp/DVFS settle
        # before the idle baseline and timed measurements.
        if worker is not None:
            worker.start()   # its first call + warm-up run beside the primary's warm-up
        if args.warmup > 0:
            phase["name"] = "warmup"
            sampler.marker("warmup")
            env0 = sampler.snapshot_env()
            prof.set_mode("warmup")
            for _ in range(args.warmup):
                _check_thermal()
                for m in modes:
                    task.run_one(m)
            env1 = sampler.snapshot_env()
            prof.exec_order.clear()  # op_sequence reflects measured modes only
            logger.info(f"warm-up: {args.warmup}x {modes} passes | "
                        f"temp {env0[0]}->{env1[0]}C | arm {env0[1]}->{env1[1]}MHz")
        if worker is not None:
            # every worker thread (and its OpenMP team) must exist before perf attaches
            phase["name"] = "warmup"
            worker.wait_warm(_check_thermal)
            logger.info(f"concurrent worker {worker.tag} warm (first call "
                        f"{worker.first_call_ms} ms)")

        # perf attaches (-p) only to threads that exist, so start it after warm-up
        if perf_on:
            phase["name"] = "perf_start"
            if perf.start():
                logger.info(f"perf stat started: {args.perf_interval_ms} ms, "
                            f"{len(perf.events)} events")
            else:
                logger.error(f"perf counters unavailable, continuing without: {perf.error}")

        # ABAB global baseline, slice A (2 s) — slice B follows the forward loops
        if monsoon:
            sync_pulse_pinned(sampler, "start", args.governor, args.cpu_mhz)
        phase["name"] = "baseline_A"
        sampler.marker("baseline_A")
        tA0 = sampler.now_ms()
        time.sleep(2.0)
        _check_thermal()
        baseA = sampler.window(tA0, sampler.now_ms())
        logger.info(f"baseline_A = {baseA['mean_P']} W ({baseA['count']} samples)")

        # ---- per-mode measurement -------------------------------------------
        op_rows = []
        frames_done: dict[str, int] = {}
        lat_ms: list[float] = []
        for mode in modes:
            logger.info(f"== mode: {mode} ==")
            prof.set_mode(mode)
            phase["name"] = "soak" if soak else mode
            sampler.marker(mode)  # tag phase so samples.csv carries measured mean-P
            n_done, mode_lat, mode_ops = measured_loop(
                task, prof, mode, args.frames, args.soak_minutes, _check_loop,
                on_first_frame=worker.release if worker is not None else None)
            op_rows += mode_ops
            lat_ms += mode_lat
            frames_done[mode] = n_done
            logger.info(f"  mode {mode}: {n_done} iterations done")
        if worker is not None:
            phase["name"] = "concurrent_drain"
            end_concurrent(worker, sampler.marker)   # the second workload ends with the primary
            logger.info(f"concurrent worker {worker.tag}: {worker.iterations} iterations")

        # ---- serve: isolated requests with idle gaps (race-to-idle transitions)
        serve_missed = None
        serve_wall_s = None
        if args.serve > 0 and args.task != "train":
            prof.set_mode("serve")
            phase["name"] = "serve"
            sampler.marker("serve")
            serve_missed = 0 if args.serve_fps is not None else None
            t_serve0 = time.monotonic()
            for r in range(args.serve):
                _check_thermal()
                prof.set_frame(r)
                for m in modes:
                    task.run_one(m)
                if args.serve_fps is None:
                    with task.step("idle_between_requests"):
                        time.sleep(args.serve_gap_ms / 1000.0)
                elif r + 1 < args.serve:
                    idle_s, missed = serve_wait(t_serve0, r + 1, args.serve_fps, time.monotonic())
                    serve_missed += missed
                    if idle_s > 0:
                        with task.step("idle_between_requests"):
                            time.sleep(idle_s)
            serve_wall_s = round(time.monotonic() - t_serve0, 3)
            if args.serve_fps is None:
                logger.info(f"serve: {args.serve} requests, gap {args.serve_gap_ms:.0f} ms")
            else:
                logger.info(f"serve: {args.serve} requests at {args.serve_fps} fps, "
                            f"{serve_missed} missed deadlines, {serve_wall_s} s")

        # ABAB global baseline, slice B (2 s); the run baseline is mean(A, B)
        phase["name"] = "baseline_B"
        sampler.marker("baseline_B")
        tB0 = sampler.now_ms()
        time.sleep(2.0)
        _check_thermal()
        baseB = sampler.window(tB0, sampler.now_ms())
        aP, bP = baseA["mean_P"], baseB["mean_P"]
        baseline_W = (aP + bP) / 2 if aP is not None and bP is not None else None
        baseline_drift_W = abs(aP - bP) if aP is not None and bP is not None else None
        if baseline_W is None or not (bmin <= baseline_W <= bmax):
            raise SystemExit(f"baseline_W={baseline_W} outside "
                             f"[{bmin}, {bmax}] - aborting")
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
        executed_dtype = yolo_variants.executed_conv_dtype(layer_rows)
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
        saved_events, prof.events = prof.events, None   # benches are labeled by bench:<cid> phase markers
        phase["name"] = "bench"
        try:
            if args.no_bench:
                perop = []
                logger.info("  skipped (--no-bench)")
            else:
                perop = bench_configs(prof, sampler, baseline_W, args.bench,
                                      require_samples=MIN_SAMPLES, max_retries=2,
                                      log=logger.info)
        finally:
            prof.events = saved_events
        _check_thermal()
        thr1 = sampler.snapshot_env()[2]
        _write_csv(os.path.join(outdir, "perop_power.csv"),
                   ["config_id", "class", "config_json", "input_kind",
                    "input_shapes", "input_dtypes", "macs", "bytes_moved",
                    "t_eff", "seq_len",
                    "window_s", "N_calls", "t_per_call_ms", "P_mean_W",
                    "P_delta_W", "baseline_interp_W", "P_delta_ABAB_W",
                    "E_per_call_mJ", "temp_start_C", "temp_end_C",
                    "freq_start_MHz", "freq_end_MHz", "samples_used", "SKIPPED",
                    "skip_reason", "error"], perop)
        if monsoon:
            sync_pulse_pinned(sampler, "end", args.governor, args.cpu_mhz)
            if perf_on:
                try:
                    perf.stop()
                except Exception as e:   # counters must never crash a measurement
                    logger.error(f"perf stop failed: {e}")
                    perf.note_error(f"perf stop failed: {e}")
                try:
                    perfcounters.write_counters_csv(os.path.join(outdir, "counters.csv"),
                                                    perf.wide_rows(), perf.events)
                except OSError as e:
                    logger.error(f"counters.csv not written: {e}")
                if perf.error:
                    logger.error(f"perf counters: {perf.error}")
            sampler.marker("done")
            sampler.events.write_csv(os.path.join(outdir, "events.csv"))

        continuity = sampler.check_recorder_continuity() if monsoon else []
        pin_status_end = pinning.status()
        fan_end = None
        if fan_mode == "full":
            try:
                fan_end = pinning._read_fan()
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"fan state unreadable at end: {e}")

        srows = sampler.rows()
        if not monsoon:
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

        _check_thermal()   # a late crossing must not be reported as a clean run
        # per-frame latency is only well defined for a single measured mode (image, train)
        lat_stats = latency_stats(lat_ms) if len(modes) == 1 else None
        lat_img = ({k: (round(v / args.batch, 3) if k != "n" else v)
                    for k, v in lat_stats.items()} if lat_stats else None)
        variant_name = yolo_variants.variant_id(args.model, args.quant, variant.get("prune_ratio"))
        map_ref = yolo_variants.lookup_map_reference(MAP_REFERENCE_PATH, variant_name, args.imgsz)
        inputs = (task.input_identity() if hasattr(task, "input_identity")
                  else {"source": "synthetic", "files": []})
        concurrent_meta = (concurrent_block(args, split, worker, frames_done, lat_stats, load2_s,
                                            serve_concurrent=False) if worker is not None else None)
        meta = {
            "task": args.task, "model": args.model, "modes": modes,
            "imgsz": args.imgsz, "conf": args.conf, "seq_len": args.seq,
            "batch": args.batch if args.task in ("train", "image") else 1,
            "optimizer": args.optimizer if args.task == "train" else None,
            "train_loss": [round(v, 5) for v in getattr(task, "losses", [])] or None,
            "bench_skipped": args.no_bench,
            "bench_forced_off_by_soak": soak,
            "bench_forced_off": bench_forced_off,
            "variant": variant_name,
            "quant": args.quant,
            "prune": variant.get("prune"),
            "quant_weights_sha256": variant.get("quant_weights_sha256"),
            "quantized_modules": variant.get("quantized_modules"),
            # same number: real quantized Conv/Linear modules only (no Quantize/DeQuantize stubs)
            "quantized_compute_modules": variant.get("quantized_modules"),
            "executed_dtype": executed_dtype,
            "map_reference": map_ref,
            "serve_requests": args.serve if args.task != "train" else 0,
            "serve_gap_ms": args.serve_gap_ms,
            "serve_fps": args.serve_fps,
            "serve_missed_deadlines": serve_missed,
            "serve_wall_s": serve_wall_s,
            "soak_minutes": args.soak_minutes,
            "concurrent": concurrent_meta,
            **({"layer_events": False} if worker is not None else {}),
            "frames_done": frames_done,
            "latency_ms_per_frame": lat_stats,
            "latency_ms_per_image": lat_img,
            "inputs": inputs,
            "holdout": None if args.holdout in (None, "none") else args.holdout,
            "dataset_version": dataset_version(args),
            "governor": args.governor,
            "fan_mode": fan_mode,
            "load_weights_s": round(load_s, 4), "first_call_ms": first_ms,
            "threads": args.threads, "threads_detail": threads_detail,
            "frames_per_mode": args.frames,
            "warmup_passes": args.warmup, "params_M": round(n_params / 1e6, 3),
            "mean_fps": None,
            "baseline_W": round(baseline_W, 6) if baseline_W is not None else None,
            "baseline_A_W": round(aP, 6) if aP is not None else None,
            "baseline_B_W": round(bP, 6) if bP is not None else None,
            "baseline_drift_W": round(baseline_drift_W, 6) if baseline_drift_W is not None else None,
            "schema_version": 4 if monsoon else 3,
            "sampler": args.sampler,
            "recorder_url": args.recorder_url if monsoon else None,
            "recorder_status_start": sampler.status_start if monsoon else None,
            "recorder_status_end": sampler.status_end if monsoon else None,
            "recorder_continuity_errors": continuity if monsoon else None,
            "perf_counters": perf.meta(perf_on, perfcounters.read_paranoid()),
            "sysfs_period_s": sysfs.period_s, "max_temp_C": sysfs.max_temp_C(),
            "thermal_abort": None,
            "throttled_start": thr0, "throttled_end": thr1,
            "pin_status_start": pin_status_start, "pin_status_end": pin_status_end,
            "t0_real_ns": getattr(sampler, "_t0_ns", None),
            "sync_tags": ["start", "end"] if monsoon else [],
            "cpu_mhz_pinned": args.cpu_mhz, "fan_full": fan_mode == "full", "pin_status": pin_status,
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
                           "per-op power is attributed, not instantaneous"
                           + (" | monsoon: 5 V input power @5 kHz, events on lab PTP wall clock, "
                              "aligned offline by sync pulses (host/align.py)" if monsoon else ""),
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
        min_hz = MONSOON_MIN_HZ if monsoon else PMIC_MIN_HZ
        if achieved_hz < min_hz:
            errors.append(f"achieved sample Hz={achieved_hz:.1f} < {min_hz}")
        if baseline_W is None or not (bmin <= baseline_W <= bmax):
            errors.append(f"baseline_W={baseline_W} out of range")
        if baseline_drift_W is None:
            errors.append("baseline drift unknown (empty A/B window)")
        elif baseline_drift_W > 0.5:
            errors.append(f"baseline drift {baseline_drift_W:.3f}W > 0.5W")
        thr = ctx[-1].get("throttle_bits") if ctx else None
        if thr is not None and (thr & 0x1) != 0:
            errors.append(f"throttled during run (throttle_bits=0x{thr:x})")
        if args.governor == "ondemand":
            errors += pinning.clock_range_errors(sysfs.rows()) + pinning.throttle_errors(ctx)
        elif args.cpu_mhz is not None:
            errors += pinning.clock_errors(ctx, args.cpu_mhz)
        if monsoon:
            if args.cpu_mhz is None:
                errors += pinning.throttle_errors(ctx)   # clock_errors already covers pinned runs
            if thr0 is None or thr1 is None:
                errors.append("throttle state unreadable at start/end (vcgencmd snapshot failed)")
            elif pinning.new_undervoltage(thr0, thr1):
                errors.append(f"under-voltage occurred during run "
                              f"(throttled 0x{thr0:x} -> 0x{thr1:x})")
            errors += continuity
        if fan_mode == "full":
            if fan_end is None or fan_end[0] < fan_end[1]:
                errors.append("fan not at max at end of run")
        errors += yolo_variants.quant_count_errors(args.quant, variant.get("quantized_modules"))
        if args.quant == "fp16":
            errors += yolo_variants.fp16_errors(layer_rows)
        if worker is not None:
            if worker.iterations < 1:
                errors.append(f"concurrent worker {worker.tag} completed no iterations")
            if worker_task is not None and not worker_task.oracle():
                errors.append(f"task oracle failed for the concurrent model {args.concurrent}")
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
    except ThermalAbort as e:
        sys.exit(_thermal_exit(e.details))
    except KeyboardInterrupt:
        # only the sysfs watchdog's hard stop is special; any other Ctrl-C behaves as before
        if sysfs.thermal_abort is not None and sysfs.thermal_abort["level"] == "hard_stop":
            sys.exit(_thermal_exit(sysfs.thermal_abort))
        raise
    finally:
        try:   # whatever below raises, a fan-off cell must get its fan back
            if worker is not None:
                try:
                    # idempotent: an abort or crash must not leave it running
                    worker.stop(strict=False)
                    if worker.alive():
                        logger.error(f"concurrent worker {worker.tag} still running "
                                     f"{worker.join_timeout_s} s after stop")
                except Exception as e:
                    logger.error(f"concurrent worker stop failed: {e}")
            try:
                perf.stop()   # best effort, idempotent: an abort or crash must not leave perf running
            except Exception as e:
                logger.error(f"perf stop failed: {e}")
                perf.note_error(f"perf stop failed: {e}")
            sysfs.stop()
            try:
                sysfs.write_csv(os.path.join(outdir, "sysfs.csv"))
            except OSError as e:
                logger.error(f"sysfs.csv not written: {e}")
            sampler.stop()
            try:
                if task is not None:
                    task.stop_camera()
            except AttributeError:
                pass
        finally:
            restore_fan_if_off(fan_mode, log=logger.error)


if __name__ == "__main__":
    main()