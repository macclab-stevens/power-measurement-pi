#!/usr/bin/env python3
"""Cell plan of the v3 collection (2026-11, approved): docs/review/next-collection.md.

Three parts, each its own manifest (experiment.py run --manifest ...):
  calibration  hooks on vs off on the same configurations, same session: the instrumented-to-plain ratio
               (7 models x {on, off} x 3 repeats = 42 cells)
  additions    attention / modern architectures for training, hooks off (5 models at 2400 MHz plus 2 at
               1500 MHz, 3 repeats = 21 cells)
  lockbox2     never-collected models similar to seen ones, hooks off, measured only after the R3 predictions
               are frozen and committed (5 models x 3 repeats = 15 cells)

Every cell: 2400 MHz pinned (unless stated), 4 threads, fan full, perf counters on, no per-op benches (they need
the layer hooks), 10 serve requests with 200 ms gaps, sysfs at 10 Hz. Frames per cell: 30 for YOLO n/s and 20
for YOLO m, as in v2; torchvision models get ceil(12 s / latency) clipped to [30, 300], from the latencies
probed on the Pi on 2026-11-24 (hooks off, 2400 MHz). Rounds hold one repeat of every configuration, shuffled
per round with a fixed seed, so slow drifts spread over all configurations.

Usage: uv run python matrix_v3.py {calibration,additions,lockbox2} --out ../runs/v3/<part>.jsonl
"""
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

THREADS = 4
REPEATS = 3
SERVE = ("--serve", "10", "--serve-gap-ms", "200")
SYSFS_PERIOD_S = "0.1"
SEED = 20261124
# measured latency (ms, hooks off, 2400 MHz, 4 threads) for the frame count of torchvision models
PROBED_MS = {"mobilenet_v3_small": 21.7, "efficientnet_b0": 84.8, "resnet18": 79.4, "swin_t": 215.0,
             "vit_b_16": 636.0, "convnext_tiny": 172.0, "maxvit_t": 458.0, "resnet34": 128.0,
             "mobilenet_v3_large": 54.0, "efficientnet_b1": 117.0, "swin_s": 372.0}
SHORT = {"mobilenet_v3_small": "mbv3", "efficientnet_b0": "effb0", "resnet18": "resnet18", "swin_t": "swint",
         "vit_b_16": "vitb16", "convnext_tiny": "convnextt", "maxvit_t": "maxvitt", "resnet34": "resnet34",
         "mobilenet_v3_large": "mbv3l", "efficientnet_b1": "effb1", "swin_s": "swins"}


@dataclass(frozen=True)
class Cfg:
    model: str            # "yolo11n" (ultralytics, <model>.pt) or "tv:<torchvision name>"
    imgsz: int
    hooks: str = "off"
    mhz: int = 2400


def is_tv(model: str) -> bool:
    return model.startswith("tv:")


def model_key(model: str) -> str:
    return SHORT[model[3:]] if is_tv(model) else model


def frames(c: Cfg) -> int:
    if is_tv(c.model):
        return max(30, min(300, math.ceil(12000.0 / PROBED_MS[c.model[3:]])))
    return 20 if c.model.endswith("m") else 30


def cell_id(c: Cfg, repeat: int, prefix: str) -> str:
    return f"{prefix}_{model_key(c.model)}_{c.imgsz}_t{THREADS}_h{c.hooks}_{c.mhz}_r{repeat:02d}"


def args_for(c: Cfg) -> list[str]:
    model = f"torchvision:{c.model[3:]}" if is_tv(c.model) else f"{c.model}.pt"
    a = ["--task", "image", "--model", model]
    if not is_tv(c.model):
        a += ["--check-image", "bus.jpg"]
    a += ["--sampler", "monsoon", "--perf-counters", "on", "--holdout", "none", "--threads", str(THREADS),
          "--imgsz", str(c.imgsz), "--cpu-mhz", str(c.mhz), "--fan", "full", "--frames", str(frames(c)),
          "--warmup", "10", "--no-bench", *SERVE, "--layer-hooks", c.hooks, "--sysfs-period-s", SYSFS_PERIOD_S]
    return a


CALIBRATION = [Cfg(m, s, h) for m, s in (("yolo11n", 640), ("yolov8n", 640), ("yolo11s", 640), ("yolo12n", 640),
                                          ("tv:mobilenet_v3_small", 224), ("tv:efficientnet_b0", 224),
                                          ("tv:resnet18", 224)) for h in ("on", "off")]
ADDITIONS = ([Cfg(m, s) for m, s in (("yolo12s", 640), ("tv:swin_t", 224), ("tv:vit_b_16", 224),
                                     ("tv:convnext_tiny", 224), ("tv:maxvit_t", 224))]
             + [Cfg("yolo12s", 640, mhz=1500), Cfg("tv:swin_t", 224, mhz=1500)])
LOCKBOX2 = [Cfg(m, s) for m, s in (("tv:resnet34", 224), ("tv:mobilenet_v3_large", 224), ("tv:efficientnet_b1", 224),
                                   ("tv:swin_s", 224), ("yolo12m", 640))]
PARTS = {"calibration": (CALIBRATION, "v3cal"), "additions": (ADDITIONS, "v3add"), "lockbox2": (LOCKBOX2, "lockbox2")}


def cells(part: str, seed: int = SEED) -> list[dict]:
    """Manifest rows of one part: rounds of one repeat each, shuffled per round with `seed`."""
    cfgs, prefix = PARTS[part]
    rng = random.Random(f"{seed}:{part}")
    rows = []
    for r in range(REPEATS):
        order = list(cfgs)
        rng.shuffle(order)
        for c in order:
            rows.append({"cell_id": cell_id(c, r, prefix), "family": prefix, "model_key": model_key(c.model),
                         "cpu_mhz": c.mhz, "repeat": r, "round": r, "holdout": None, "layer_hooks": c.hooks,
                         "args": args_for(c)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    rows = cells(a.part, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} cells -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
