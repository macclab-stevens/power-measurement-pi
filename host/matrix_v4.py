#!/usr/bin/env python3
"""Cell plan of the v4 collection (2026-11-25): deployment (hooks-off) truth with kernel-visible counters, and lockbox 3.

Why (`docs/review/2026-11-25-r4-plan.md`): every training configuration measured before v3 ran with the layer hooks
on, which add 3-25 % to the energy per image, and every counter cell so far was collected at perf_event_paranoid=2,
so context switches read 0. v4 re-measures them hooks-off on the current rig (5.40 V) at paranoid=1.

Two parts, each its own manifest (experiment.py run --manifest ...):
  deploy    every hooks-on training configuration (v1, v2, v3, lockbox 1) again with the hooks off, plus a clock sweep
            (1500 / 2000 MHz) for the models measured only at 2400 MHz and a 320 px size for the torchvision CNNs
            (2 repeats)
  lockbox3  six never-collected torchvision models, one similar training model each, hooks off, at mixed clocks;
            measured only after the R4 predictions are frozen, committed and tagged (3 repeats)

Every cell: pinned clock, fan full, perf counters on, no per-op benches (they need the hooks), 10 serve requests with
200 ms gaps, sysfs at 10 Hz. Frames: as v2 for YOLO (30 for n/s, 20 for m, divided by the batch, at least 5); for
torchvision models ceil(12 s / latency) clipped to [30, 300], from the 2400 MHz hooks-off latencies probed on the
Pi (2026-11-24/25). Rounds hold one repeat of every configuration, shuffled per round with a fixed seed.

Usage: uv run python matrix_v4.py {deploy,lockbox3} --out ../runs/v4/<part>.jsonl
"""
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matrix_v3 as m3

SERVE = m3.SERVE
SYSFS_PERIOD_S = m3.SYSFS_PERIOD_S
SEED = 20261125
REPEATS = {"deploy": 2, "lockbox3": 3}
# latency in ms (hooks off, 2400 MHz, 4 threads) keyed by (torchvision name, imgsz)
PROBED_MS = {**{(k, 224): v for k, v in m3.PROBED_MS.items()},
             ("mobilenet_v2", 224): 68.6, ("shufflenet_v2_x1_0", 224): 30.9, ("resnet50", 224): 223.5,
             ("efficientnet_b2", 224): 126.0, ("vit_b_32", 224): 249.0, ("convnext_small", 224): 313.0,
             ("resnet18", 320): 133.2, ("mobilenet_v3_small", 320): 35.4, ("efficientnet_b0", 320): 172.0,
             ("convnext_tiny", 320): 292.0}
SHORT = {**m3.SHORT, "mobilenet_v2": "mbv2", "shufflenet_v2_x1_0": "shufv2", "resnet50": "resnet50",
         "efficientnet_b2": "effb2", "vit_b_32": "vitb32", "convnext_small": "convnexts"}


@dataclass(frozen=True)
class Cfg:
    model: str              # "yolo11n" (ultralytics <model>.pt) or "tv:<torchvision name>"
    imgsz: int
    mhz: int = 2400
    threads: int = 4
    batch: int = 1
    variant: str = ""       # "", "int8s", "fp16"


def model_key(model: str) -> str:
    return SHORT[model[3:]] if m3.is_tv(model) else model


def frames(c: Cfg) -> int:
    if m3.is_tv(c.model):
        return max(30, min(300, math.ceil(12000.0 / PROBED_MS[(c.model[3:], c.imgsz)])))
    base = 20 if c.model.endswith("m") else 30
    return max(5, math.ceil(base / c.batch)) if c.batch > 1 else base


def cell_id(c: Cfg, repeat: int, prefix: str) -> str:
    parts = [prefix, model_key(c.model), c.variant, str(c.imgsz), f"t{c.threads}", f"b{c.batch}" if c.batch > 1 else "",
             str(c.mhz), f"r{repeat:02d}"]
    return "_".join(p for p in parts if p)


def args_for(c: Cfg) -> list[str]:
    tv = m3.is_tv(c.model)
    a = ["--task", "image", "--model", f"torchvision:{c.model[3:]}" if tv else f"{c.model}.pt"]
    if not tv:
        a += ["--check-image", "bus.jpg"]
    a += ["--sampler", "monsoon", "--perf-counters", "on", "--holdout", "none", "--threads", str(c.threads),
          "--imgsz", str(c.imgsz), "--cpu-mhz", str(c.mhz), "--fan", "full"]
    if c.batch > 1:
        a += ["--batch", str(c.batch)]
    a += ["--frames", str(frames(c)), "--warmup", "10", "--no-bench", *SERVE, "--layer-hooks", "off",
          "--sysfs-period-s", SYSFS_PERIOD_S]
    if c.variant == "int8s":
        a += ["--quant", "int8-static", "--quant-weights", f"models/{c.model}_int8s.pt"]
    elif c.variant == "fp16":
        a += ["--quant", "fp16"]
    return a


def _clocks(model, imgsz=640, mhz=(1500, 2000, 2400), **kw):
    return [Cfg(model, imgsz, m, **kw) for m in mhz]


# Hooks-on training configurations (v1 / v2 / v3 / lockbox 1) not yet measured hooks-off. The ones measured hooks-off
# in v3 (yolo11n, yolov8n, yolo11s, yolo12n at 640 px and mbv3, effb0, resnet18 at 224 px, all 2400 MHz) are left out.
REPLICA = (
    _clocks("yolo11m") + [Cfg("yolo11m", s) for s in (320, 480, 960)]
    + _clocks("yolo11s", mhz=(1500, 2000)) + [Cfg("yolo11s", s) for s in (320, 480, 960)]
    + [Cfg("yolo11s", 640, threads=t) for t in (1, 2)] + [Cfg("yolo11s", 640, variant="int8s")]
    + _clocks("yolo11n", mhz=(1500, 2000)) + [Cfg("yolo11n", 480)]
    + [Cfg("yolo11n", s, threads=t) for s in (320, 960) for t in (1, 2, 4)]
    + [Cfg("yolo11n", 640, threads=t) for t in (1, 2)] + [Cfg("yolo11n", 640, batch=b) for b in (2, 4, 8)]
    + [Cfg("yolo11n", 640, variant="fp16")] + _clocks("yolo11n", variant="int8s")
    + _clocks("yolov8m") + _clocks("yolov8s") + _clocks("yolov8n", mhz=(1500, 2000)) + [Cfg("yolov8n", 480)]
    + [Cfg("yolo12n", 320)]
    + [Cfg(f"tv:{m}", 224, mhz) for m in ("mobilenet_v3_small", "efficientnet_b0", "resnet18") for mhz in (1500, 2000)]
)
# New coverage: every model at three clocks, the torchvision CNNs at a second size.
SWEEP = (
    _clocks("yolo12n", mhz=(1500, 2000)) + [Cfg("yolo12s", 640, 2000)] + [Cfg("yolov8n", s) for s in (320, 960)]
    + [Cfg(f"tv:{m}", 224, mhz) for m in ("convnext_tiny", "vit_b_16", "maxvit_t") for mhz in (1500, 2000)]
    + [Cfg("tv:swin_t", 224, 2000)]
    + [Cfg(f"tv:{m}", 320) for m in ("resnet18", "mobilenet_v3_small", "efficientnet_b0", "convnext_tiny")]
)
DEPLOY = REPLICA + SWEEP
# Lockbox 3: never-collected base models, each with a similar training model, at mixed clocks (pre-registered here).
#   mbv2 ~ mbv3 / mbv3l   shufv2 ~ mbv3 (new op: channel shuffle)   resnet50 ~ resnet18 / resnet34
#   effb2 ~ effb0 / effb1   vitb32 ~ vitb16   convnexts ~ convnextt
LOCKBOX3 = [Cfg("tv:mobilenet_v2", 224, 2400), Cfg("tv:shufflenet_v2_x1_0", 224, 2000), Cfg("tv:resnet50", 224, 1500),
            Cfg("tv:efficientnet_b2", 224, 2400), Cfg("tv:vit_b_32", 224, 2000), Cfg("tv:convnext_small", 224, 2400)]
PARTS = {"deploy": (DEPLOY, "v4"), "lockbox3": (LOCKBOX3, "lockbox3")}


def cells(part: str, seed: int = SEED) -> list[dict]:
    """Manifest rows of one part: rounds of one repeat each, shuffled per round with `seed`."""
    cfgs, prefix = PARTS[part]
    rng = random.Random(f"{seed}:{part}")
    rows = []
    for r in range(REPEATS[part]):
        order = list(cfgs)
        rng.shuffle(order)
        for c in order:
            rows.append({"cell_id": cell_id(c, r, prefix), "family": prefix, "model_key": model_key(c.model),
                         "cpu_mhz": c.mhz, "repeat": r, "round": r, "holdout": None, "layer_hooks": "off",
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
