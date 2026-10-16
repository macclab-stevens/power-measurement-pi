"""SP3 experiment matrix: which models, at which pinned clocks, how many repeats.

Main dataset: every model at 2400 MHz x 10 repeats. Clock sweep: every model at 1500 and
2000 MHz x 3 repeats. `cells()` expands this into manifest rows; `experiment.py plan` writes them.
"""
from __future__ import annotations

import random

MAIN_MHZ = 2400
MAIN_REPEATS = 10
SWEEP_MHZ = (1500, 2000)
SWEEP_REPEATS = 3

COMMON = ["--sampler", "monsoon", "--fan-full", "--threads", "4"]
# --frames per model targets roughly 10-30 s of measured iterations at 2400 MHz (smoke run
# 2026-10-07: yolo11n 340 ms/iter, resnet18 train b8@128 870 ms/step, lm64 train 20 ms/step)


def _infer(frames: int) -> list[str]:
    return ["--frames", str(frames), "--warmup", "10", "--serve", "10", "--serve-gap-ms", "200"]


def _train(frames: int) -> list[str]:
    return ["--frames", str(frames), "--warmup", "5", "--no-bench"]


MODELS: dict[str, list[str]] = {
    "yolov8n": ["--task", "image", "--model", "yolov8n.pt", "--check-image", "bus.jpg",
                "--imgsz", "640", "--bench", "2", *_infer(60)],
    "yolo11n": ["--task", "image", "--model", "yolo11n.pt", "--check-image", "bus.jpg",
                "--imgsz", "640", "--bench", "2", *_infer(60)],
    "effb0": ["--task", "image", "--model", "torchvision:efficientnet_b0", "--imgsz", "224",
              "--bench", "2", *_infer(200)],
    "mbv3": ["--task", "image", "--model", "torchvision:mobilenet_v3_small", "--imgsz", "224",
             "--bench", "2", *_infer(600)],
    "resnet18": ["--task", "image", "--model", "torchvision:resnet18", "--imgsz", "224",
                 "--bench", "2", *_infer(300)],
    "lm16": ["--task", "lm", "--model", "minigpt-explicit", "--seq", "16", "--bench", "1", *_infer(1000)],
    "lm64": ["--task", "lm", "--model", "minigpt-explicit", "--seq", "64", "--bench", "1", *_infer(1000)],
    "lm128": ["--task", "lm", "--model", "minigpt-explicit", "--seq", "128", "--bench", "1",
              *_infer(1000)],
    "train_mbv3_sgd": ["--task", "train", "--model", "torchvision:mobilenet_v3_small",
                       "--batch", "8", "--imgsz", "128", "--optimizer", "sgd", *_train(150)],
    "train_mbv3_adam": ["--task", "train", "--model", "torchvision:mobilenet_v3_small",
                        "--batch", "8", "--imgsz", "128", "--optimizer", "adam", *_train(150)],
    "train_resnet18_sgd": ["--task", "train", "--model", "torchvision:resnet18",
                           "--batch", "8", "--imgsz", "128", "--optimizer", "sgd", *_train(30)],
    "train_resnet18_adam": ["--task", "train", "--model", "torchvision:resnet18",
                            "--batch", "8", "--imgsz", "128", "--optimizer", "adam", *_train(30)],
    "train_lm64_sgd": ["--task", "train", "--model", "minigpt-explicit", "--batch", "8",
                       "--seq", "64", "--optimizer", "sgd", *_train(1000)],
    "train_lm64_adam": ["--task", "train", "--model", "minigpt-explicit", "--batch", "8",
                        "--seq", "64", "--optimizer", "adam", *_train(1000)],
}

SMOKE = (("yolo11n", MAIN_MHZ), ("train_lm64_adam", MAIN_MHZ), ("train_resnet18_sgd", MAIN_MHZ))


def _cell(key: str, mhz: int, rep: int, round_: int, prefix: str = "") -> dict:
    return {"cell_id": f"{prefix}{key}_{mhz}_r{rep:02d}", "model_key": key, "cpu_mhz": mhz,
            "repeat": rep, "round": round_,
            "args": [*MODELS[key], *COMMON, "--cpu-mhz", str(mhz)]}


def sweep_rounds() -> list[int]:
    """Rounds that carry the sweep repeats, evenly spread (1, 5, 8 for 3 of 10)."""
    return [int((i + 0.5) * MAIN_REPEATS / SWEEP_REPEATS) for i in range(SWEEP_REPEATS)]


def cells(seed: int = 7, smoke: bool = False, only: list[str] | None = None,
          probe: bool = False) -> list[dict]:
    """Manifest rows. Round r holds main repeat r of every model, plus one sweep repeat at the
    rounds from sweep_rounds(); rows are shuffled within a round, so the repeats of one model
    are spread over the whole matrix."""
    unknown = sorted(set(only or ()) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown model keys: {unknown}")
    keys = [k for k in MODELS if not only or k in only]
    if smoke:
        return [_cell(k, mhz, 0, 0, prefix="smoke_") for k, mhz in SMOKE if k in keys]
    if probe:                                   # every model once at the main clock
        return [_cell(k, MAIN_MHZ, 0, 0, prefix="probe_") for k in keys]
    rng = random.Random(seed)
    sweep_at = {r: i for i, r in enumerate(sweep_rounds())}
    out: list[dict] = []
    for r in range(MAIN_REPEATS):
        rnd = [_cell(k, MAIN_MHZ, r, r) for k in keys]
        if r in sweep_at:
            rnd += [_cell(k, mhz, sweep_at[r], r) for k in keys for mhz in SWEEP_MHZ]
        rng.shuffle(rnd)
        out += rnd
    return out
