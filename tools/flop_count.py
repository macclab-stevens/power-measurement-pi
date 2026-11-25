"""Whole-model MAC count (torch.utils.flop_counter.FlopCounterMode) per (model, imgsz), on the NUC.

The layer table counts MACs per module, so matrix products written as functional calls inside a module (attention:
q @ k^T, softmax(.) @ v, scaled_dot_product_attention) are missing from it: Swin-T counts 0.66 and ViT-B/16 0.64 of
their published MACs. FlopCounterMode counts every aten matmul / conv / sdpa, so `macs_total` = flops / 2 is the
full count. Built from the same `src/tasks.py` FP32 module run.py measures, on a zeros (1, 3, N, N) input.

  cd tools && .venv/bin/python flop_count.py --out ../analytics/static_flops.csv yolo11n:640 tv:resnet18:224 ...
Model specs: `<name>:<imgsz>` with <name> a YOLO weights file stem in the repo root (yolo11n -> ../yolo11n.pt) or
`tv:<torchvision name>`. The output keys rows by the analytics base-model name (see KEYS).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from tasks import make_task  # noqa: E402

KEYS = {"mobilenet_v3_small": "mbv3", "efficientnet_b0": "effb0", "resnet18": "resnet18", "swin_t": "swint",
        "vit_b_16": "vitb16", "convnext_tiny": "convnextt", "maxvit_t": "maxvitt", "resnet34": "resnet34",
        "mobilenet_v3_large": "mbv3l", "efficientnet_b1": "effb1", "swin_s": "swins"}


def _sdpa_cpu_flops(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """FLOPs of the CPU attention kernel (no formula ships for it): q @ k^T and softmax(.) @ v, 2 FLOPs per MAC."""
    b, h, s_q, d = query_shape
    s_k, d_v = key_shape[-2], value_shape[-1]
    return 2 * b * h * s_q * s_k * d + 2 * b * h * s_q * s_k * d_v


CUSTOM = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}


def count(spec: str) -> dict:
    name, imgsz = spec.rsplit(":", 1)
    imgsz = int(imgsz)
    if name.startswith("tv:"):
        model, key = f"torchvision:{name[3:]}", KEYS[name[3:]]
    else:
        model, key = str(REPO / f"{name}.pt"), name
    opt = SimpleNamespace(imgsz=imgsz, conf=0.25, check_image=None, live_camera=False, batch=1, input_image=None,
                          frames_source="repeat", quant="fp32", prune=None, quant_weights=None)   # static_features.py
    task = make_task("image", model, opt)
    mod = task.module.eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        mod(x)                                   # warm (fuses / builds what the first call builds)
    # counted with autograd ON: under no_grad nn.MultiheadAttention takes a fused fast path
    # (aten._native_multi_head_attention) that FlopCounterMode does not see (ViT-B/16 would count 11.3 of 17.6 GMACs)
    fc = FlopCounterMode(display=False, custom_mapping=CUSTOM)
    with fc:
        mod(x)
    return {"base_model": key, "imgsz": imgsz, "macs_total": fc.get_total_flops() / 2.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    rows = []
    for s in a.specs:
        r = count(s)
        rows.append(r)
        print(f"{s}: {r['macs_total'] / 1e9:.3f} GMACs", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["base_model", "imgsz", "macs_total"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["base_model"], r["imgsz"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
