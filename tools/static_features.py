"""Static layer table of a model, built on the NUC exactly as the Pi builds it (no Pi, no Monsoon).

  cd tools
  uv run python static_features.py --model ../yolo11n.pt --imgsz 640 --out /tmp/yolo11n_640.csv
  uv run python static_features.py --model torchvision:resnet18 --imgsz 224 --out /tmp/r18.csv

The module comes from `src/tasks.py` (`make_task("image", ...)`, the FP32 path `src/run.py` uses), is
wrapped in `src/layers.py` LayerProfiler and run twice on a zeros (1, 3, N, N) input. The output has
the `layer_table.csv` columns, so `analytics/static.py` `features_from_layer_table` reads it like a
measured cell. Counts are 2 per call site (two frames); the features use count / min_count, so the
frame number does not matter. Latency columns are NUC timings and are not meaningful.
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

from layers import LayerProfiler  # noqa: E402
from tasks import make_task  # noqa: E402

# the layer_table.csv column list of src/run.py
COLUMNS = ["path", "config_id", "class", "config_json", "input_shapes", "input_dtypes", "out_shapes",
           "callsite", "mode", "leaf", "params", "macs", "bytes_moved", "arith_intensity", "t_eff",
           "seq_len", "count", "lat_mean_ms", "lat_median_ms", "lat_std_ms"]
N_FRAMES = 2


def _opts(imgsz: int) -> SimpleNamespace:
    """The run.py arguments ImageTask.load reads, at their FP32 batch-1 defaults. Camera, check
    image and stream frames are Pi-side inputs that only affect the input tensor, not the module."""
    return SimpleNamespace(imgsz=imgsz, conf=0.25, check_image=None, live_camera=False, batch=1,
                           input_image=None, frames_source="repeat", quant="fp32", prune=None,
                           quant_weights=None)


def _resolve(model: str) -> str:
    """A bare weights file missing from the cwd is looked up at the repo root (never downloaded)."""
    if model.startswith(("torchvision:", "tv:")) or Path(model).is_file():
        return model
    cand = REPO / model
    return str(cand) if cand.is_file() else model


def layer_rows(model: str, imgsz: int) -> list[dict]:
    task = make_task("image", _resolve(model), _opts(imgsz))
    module = task.module.eval()
    prof = LayerProfiler(module)
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for frame in range(N_FRAMES):
            prof.set_frame(frame)
            module(x)
    return prof.aggregated(exclude_frame=-1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="yolo11n.pt | torchvision:resnet18")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--out", required=True, help="output CSV (layer_table.csv columns)")
    a = ap.parse_args(argv)
    rows = layer_rows(a.model, a.imgsz)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.model} @ {a.imgsz}: {len(rows)} rows, {sum(1 for r in rows if r['leaf'])} leaf -> {a.out}")


if __name__ == "__main__":
    main()
