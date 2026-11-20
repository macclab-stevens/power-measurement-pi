# tools/ - NUC-side dataset tools

A separate uv project (Python 3.12, CPU torch 2.14.0, `ultralytics==8.4.152`, `torch-pruning`).
It is independent of the Pi project at the repo root. Run everything from this directory:

```
cd tools
~/.local/bin/uv sync                  # once; creates tools/.venv from tools/uv.lock
~/.local/bin/uv run pytest tests      # unit tests (run on the NUC)
```

What these tools are for, in one line each: the NUC builds weights offline (INT8 static, pruned)
and prepares data (COCO subsets); the numbers that go into the dataset (power, and the mAP that
travels with it) are measured on the Pi. See [../docs/dataset-v2.md](../docs/dataset-v2.md).

## Tools

| Tool | What it does | Output |
|---|---|---|
| `coco_subset.py` | Downloads COCO val2017 images and YOLO labels (resumable, zip integrity checked) and builds deterministic subsets from the sorted file names: 500 eval, 200 calibration (disjoint), 50 stream, one detector-empty and one crowded scene. The empty scene is the first remaining zero-label image on which yolo11n FP32 (imgsz 640, conf 0.25, iou 0.45) detects nothing; it uses `../yolo11n.pt` (`--weights`, downloaded by ultralytics if absent). | `data/coco500/{images,labels}`, `data/coco500.yaml`, `data/calib200/images`, `../inputs/stream50/*.jpg`, `../inputs/empty.jpg`, `../inputs/crowded.jpg`, `../inputs/MANIFEST.json` |
| `quantize_yolo.py` | INT8 static post-training quantization for the Pi's `qnnpack` engine: PyTorch FX graph mode, Conv+BatchNorm fused first, calibrated on `data/calib200` with the Pi's own letterbox at 640, activations `quint8`, weights `qint8` per channel. Options: `--model`, `--out-dir`, `--calib-dir`, `--calib-limit N`, `--per-tensor`, `--observer`, `--verify-only`. Reloads the artifact in a fresh process (only `src/` on the path) as a check. | `../models/yolo11n_int8s.pt`, `yolo11s_int8s.pt` (`pi-power-int8s/1`, gitignored) |
| `eval_map.py` | One-time mAP50-95 / mAP50 of a model or variant artifact at an imgsz on `data/coco500`, through one custom loop (the Pi's letterbox, NMS conf 0.001 / IoU 0.7, ultralytics metrics). `--sanity` compares an FP32 model with ultralytics `.val()`; `--all` runs the plan's set (not int8-dynamic: it quantizes 0 modules on YOLO); `--skip-existing`, `--data-dir`. | merges `../data/map_reference.json` keys `<variant_id>@<imgsz>` (committed; run.py copies them into run_meta) |
| `prune_yolo.py` | Global channel pruning of yolo11n at 0.30 and 0.50 with torch-pruning (L2 weight magnitude, Detect output convs and C2PSA attention ignored, **no fine-tuning**), then BatchNorm statistics re-estimated on `calib200`. C2f/C3k2/C2PSA are first replaced by split-free equivalents from `src/yolo_variants.py` (identical outputs). Reloads each artifact in a fresh process with only `src/` on the path. **Not used in the default matrix, see below.** | `../models/yolo11n_p30.pt`, `yolo11n_p50.pt` (`pi-power-pruned/1`, gitignored) |

```
uv run python coco_subset.py                        # downloads into data/downloads/ if needed
uv run python coco_subset.py --zips-dir DIR         # reuse val2017.zip + coco2017labels.zip in DIR
uv run python quantize_yolo.py --model ../yolo11n.pt      # then ../yolo11s.pt
uv run python prune_yolo.py --model ../yolo11n.pt
```

Reruns rebuild the outputs and produce an identical `MANIFEST.json` (rule + `{coco_file, sha256, n_labels}`
per file). Only `inputs/MANIFEST.json` is committed.

## What the INT8 models really are

`--quant int8-static` runs a **mixed** model, and the artifact says so (`fp32_modules`, plus the
count `quantized_compute_modules`):

- **80 convolutions are quantized** in each of yolo11n and yolo11s: the backbone, the neck and the
  Detect head's box and class branches. `run.py` counts them at load and writes the number to
  `run_meta.json` (`quantized_modules`). A run that quantized 0 modules fails.
- The **C2PSA attention block** and the **Detect decode** (DFL, anchors, box decoding, sigmoid)
  stay in FP32: they cannot be traced for quantization.
  **SiLU** has no INT8 kernel, so it also runs in FP32 between the quantized convolutions
  (dequantize, SiLU, quantize).

So the INT8 cells measure INT8 convolutions inside a model that still has FP32 parts. They are not a
fully INT8 network, and a reader of the dataset should not call them one.

## Variants that are not in the default matrix

| Variant | Why it is out | Still buildable |
|---|---|---|
| **INT8 dynamic** (`--quant int8-dynamic`) | Dynamic quantization only replaces Linear-type layers. YOLO has (almost) none, so it quantizes 0 modules and would be FP32 under an INT8 label. `run.py` fails such a run on purpose. | `plan --v2 ... --include-int8d` |
| **Pruned yolo11n** (`prune_yolo.py`) | Structured pruning with no fine-tuning breaks the model. mAP50-95 is about 0 at 30 % and 50 % (0.021 even at 10 %), even after the BatchNorm re-estimation, and the sanity check does not find the bus in `bus.jpg`. Fine-tuning would be a training project of its own. | `plan --v2 ... --include-prune`; the tool, the artifacts and the numbers in `data/map_reference.json` stay as the negative result |
| **FP16** | It runs: the convolutions really execute in float16 on the Pi CPU (`executed_dtype`). But in an informal check whole yolo11n at 640 was slower than FP32 (about 553 ms per frame against about 330 ms). | `plan --v2 ... --include-fp16` (3 cells) |

## mAP numbers: provisional until measured on the Pi

`data/map_reference.json` holds one entry per model / variant / image size, and `run.py` copies the
matching entry into every cell's `run_meta.json` as `map_reference`. **The entries in the file today
were measured on the NUC (x86).** INT8 (`qnnpack` on ARM) and FP16 run different kernels on the Pi, so
the decision is to measure accuracy on the Pi itself (plan Task 16, still to run). Until then:

- treat the numbers as provisional;
- `host/verify_cell.py` warns for every YOLO cell whose entry does not carry
  `measured_on.machine == "aarch64"`.

To remove the warning, `eval_map.py` is re-run on the Pi with the Pi project environment and the
`coco500` data copied there, and the NUC entries are replaced.

## Where things live

- `tools/data/` - downloads, subsets, dataset YAML (gitignored, never copied to the Pi, except `coco500` for the Pi-side mAP run).
- `inputs/` - images the Pi reads (`stream50/`, `empty.jpg`, `crowded.jpg`); only `MANIFEST.json` is tracked.
- `models/` - exported/quantized/pruned weights (gitignored).
- `../*.pt` - the six YOLO weights (yolo11n/s/m, yolov8n/s/m): gitignored, on the NUC and the Pi.

## Copy inputs and models to the Pi

`inputs/` and `models/` are not in git, so after building them on the NUC:

```
rsync -a inputs models maccpi@192.168.0.243:Desktop/power-measurement-pi/
```

(`rsync` fails with code 23 if `models/` does not exist yet: build the INT8 models first, or copy
`inputs` alone.) The six `*.pt` weights are on the Pi already; if one is missing, copy it from the repo
root the same way. The Pi never needs `tools/`.
