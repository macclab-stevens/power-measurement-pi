# tools/ - NUC-side dataset tools

A separate uv project (Python 3.12, CPU torch 2.14.0, `ultralytics==8.4.152`, `torch-pruning`).
It is independent of the Pi project at the repo root. Run everything from this directory:

```
cd tools
~/.local/bin/uv sync                  # once; creates tools/.venv from tools/uv.lock
~/.local/bin/uv run pytest tests      # unit tests (run on the NUC)
```

## Tools

| Tool | What it does | Output |
|---|---|---|
| `coco_subset.py` | Downloads COCO val2017 images and YOLO labels (resumable, zip integrity checked) and builds deterministic subsets from the sorted file names: 500 eval, 200 calibration (disjoint), 50 stream, one detector-empty and one crowded scene. The empty scene is the first remaining zero-label image on which yolo11n FP32 (imgsz 640, conf 0.25, iou 0.45) detects nothing; it uses `../yolo11n.pt` (`--weights`, downloaded by ultralytics if absent). | `data/coco500/{images,labels}`, `data/coco500.yaml`, `data/calib200/images`, `../inputs/stream50/*.jpg`, `../inputs/empty.jpg`, `../inputs/crowded.jpg`, `../inputs/MANIFEST.json` |
| `eval_map.py` | One-time mAP50-95 / mAP50 of a model or variant artifact at an imgsz on `data/coco500`, through one custom loop (the Pi's letterbox, NMS conf 0.001 / IoU 0.7, ultralytics metrics). `--sanity` compares an FP32 model with ultralytics `.val()`; `--all` runs the plan's set (not int8-dynamic: it quantizes 0 modules on YOLO). | merges `../data/map_reference.json` keys `<variant_id>@<imgsz>` (committed; run.py copies them into run_meta) |

```
uv run python coco_subset.py                        # downloads into data/downloads/ if needed
uv run python coco_subset.py --zips-dir DIR         # reuse val2017.zip + coco2017labels.zip in DIR
```

Reruns rebuild the outputs and produce an identical `MANIFEST.json` (rule + `{coco_file, sha256, n_labels}`
per file). Only `inputs/MANIFEST.json` is committed.

## Where things live

- `tools/data/` - downloads, subsets, dataset YAML (gitignored, never copied to the Pi).
- `inputs/` - images the Pi reads (`stream50/`, `empty.jpg`, `crowded.jpg`); only `MANIFEST.json` is tracked.
- `models/` - exported/quantized/pruned weights (gitignored).

## Copy inputs and models to the Pi

```
rsync -a inputs models maccpi@192.168.0.243:Desktop/power-measurement-pi/
```
