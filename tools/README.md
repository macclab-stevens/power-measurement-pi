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
| `coco_subset.py` | Downloads COCO val2017 images and YOLO labels (resumable, zip integrity checked) and builds deterministic subsets from the sorted file names: 500 eval, 200 calibration (disjoint), 50 stream, one empty and one crowded scene. | `data/coco500/{images,labels}`, `data/coco500.yaml`, `data/calib200/images`, `../inputs/stream50/*.jpg`, `../inputs/empty.jpg`, `../inputs/crowded.jpg`, `../inputs/MANIFEST.json` |

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
