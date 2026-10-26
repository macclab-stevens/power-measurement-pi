# SP2 + SP3: lifecycle/training phases and the experiment matrix

Date: 2026-10-07. Base: `monsoon-experiment@e0ad3e1` (SP1 done, see `sp1-bringup-log.md`).
Spec: `docs/superpowers/specs/2026-10-06-monsoon-experiment-design.md` (SP2, SP3, Rev. 1-2).

## User decisions carried in

- 10 runs per model at the pinned 2400 MHz clock (main dataset).
- 1500 and 2000 MHz sweep points: a subset per model (3 repeats each).
- Occasional unreported sample losses are accepted (`slip_warning`); the repeats average them out.

## SP2 design (Pi side)

### Sub-phase ("step") events

`EventLog.step(name)` is a context manager that writes `step_begin` / `step_end` events. Steps do
not change the layer `depth`, so layer spans keep their nesting. `align.py` adds a `step` column
to `power_trace.parquet`: the innermost step span that contains the sample, or `""`. The labeling
order is now phase (marker) → mode/iter → step → layer.

Adapters get a `step` attribute (default: a no-op context manager). `run.py` sets it to
`sampler.events.step` in monsoon mode. In PMIC mode nothing changes.

### Lifecycle phases (all tasks, monsoon mode)

Phases (markers) of one run, in order:

| Phase | What runs |
|---|---|
| `load_weights` | `make_task` (model construction / checkpoint load) |
| `first:<mode>` | the first call of each mode (lazy init, allocator, oneDNN primitive creation) |
| `warmup` | as before |
| `sync:start` … `baseline_A` | as before |
| `<mode>` | the measured iterations, each split into steps |
| `serve` | `--serve N` requests with `--serve-gap-ms` idle between them (step `idle_between_requests`) |
| `baseline_B`, `bench:*`, `sync:end`, `done` | as before |

The sampler is started before the model loads, so `load_weights` is on the trace.

Inference steps per iteration:

- `ImageTask`: `camera_capture` (live camera only) → `preprocess` (letterbox/resize of a uint8
  HWC frame, to float CHW /255) → `forward` → `postprocess` (detection: ultralytics NMS;
  classification: softmax + top-5).
- `TokenLMTask`: `preprocess` (token ids) → `forward` → `postprocess` (softmax over the last
  position + greedy/argmax pick).

### Training (`--task train`)

`TrainTask`, mode `train`, steps per iteration: `dataload` → `zero_grad` → `forward` → `loss` →
`backward` → `optimizer_step`.

- Models: `torchvision:mobilenet_v3_small`, `torchvision:resnet18` (synthetic images
  `--batch` × 3 × `--imgsz`², 10 classes), `minigpt-explicit` (random tokens `--batch` ×
  `--seq`, next-token loss).
- `dataload` builds a uint8 batch from a fixed synthetic pool, applies a random flip, and
  normalizes it, so it is real CPU work and not a single `torch.rand`.
- `--optimizer sgd|adam` (SGD momentum 0.9, lr 0.01; Adam lr 1e-3). `zero_grad(set_to_none=False)`,
  so it is a real memset and visible.
- Oracle: every loss is finite. The loss curve is written to `run_meta.json` (`train_loss`).
- Training cells run with `--no-bench`: the per-op isolation benches are inference-only and
  already covered by the inference cells of the same model.

### `--no-bench`

Skips `bench_configs`; `perop_power.csv` is written with the header only, and the
dominant-op acceptance gates are skipped (`run_meta.bench_skipped = true`).

## SP3 design (EliteBook side)

`host/experiment.py`:

- `plan`: expands `host/matrix.py` into `manifest.jsonl` (one cell per line: `cell_id`,
  `model_key`, `cpu_mhz`, `repeat`, `args`). Order: rounds by repeat index, cells shuffled within
  a round with a fixed seed, so repeats of one model are spread over the whole run.
- `run`: resumable. Skips cells already `ok` in `results.jsonl`. For each cell:
  1. Recorder `/status` must be `recording`.
  2. Thermal gate: Pi below 50 C (polls every 10 s, max 10 min).
  3. `ssh pi 'cd … && uv run --frozen python src/run.py … > runs/<cell>/../<cell>.log 2>&1'`.
  4. `rsync` the run directory back.
  5. `align.py`, then `verify_cell.py` (writes `phase_energy.csv`, checks the cell).
  6. Append the result line. A failed cell is retried once, at the end of its round.
- Runs under `systemd-run --user` so it survives SSH disconnects; `status` prints progress.

`host/verify_cell.py` gates: alignment pass; `run_meta` schema 4 with the expected pinned clock
and fan; all expected phases and steps present with samples; power within 1.5-25 W; energy closure
(sum of per-phase energy equals the total within 1%).

### Matrix (`host/matrix.py`)

| Key | Task | Args |
|---|---|---|
| yolov8n | image | `--model yolov8n.pt --check-image bus.jpg --imgsz 640` |
| yolo11n | image | `--model yolo11n.pt --check-image bus.jpg --imgsz 640` |
| effb0 | image | `--model torchvision:efficientnet_b0 --imgsz 224` |
| mbv3 | image | `--model torchvision:mobilenet_v3_small --imgsz 224` |
| resnet18 | image | `--model torchvision:resnet18 --imgsz 224` |
| lm16 / lm64 / lm128 | lm | `--model minigpt-explicit --seq 16/64/128` |
| train_mbv3_sgd / _adam | train | `--model torchvision:mobilenet_v3_small --batch 8 --imgsz 128` |
| train_resnet18_sgd / _adam | train | `--model torchvision:resnet18 --batch 8 --imgsz 128` |
| train_lm64_sgd / _adam | train | `--model minigpt-explicit --batch 8 --seq 64` |

Every cell: `--sampler monsoon --fan-full --threads 4 --serve 10` (inference only). 2400 MHz × 10
repeats, plus 1500 and 2000 MHz × 3 repeats: 14 models × 16 = 224 cells.

Smoke matrix first (one cell each of yolo11n inference, train_lm64_adam, train_resnet18_sgd at 2400 MHz), then the full matrix.
