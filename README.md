# Pi 5 YOLO Power-Measurement Harness — Runbook & Reference

**Model-agnostic YOLO power/energy data collection on a Raspberry Pi 5.**
Live camera + a YOLO detection model (any ultralytics weights) while capturing
fine-grained per-layer, per-op timing, and per-operation power from the Pi's
on-board PMIC. Output is a joinable per-run dataset for later training a
power-prediction model.

> **Starting fresh / another session?** Read this file. All code and output live
> on the Pi at `/home/macc2026/Desktop/pi_power_measurement`. To run: `cd` there
> and `~/.local/bin/uv run src/run.py`. Every step below is reproducible.

---

## 1. Goal & status

- Execute a YOLO detection model on live camera frames at useful FPS.
- For each run, capture per-layer profiles, per-aten-op timing, and per-operation
  steady-state power.
- Store a clean, joinable dataset for a downstream power-prediction model
  (`bandwidth + op cost + freq + temp -> power`).
- Phase 1 (this): PyTorch-CPU YOLO + camera + on-board PMIC power sampling.
- Later phases (designed for, not built): **Monsoon** power monitor (high-rate
  input power), **ONNX/TFLite** backends, and the power-prediction model itself.

**Status: WORKING.** A full 30-frame + per-op bench run passes acceptance.
See §6 for measured numbers from a verified run.

---

## 2. Hardware (verified)

| Item | Detail |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.0, 8 GB RAM |
| OS | Debian GNU/Linux 13 (trixie), kernel `6.18.39+rpt-rpi-2712` (aarch64) |
| CPU | 4× Cortex-A76 @ up to 2.4 GHz (`vcgencmd` shows ~2.4 GHz under load) |
| Cooling | Active cooler (idle ~47 °C, 4-core burn ~62 °C, no current throttle) |
| Camera | `ov5647` (Pi Camera v1, 5 MP) on the CSI port; libcamera stack |
| Access | `ssh -i ~/.ssh/id_pi_ml_model_pow_cap macc2026@192.168.1.153` |

**PMIC power source:** the Pi 5's DA9090 PMIC exposes rail currents/voltages via
`vcgencmd pmic_read_adc`. `vcgencmd pmic_read_adc 7 15` returns VDD_CORE current
and voltage in ~4–5 ms (~190–200 Hz, no root). Full 26-rail dump ~24 ms (used as
run-wide context only). The ML/CPU load lands on VDD_CORE, so it is the primary
power rail for per-op attribution.

---

## 3. Environment (already installed — do not reinstall)

Everything lives under `/home/macc2026/Desktop/pi_power_measurement`:

```
.venv/            # uv venv, --python /usr/bin/python3 --system-site-packages
pyproject.toml    # pinned deps
uv.lock           # lockfile = source of truth for future syncs
src/
  run.py            # orchestrator + acceptance asserts (entry point)
  cam.py            # Picamera2 wrapper (640x480 RGB888, AE/AWB locked)
  detector.py       # ultralytics YOLO wrapper (any weights)
  powersampler.py   # PMIC sampling daemon + sampler contract
  layers.py         # per-module forward hooks, config_id, aten profiling
  microbench.py     # per-op steady-state power benches
  verify.py         # prints a dataset-quality summary for the latest run
runs/<iso-ts>/      # one output dir per run
```

Stack and install (already done, for reference):

| Package | Version | Notes |
|---|---|---|
| uv | 0.12.14 | at `~/.local/bin/uv` |
| Python | 3.13.5 (system `/usr/bin/python3`) | venv uses it for picamera2 ABI |
| torch | 2.14.0 | wheel tagged `+cu130` but **CPU-only** (`torch.cuda.is_available()==False`) |
| torchvision | 0.29.0 | |
| ultralytics | 8.4.152 | |
| opencv-python-headless | 5.0.0.93 | headless avoids libGL over SSH |
| python3-picamera2 | 0.3.37-1 | apt-installed (pulls python3-libcamera) |

Install recipe (if you ever rebuild the Pi): `uv init --bare` **first** (uv
requires a pyproject to exist before `uv add`), then
`uv add ultraalytics==8.4.152 torch==2.14.0 opencv-python-headless`
(match exact pins to keep power data cross-run comparable). `apt-get install -y
python3-picamera2` was run with a user-supplied sudo password.

sudo requires the user's password interactively (`sudo -n` does NOT work on this
unit) — any future sudo command needs an interactive TTY.

---

## 4. Run it

```bash
ssh -i ~/.ssh/id_pi_ml_model_pow_cap macc2026@192.168.1.153
cd /home/macc2026/Desktop/pi_power_measurement
# full capture: 30 frames of detection + per-op power benches (3 s/op)
~/.local/bin/uv run src/run.py --frames 30 --bench 3
```

Put any recognisable object in front of the camera (COCO-80 classes work best —
a keyboard, mouse, toy airplane, mug, bottle, book…). **Acceptance requires at
least one object detected at conf ≥ 0.25**; otherwise the run exits non-zero
with an explicit message.

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | `yolo11n.pt` | any ultralytics weights path/name (auto-downloads) |
| `--frames` | 30 | detection frames captured |
| `--bench` | 3.0 | seconds per isolated op benchmark |
| `--warmup` | 10 | warm-up inference passes before measurement (thermal/DVFS stabilization) |
| `--conf` | 0.25 | detection confidence threshold |
| `--imgsz` | 640 | inference input size |
| `--threads` | 4 | torch threads (matches the 4-core Pi; recorded in meta) |
| `--out` | auto | explicit output dir |

Run takes ~4–5 min (30 frames at ~4 fps + ~45 op benches). Summary line prints
fps, baseline W, sampling Hz, row counts, best detection.

---

## 5. Swap the model (model-agnostic)

The harness has **no hardcoded model/class dependency**:

- `--weights` accepts any ultralytics `.pt` (yolo11n, yolov8n, yolo12n, yolov8s,
  yolov8m, …) or an absolute path to a trained weights file. The model is
  auto-downloaded into ultralytics' cache on first use.
- Acceptance is **class-agnostic**: `Detector.max_conf()` reports the strongest
  detection across all classes — the harness never cares which object is present.
- Layer hooks walk the model's inference graph generically: ultralytics models
  expose their graph at `.model`; any raw `torch.nn.Module` is walked directly
  (`_iter_inference_modules` in `layers.py`).
- Pinned torch/ultralytics versions are recorded in `run_meta.json` (not hardcoded).

So swapping models is just passing a different `--weights`; nothing else changes.
*Verified by running the identical pipeline with `yolov8n.pt`.*

Caveat: per-op microbenchmarks require the model to be a **PyTorch `nn.Module`**
with normal `nn.Conv2d`/`nn.BatchNorm2d`/etc. structure (true for the whole
ultralytics YOLO family). Non-PyTorch or opaque backends are a separate adapter
(see §9).

---

## 6. Verified run (dataset reality check)

Best full run: `runs/20260915T023947` (YOLO11n):

```
fps=4.13  baseline_W=1.46  hz=189  layer_rows=226  op_rows=93  perop_rows=45  skipped=0
detection: airplane @ 0.547 (class-agnostic acceptance)
```

Top per-op power (VDD_CORE, steady-state isolation loops):

```
Conv2d  P_delta=5.34 W  E_per_call=15.6 mJ  t=2.92 ms  samples=517
Conv2d  P_delta=5.02 W  E_per_call=3.91 mJ  t=0.78 ms  samples=614
... 40/40 Conv2d rows with P_delta > 0.01 W, each ≥ 500 samples
```

- Raw trace: 28,695 PMIC samples split `idle`(1.3k) / `inference`(1.8k) /
  `bench:<config>`(25.6k) — every sample is phase/window-marked.
- Aten ops: `mkldnn_convolution` dominates (78 calls, ~1.9 ms mean), plus
  `silu_`, `cat`, `copy_`, etc.
- Confounders captured per op window: temp 59.8→62.0 °C, arm freq ~2.4 GHz.
- Idle baseline is sanity-bounded (0.5–8 W) and aborts if the PMIC parse is wrong.

---

## 7. Output schema & relationships (for the ML model later)

Every run writes `runs/<iso-ts>/`:

| File | Content | Key columns |
|---|---|---|
| `samples.csv` | raw PMIC samples | `t_ms, phase, window_id, I_A, V_V, P_W` |
| `context.csv` | env every ~5 s | `t_ms, temp_C, arm_MHz, throttle_bits, rail_*` |
| `layer_table.csv` | per-module profile | `path, config_id, class, config_json, leaf, params, macs, input_shapes, out_shapes, lat_mean/median/std_ms` |
| `op_table.csv` | aten-op timing | `op, count, self_total_us, self_mean_us, input_shapes` |
| `perop_power.csv` | per-op power | `config_id, class, config_json, t_per_call_ms, P_mean_W, P_delta_W, E_per_call_mJ, temp_start/end_C, freq_start/end_MHz, samples_used, SKIPPED` |
| `annotated.jpg` | best-detection frame | — |
| `run_meta.json` | run metadata + methodology | versions, baseline_W, hz, fps, throttle_bits, join_contract |

**Joins:** `layer_table.config_id ↔ perop_power.config_id` (authoritative).
`op_table` is aten-level timing (module join is via layer_table). All tables are
time-anchored by the run's `ts`/`outdir`.

**Units contract:** numeric columns are suffixed `_W/_ms/_mJ/_C/_MHz`;
`E_mJ = P_W × t_ms` (W·ms = mJ). Documented in `run_meta.json` so a future
Monsoon backend cannot introduce a 1000× unit bug.

**Methodology (so training knows exactly what a column means):** before any
measurement the harness runs a **warm-up phase** (`--warmup`, default 10
inference passes on a synthetic frame) so temperature and CPU DVFS reach steady
state and the expensive first-inference transient (lazy init / graph fusion /
JIT — the first real inference is ~2.8 s vs ~0.23 s for later frames) is
absorbed *before* the idle baseline. The idle baseline is thus taken warm and
stable. Per-op power is measured by *steady-state isolation microbenchmark* —
deepcopy each leaf module, feed it cache-hot inputs of its real shapes, loop it
for `~bench` seconds while sampling VDD_CORE, `P_delta = P_mean − idle_baseline`.
This is attributed power, **not** instantaneous per-op power inside a live run
(the honest physical limit at ~200 Hz for ms-scale ops). DVFS is not fought —
arm freq and temp are captured as features so the predictor can regress them out.

---

## 8. Known issues, decisions & traps (do not relearn these)

1. **ultralytics `fuse()` folds BatchNorm into conv at load.** `*.bn` modules
   never execute. Hooks only aggregate modules that actually ran (which is the
   real deployed fused graph — correct to measure).
2. **`Concat.forward(x)` takes a list, not splatted tensors.** `microbench.py`
   dispatches `list`-vs-`tensor` inputs (`_forward`).
3. **Sample deque race:** `window()` must copy the deque under the sampler lock
   (`deque mutated during iteration` otherwise — surfaced on trivial ops).
4. **`uv add` fails without a pyproject** → `uv init --bare` first.
5. **PyPI default `torch` resolves to the CUDA-tagged build** (torch 2.14
   `+cu130`, ~5.6 GB) even on aarch64; it runs CPU-only and is fine, just big.
   If size matters, use the CPU-only index (`https://download.pytorch.org/whl/cpu`).
6. **Dark image ⇒ no detection** — the ov5647 AE/AWB is locked after a 2 s
   warmup; if the scene is dim the frame is near-black and nothing is detected.
   Light the object. (Console: `mean` brightness ~110 in the working case.)
7. **`vcgencmd get_throttled` = `0x50000`** (latched under-voltage + throttled
   *occurred at some point*). Current state bits 0/1 are 0, but the latched flags
   indicate the supply possibly sags under load. **Use a quality 5 V/5 A PSU and
   a proper USB-C PD cable** before the Monsoon phase to avoid CPU freq caps
   corrupting power data.
8. **sudo needs a password** on this unit (`sudo -n` fails). Interactive TTY only.

---

## 9. Monsoon (next power source) — the drop-in boundary

The sampler is the single integration point. A `MonsoonSampler` must implement
the same public contract as `PowerSampler` in `powersampler.py`:

```python
start()                 # begin sampling on a daemon thread
stop()                  # stop + raise if sampling failed
marker(name)            # label the current phase; bumps window_id on new label
now_ms()                # sampler-relative ms (for window brackets)
window(t0_ms, t1_ms)    # -> {"count", "mean_P"(W), "max_P"}
snapshot_env()          # -> (temp_C, arm_MHz, throttled_bits)  [brackets bench windows]
rows()                  # -> [{t_ms, phase, window_id, I_A, V_V, P_W}]
```

Monsoon measures **input 5 V rail power** at kHz — a strictly better ground truth
than the summed PMIC rails. Keep the same row schema so `run.py`, the CSVs, and
the later predictor are unchanged.

## 10. Next phases (not built)

1. **MonsoonSampler** backend (above) — high-rate, whole-board input power.
2. **ONNX / TFLite int8** inference backends (faster fps; per-op attribution
   coarser — TFLite has no PyTorch-style hooks).
3. **Power-prediction model**: features = op class, tensor shapes, MACs, params,
   arm freq, temp, thread count, backend; target = `P_delta_W` / `E_per_call_mJ`.
   The dataset schema (§7) is designed for this.

---

## 11. Long-duration collection (`collect.py`)

`collect.py` runs the full capture pipeline **repeatedly** so you can leave the
Pi collecting per-layer / per-op power data unattended for hours.

### Usage
```bash
cd /home/macc2026/Desktop/pi_power_measurement
uv run python collect.py --hours 6                        # 6 hours, default cycle
uv run python collect.py --runs 50                        # fixed number of cycles
uv run python collect.py --hours 8 --frames 40 --bench 5 --warmup 10 --conf 0.25
```

### Flags (forwarded to `src/run.py` per cycle)
| Flag | Default | Meaning |
|---|---|---|
| `--hours` | 0 | stop after this many hours (0 = no time budget) |
| `--runs` | 0 | stop after this many cycles (0 = no count budget; give at least one) |
| `--frames` | 30 | detection frames per cycle |
| `--bench` | 3.0 | seconds per isolated op benchmark per cycle |
| `--warmup` | 10 | warm-up inference passes before baseline per cycle |
| `--conf` | 0.25 | detection confidence threshold |
| `--imgsz` | 640 | inference input size |
| `--threads` | 4 | torch threads |
| `--weights` | `yolo11n.pt` | model to run (swap freely) |
| `--cooldown` | 8.0 | seconds idle between cycles (heat/DVFS settle) |
| `--max-fail` | 1 | consecutive failed cycles before aborting |
| `--out` | `runs` | output base directory |

### How it works
Each cycle is an **isolated `src/run.py` subprocess** (fresh camera, sampler,
model). That means hours of collection never accumulate memory or drift, and one
bad cycle cannot take down the rest. It honors whichever budget (`--hours` or
`--runs`) you set and stops cleanly.

### Outputs (under `runs/`)
- `collect_report.csv` — one row per cycle: `cycle, started, duration_s,
  returncode, outdir, fps, baseline_W, hz, layer_rows, op_rows, perop_rows,
  skipped, best_conf`. This is the run-level overview to build the training
  table from.
- `collect.log` — full collector log + per-cycle status (OK/FAILED + stdout tail).
- `<ts>/` — the full normal dataset for every cycle (see §7).

### Practical notes
- **Put a recognisable object in front of the camera** (COCO class preferred).
  If nothing is detected, a cycle fails acceptance (`best_conf 0.000`) and counts
  toward `--max-fail`. Detection now handles the empty-frame case cleanly
  (`Detector.max_conf` returns 0.0).
- Throughput: a `--frames 30 --bench 3` cycle takes ~4–5 min, so `--hours 6`
  yields ~70 cycles (~70 datasets). For "tons of data" just raise the hours.
- Data grows fast (≈1.5 MB samples.csv per cycle) but is gitignored — archive
  `runs/` separately if you need to keep it.

---

## 12. Git & version control (local repo, no remote)

The project is a **git repository on the Pi** at
`/home/macc2026/Desktop/pi_power_measurement` — created locally and **not pushed
to any remote**.

### Current state
```
01ef510 gitignore: ignore redownloadable yolo weights (*.pt)
e80bae4 YOLO11n Pi-5 power harness: per-layer/per-op power data collection
```

### What is / isn't tracked
- **Tracked:** `src/` (cam, detector, layers, microbench, powersampler, run),
  `collect.py`, `analyze_trends.py`, `verify.py`, `README.md`, `pyproject.toml`,
  `uv.lock`, `.gitignore`.
- **Ignored (`.gitignore`):** `runs/` (raw datasets — stay on disk, out of git),
  `.venv/`, `*.pt` (redownloadable weights), `*.log`, `__pycache__/`, `.DS_Store`.

### Everyday commands
```bash
git status                 # what changed
git log --oneline          # commit history
git add <files>            # stage changes (e.g. git add src/run.py)
git commit -m "description"
git diff                   # review unstaged changes
```
Data and weights are intentionally kept out of git; code + tooling + docs are
the tracked artifacts. `runs/` is the thing to archive/back up on its own.

---

## 13. Model-agnostic generalization (implemented, M1–M3)

The harness no longer requires YOLO. It runs **any PyTorch model** through a
`TaskAdapter` while measuring op-/layer-wise power the same way. Grounded in the
adversarial review (§ in `plan_power_generalization.md`).

### Architecture (three decoupled planes)
- **Data plane (measurement)** — `powersampler.py` (model-agnostic contract),
  `layers.py` hooks, `torch.profiler`, `microbench.py`.
- **Control plane (task drivers)** — `tasks.py`: `ImageTask` (classification +
  detection) and `TokenLMTask` (causal LM: prefill + decode). Adding a family =
  one adapter file; no data-plane change.
- **Structure plane** — deferred by review (see below); cheap structural signal
  (`path`, `callsite`, `mode`, ordered `op_sequence`) captured in the dataset now.

### Measurement identity (the critical fix)
`config_id` = sha256(class + structural attrs + **input shapes** + **input
dtypes**). It deliberately excludes path/callsite, so identical (op,shape,dtype)
configs share ONE bench. `path`, `callsite`, `mode` are separate structural
columns. Benches are deduped per measurement config.

### Input capture (multi-arg / dtype-aware replay)
Per forward argument we record kind + shape + dtype + replay spec, so
`microbench` replays correctly instead of SKIPPING:
- int64 index tensors (Embedding) and bool masks replay as **zeros** (legal values);
- multi-argument positional leaves (e.g. `nn.MultiheadAttention` q,k,v) replay
  as multiple tensors — verified end-to-end (`verify_multiarg.py`);
- Concat list-kind inputs unchanged.

### Call-site-aware timing
Per-path stack tags every invocation (fixes the old single-slot `_start[path]`
bug for shared/tied/reentrant modules). Every call is recorded with frame, mode
and input-config index.

### Features (per site)
`macs` (Conv2d/Conv1d/ConvTranspose2d/Linear), `bytes_moved` (= sum in/out
numel × dtype bytes) and `arithmetic_intensity` (= macs/bytes) — power tracks
memory bandwidth, so these matter more than MACs for attention/decode.

### Usage
```bash
uv run src/run.py --task image --model yolo11n.pt --check-image bus.jpg \
                  --frames 25 --bench 3               # detection (golden)
uv run src/run.py --task image --model <classif.pt>   # classification
uv run src/run.py --task lm --model minigpt --seq 64  # causal LM (torch-only)
uv run src/run.py --task lm --model hf:gpt2 --seq 64  # needs `uv add transformers`
uv run python verify.py runs/<ts>                     # schema/identity/acceptance
```

### Outputs (schema additions vs §7)
- `layer_table.csv` += `input_dtypes`, `macs`, `bytes_moved`, `arith_intensity`,
  `callsite`, `mode`.
- `perop_power.csv` += `input_dtypes`, `macs`, `bytes_moved`.
- `op_table.csv` += `mode` (per-mode aten timing).
- `run_meta.json` += `task`, `modes`, `arch_fingerprint`, `op_sequence`,
  `seq_len`, `params_M`.

### Acceptance (task-gated)
Per-task oracle replaces object-conf. Global invariants kept: **SKIPPED is
fatal**, samples≥30, hz≥50, baseline in range, op_table non-empty, per-dominant-op
P_delta>0.01 W. `--check-image` still requires object conf≥`--conf` for image
detection.

### Deferred by review
Full fx→networkx graph emission. Site identity + execution order are already in
the dataset (cheap); build the DAG with a future graph-aggregating envelope
predictor, whose schema will dictate it. Verified runs:
`runs/golden_yolo11n`, `runs/golden_minigpt`.

### Verified (post-implementation)
- YOLO golden: 324 layer sites, 165 configs, SKIPPED=0, dominant-op green.
- Mini-GPT LM: prefill+decode, int64 Embedding / multi-arg replay / per-mode
  distinct configs, SKIPPED=0, dominant-op green.
- `verify_multiarg.py`: 3-tensor positional replay + int64/bool + per-mode
  identity all pass.