# Monsoon-Backed Power Experiment — Design Spec (v4 schema)

**Status:** draft for user review · **Date:** 2026-10-06 · **Base:** `main@7e83d0e`

## Context

Right now the harness measures power **on the Pi itself**. `src/powersampler.py` polls the PMIC through `vcgencmd pmic_read_adc` at about 190 Hz, reading the VDD_CORE rail only. This has three problems:

- It is too slow for millisecond-scale operations, which is why per-op power currently comes from isolation microbenchmarks.
- It only sees the core rail, not the whole board.
- It spawns a subprocess every ~5 ms, which itself burns CPU power and adds to what we measure.

A Monsoon **HVPM** now powers the Pi 5 through its 5 V main channel, with the Pi's own PSU disconnected. It is controlled over USB by the **EliteBook** (`macc2026@192.168.0.11`). Goal: replace the on-Pi timing/sampling logic with Monsoon ground truth at 5 kHz (whole-board input power). Then run a full experiment covering:

- inference, training, and lifecycle phases
- load and configuration sweeps

The output is a dataset where every power sample is labeled with what the model was doing at that instant. That dataset can later train a "power at time t given current operation" predictor. The predictor itself is out of scope.

Machines:

| Machine | Address | Role |
|---|---|---|
| NUC (where Claude runs) | 192.168.0.205 | Development and git source of truth; drives the others over SSH |
| EliteBook | 192.168.0.11 | Monsoon host and experiment orchestrator |
| Pi 5 | maccpi@192.168.0.243 | Runs the workloads; project is on its Desktop |

User decisions:
- HVPM main channel powers the Pi.
- All four phase classes are in scope.
- PMIC fast loop is dropped. The 5 s context thread (temp, freq, throttle, rails) stays.

## Core design: offline alignment by event log + sync pulses

Power samples and phase labels live on two different clocks:

- **Monsoon:** sample index × 200 µs (hardware clock), anchored to the EliteBook clock.
- **Pi:** `time.monotonic_ns()`.

We do **not** stream markers live. Instead:

1. **Pi side: `EventLog` sampler**, a new class in `src/powersampler.py` that keeps the same public contract as `PowerSampler`:
   - `start`, `stop`, `marker`, `now_ms`, `window`, `snapshot_env`, `rows`, `context`
   - `marker()` appends `(t_mono_ns, t_real_ns, phase, window_id)` to an in-memory log. This costs nanoseconds, so markers can be placed at **per-layer granularity** without disturbing the workload.
   - `LayerProfiler` per-call timestamps switch to the same `monotonic_ns` clock.
   - Keeps `_ctx_loop` unchanged (context-only PMIC).
   - `window(t0, t1)` asks the EliteBook daemon over HTTP, using a coarse clock offset. This is good enough for the 2–3 s baseline and bench windows, so `run.py` and `microbench.py` acceptance gates keep working with little change. Authoritative numbers are recomputed offline.
   - Selected with `run.py --sampler {pmic,monsoon}`. The old PMIC sampler stays for back-compat and tests.
2. **Sync pulses.** At run start and run end, the Pi emits a fixed busy/idle pattern on all 4 cores with distinct pulse widths, e.g. 150/100/300/100/50 ms. Exact edge times are logged. Each pattern shows up as about a 2–3 W step in the Monsoon trace.
3. **EliteBook side: `monsoon_daemon.py`**, a new file in a `host/` directory:
   - Wraps the `monsoon` pip library (HVPM, `sampleEngine`).
   - Sets Vout to 5.1–5.2 V and the current limits to maximum.
   - Records raw 5 kHz main-channel I/V to a chunked `.npz` or parquet file, along with the sample index and host clock anchors.
   - A small HTTP API provides `start`, `stop`, `window?t0&t1`, and `status`.
4. **Offline `align.py`:**
   - Starts from the NTP wall-clock coarse offset (search window of ±1 s).
   - Refines it by cross-correlating the start and end sync pulses.
   - Fits a linear offset + drift between the two clocks.
   - Maps every Pi event onto the Monsoon sample timeline.
   - Pass condition: pulse-edge residual < 1 ms (5 samples).

## Sub-projects (each gets its own spec → plan → implement cycle, in order)

### SP0: Access and infrastructure (no code; mostly verification)

1. SSH keys from the NUC to the Pi and the EliteBook, and from the EliteBook to the Pi. The user runs `! ssh-copy-id` themselves. Passwords are never stored in the repo or in scripts.
2. On the Pi, check `git status` and the `git log` diff of the Desktop copy against this repo **before** overwriting anything. Confirm the directory name, and confirm `uv` and the `.venv` work as user `maccpi`. The README's old paths (`macc2026@192.168.1.153`, `pi_power_measurement`) get updated.
3. On the EliteBook:
   - Confirm the HVPM shows up in `lsusb` (VID 2ab9).
   - Add a udev rule for non-root access. This needs sudo, and I will ask before doing it.
   - `uv` venv with `monsoon`, `numpy`, `pyarrow`.
   - Read 10 s of idle Pi power. Expect about 2.5–3.5 W.
4. Pi on Monsoon power:
   - `vcgencmd get_throttled` must be `0x0` under a 4-core burn. If not, raise Vout.
   - Add `usb_max_current_enable=1` and EEPROM `PSU_MAX_CURRENT=5000` so the Pi does not throttle without PD negotiation. This needs sudo, and I will confirm first.
   - Optional: passwordless sudo for `cpupower` only, needed for the governor/frequency sweep. I will confirm first.
5. Deploy: `rsync` (or a git remote over SSH) from this repo to the Pi and the EliteBook.

### SP1: Measurement plane, Monsoon backend

- `src/powersampler.py`: add `EventLog`/`MonsoonSampler` and `sync_pulse()`. Drop the PMIC fast loop in monsoon mode.
- `src/layers.py`: emit per-call start/end events into the event log (`monotonic_ns`).
- `src/run.py`: `--sampler` flag; sync pulses at the start and end; write `events.csv` (Pi clock) instead of `samples.csv` in monsoon mode; `schema_version: 4`.
- `host/monsoon_daemon.py` and `host/align.py`, both new.
- TDD with synthetic traces: a known offset and drift are recovered to within 0.2 ms; pulse-pattern uniqueness (no false match within ±1 s); the window API matches offline recomputation.

### SP2: Workload expansion (new phases)

`src/tasks.py` gets new adapter modes, each wrapped in markers:

- **Lifecycle:** `load_weights`, `first_inference` (JIT/lazy init, currently thrown away during warm-up and now measured), `preprocess`, `postprocess`/NMS, `camera_capture`, `idle_between_requests`.
- **Training (`TrainTask`):**
  - Sub-phases: `dataload`, `forward`, `loss`, `backward`, `optimizer_step`, `zero_grad`.
  - Models: MobileNetV3-small / ResNet18 on synthetic 64–224 px batches, and MiniGPT-explicit on random tokens.
  - Optimizers: SGD and Adam (they have different memory traffic).
- **Inference:** the existing forward / prefill / decode modes, plus per-op microbenchmarks, kept as-is.

### SP3: Experiment orchestrator and matrix (on the EliteBook)

`host/experiment.py` is a resumable manifest runner (JSONL, one cell per line). For each cell:

1. Thermal gate: wait until the Pi is below 50 °C.
2. Start the daemon capture.
3. `ssh pi 'uv run src/run.py --sampler monsoon …'`.
4. Stop the capture.
5. `rsync` the Pi run directory back.
6. Run `align.py`, then `verify.py`.
7. Mark the cell OK or FAIL. Any failed cell is retried once.

**Matrix:** a designed subset, not full factorial, of about 60 cells × 3 repeats, with randomized order and repeats spread in time.

- **Families:** yolov8n, yolo11n, effb0, mbv3, resnet18, minigpt seq16/64/128, optional distilgpt2.
- **Modes:** inference + lifecycle, plus training for mbv3, resnet18, and minigpt.
- **Sweeps:**

  | Dimension | Values |
  |---|---|
  | threads | 1, 2, 4 |
  | batch | 1, 4 |
  | imgsz | 320, 640 |
  | seq | 16, 64, 128 |
  | governor | ondemand, performance, plus a fixed 1.5 GHz if sudo is allowed |

- **Time:** about 4 min per cell, so roughly 12 h. This runs overnight.
- **Data:** about 2 GB raw at 5 kHz.
- **Smoke matrix first:** 3 cells × 1 repeat, about 15 min.

### SP4: Aligned dataset for the predictor

Per run, `align.py` writes:

| File | Contents |
|---|---|
| `power_trace.parquet` | 5 kHz rows: `t_s, I_A, V_V, P_W, phase, mode, iter, layer_path, op_class, config_id` (the innermost active event labels the sample) |
| `phase_energy.csv` | Per-phase and per-iteration E / mean P |
| `layer_power_insitu.csv` | Per-layer in-situ ΔP. At 5 kHz a 1–3 ms conv covers 5–15 samples, which was impossible before. |

Across runs:
- `dataset_index.parquet` joins run meta (family, sweep values, env, temp/freq context) for leave-one-model-out splits.
- `trace_forecast/` and `modeling/` are later pointed at schema v4. That work is not in this plan.

## Critical files

- Modify: `src/powersampler.py`, `src/run.py`, `src/layers.py`, `src/tasks.py`, `src/microbench.py` (only if the window semantics need it), `verify.py`, `README.md` (new topology and runbook), `pyproject.toml`.
- New: `host/monsoon_daemon.py`, `host/align.py`, `host/experiment.py`, `host/matrix.yaml`, `tests/test_align.py`, `tests/test_eventlog.py`.
- Reuse:
  - The `PowerSampler` contract (README §9)
  - `LayerProfiler` hooks and call-site stacks
  - The `bench_configs` gap/bench windows
  - `collect.py`'s subprocess-per-cycle pattern and `MATRIX` idea, ported to the host orchestrator
  - `verify.py` gates

## Process gates (brainstorming → spec → plan)

1. When this plan is approved, I write the SP1–SP4 design as a spec at `docs/superpowers/specs/2026-10-06-monsoon-experiment-design.md`, commit it, and you review it.
2. SP0 can start right away. It is read-only checks plus key setup, and I confirm each sudo step.
3. Each sub-project then goes through `writing-plans` and is implemented test-first.
4. Before any long run: smoke matrix → you look at the aligned plot → full matrix.

## Verification

1. **Unit:** `uv run pytest tests/ modeling/ trace_forecast/`. Existing tests must stay green, and the new align/eventlog tests must pass.
2. **Sync check on hardware:** 20 pulse trains, then align. Edge residual must be < 1 ms, and drift must be < 50 ppm and stable across runs.
3. **Physics checks:**
   - Idle Pi power on the Monsoon is about 2.5–3.5 W.
   - A 4-core burn adds about 3–5 W.
   - Monsoon ΔP during forward correlates with the PMIC VDD_CORE ΔP from earlier `v3_*` runs: same ranking across families.
   - Energy closes: the sum of per-phase E equals the total E within 1%.
4. **Golden runs:**
   - yolo11n inference and minigpt training on the smoke matrix.
   - Plot the 5 kHz trace with phase/layer shading. You confirm visually that backward > forward and decode < prefill per token.
5. **Full matrix:** the manifest shows ≥ 95% of cells OK, and `verify.py` passes on each.

## Interface contracts (fixed for SP1–SP4)

- **Pi `events.csv`** (one row per event, Pi clock):
  `t_mono_ns, t_real_ns, kind, name, mode, iter, layer_path, config_id, window_id`
  - `kind` ∈ `marker | begin | end | pulse_on | pulse_off`.
  - `begin`/`end` pairs nest. The labeling depth order is run phase → mode → iteration sub-phase → layer.
- **EliteBook `monsoon_raw.npz`** (one per cell):
  - `I_A`, `V_V` as float32 at 5000 Hz
  - `anchor_idx`, `anchor_host_real_ns`: a host-clock anchor about once per second
  - `vout_V`, `hvpm_serial`
- **Sync pulse:** all 4 cores busy-loop with the pattern ON 150 / OFF 100 / ON 300 / OFF 100 / ON 50 ms, preceded and followed by ≥ 500 ms idle. It is emitted after warm-up at run start and again at run end. Every edge is logged as a `pulse_on`/`pulse_off` event.
- **Labeling rule:** each Monsoon sample takes the labels of the innermost span that contains it after alignment. Samples outside any span are labeled `idle`.
- **Alignment output:** `pi_to_monsoon(t_mono_ns) = a + b·t_mono_ns`. `align.py` stores `a`, `b`, the per-edge residuals, and `max_residual_ms` in `alignment.json`. A run whose `max_residual_ms` ≥ 1.0 is marked FAIL.
- **Units:** the existing `_W/_ms/_mJ` suffix contract is unchanged. `E_mJ = P_W × t_ms`.

## Non-goals

- The predictor model itself.
- Fused backends (ONNX/TFLite) and GPU/NPU.
- Per-rail power. The Monsoon measures whole-board 5 V input; the PMIC context gives rail snapshots only.
- Mixing v3 (PMIC) and v4 (Monsoon) power targets in one training set.
