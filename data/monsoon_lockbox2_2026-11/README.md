# Monsoon power dataset v2: Raspberry Pi 5 YOLO workloads (monsoon_lockbox2_2026-11)

Whole-board input power of a Raspberry Pi 5, measured by a Monsoon HVPM at about 5,000 samples
per second, while the Pi runs YOLO object detection under many conditions. Every power sample is
labelled with what the Pi was doing at that instant (the **phase**, the **iteration**, the
**step** preprocess / forward / postprocess, the **network layer**), and every v2 sample also
carries the Pi's hardware performance counters (10 Hz) and its CPU clock, temperature and fan
state (1 Hz). It extends dataset v1 (branch `dataset-2026-10-matrix`, 224 cells).

| | |
|---|---|
| Runs ("cells") | **15** of 15 planned (every manifest cell reached `ok`) |
| Labelled samples | 3.3 million, about 0.19 hours of trace |
| Collected | 2026-11-24T23:43:43 to 2026-11-24T23:56:55 (the EliteBook's local time) |
| Code | commit(s) `cab2df0` of this repo, on the Pi |
| Size | 77.9 MB (Git LFS: 30 objects, `power_trace.parquet` and `events.csv.gz` of every cell) |

The rig, the method and the code are described in the repo's `README.md` (sections 1, 6, 10 and
11) and `docs/dataset-v2.md`.

## What was measured

Every cell is a YOLO detection run (`--task image`, `bus.jpg` as the detection sanity check,
Ultralytics weights, PyTorch eager on the CPU, 10 warm-up passes), repeated as the table says.
The last column is cells in this dataset / cells planned.

| Family | What varies | Cells |
|---|---|---|
| `lockbox2` | second holdout (R3), hooks off: resnet34, mobilenet_v3_large, efficientnet_b1, swin_s, yolo12m at 2400 MHz, 3 repeats each, measured after the R3 predictions were frozen | 15 / 15 |

Cell counts per value of every feature are in `coverage.csv`.

## Conditions

- **Supply:** Monsoon output 5.40 V (v1 used 5.25 V; it was raised after under-voltage with the
  s / m models). A cell that logged under-voltage (kernel message, the live `sys_uv_alarm` or the
  `get_throttled` bits) failed and is not in the dataset.
- **Kernel:** 6.18.60+rpt-rpi-2712 (v1 ran on 6.18.50; the `ref` family measures the difference).
- **CPU clock:** pinned with the performance governor except `governor` = `ondemand` cells (DVFS:
  the clock is pinned only around the sync pulses).
- **Fan:** `full` 15 cells. The soaks use `full`.
- **Temperature:** each cell started below 50 C (the orchestrator waits); `run.py` aborts at 85 C and
  stops at 90 C, and a cell with `thermal_abort.json` or `max_temp_C` >= 85 fails.
  Highest temperature in this dataset: 68.3 C.
- **Hardware counters:** `perf stat` every 100 ms with `perf_event_paranoid` = 2, so
  `scope` = user: only the run's **user-space** code is counted (events print as `cycles:u`);
  kernel time is not included. Events: cycles, instructions, cache-references, cache-misses, branch-misses, stalled-cycles-backend, LLC-load-misses, task-clock, context-switches, page-faults, cpu-migrations.
  Lowest share of an interval in which any hardware event was counted: 100.0 %.
  Counters are on in 15 of the 15 cells (all `dataset_version` 2 cells).
- **Per-layer benchmarks** (`bench:*` phases, `perop_power.csv`) run only on **repeat 0** of each
  A / A-input / C configuration (Ruling R31): per-op steady-state power is a per-configuration
  property, while energy and latency per frame come from every repeat's trace. Cells with
  `bench` = True: none. INT8 and soak cells have none.
- **Wi-Fi** on the Pi was off while measuring.
- **Serve phase:** 10 single requests with 200 ms gaps (or the family's request rate).

## Layout

```
data/monsoon_lockbox2_2026-11/
  README.md                  this file
  dataset_index.csv          one row per run: settings, derived numbers, quality flags
  coverage.csv               every feature dimension, its values and the number of cells
  excluded.csv               manifest cells that did not reach ok, with the reason
  readiness.md               the predictor-readiness checklist, computed from this data
  manifest.jsonl             the planned cells and their exact run.py arguments
  results.jsonl              the orchestrator's log, one line per attempt (incl. failed ones)
  cells/<cell_id>/           one folder per run
    power_trace.parquet      the labelled 5 kHz trace (main data), zstd
    events.csv.gz            the Pi's raw event log (input to the alignment)
    run_meta.json            settings, software versions, temperatures, baseline, perf-counter setup, mAP reference
    alignment.json           clock alignment quality and detected sample losses
    derived.json             energy and latency per frame (measured iterations only)
    phase_energy.csv         energy and mean power per phase / iteration / step
    layer_table.csv          per-layer timing, shapes (and MACs / bytes for FP32 cells)
    perop_power.csv          isolated per-layer-type power (repeat 0 of A / A-input / C; otherwise empty)
    op_table.csv             PyTorch operator timings (first iteration)
    context.csv              temperature, CPU clock and throttle flags every 5 s
    sysfs.csv                per-core clock, temperature, fan state and under-voltage alarm every second
    counters.csv             hardware counters, one row per 100 ms (v2 cells)
    thermal_abort.json       only if run.py stopped the run for heat (such cells are not kept)
    pi_console.log, pi_dmesg.log   run.py's console output and the Pi's kernel messages, when present
```

## `dataset_index.csv` columns

| Column | Meaning |
|---|---|
| `cell_id` | Run id and folder name under `cells/`: `{model}[_{variant}][_{condition}]_{imgsz}_t{threads}[_b{batch}]_{clock}_r{repeat}`; `clock` is the pinned MHz or `dvfs`; `v1ref_` cells are the v1 cross-check. |
| `model_key` | Model family key of the configuration (`yolo11s`, `yolov8m`, ...). |
| `task` | `image` (all v2 cells). |
| `model` | Weights file the run loaded (`yolo11n.pt`; the INT8 variant is in `variant`). |
| `cpu_mhz` | Pinned CPU clock in MHz. For a DVFS cell (`governor` = `ondemand`) it is 2400 and only the clock around the sync pulses; the clock then moved freely inside 1500-2400 MHz. |
| `repeat` | Repeat number of this configuration (0-based). |
| `round` | Round in which the cell ran (round r holds repeat r of every configuration). |
| `modes` | Measured modes (`forward`). |
| `frames_per_mode` | Nominal measured iterations per mode (a soak runs until its time is up; see `derived.json` `n_frames`). |
| `imgsz` | Image size in px (letterboxed square). |
| `batch` | Images per forward call. |
| `started` | When the orchestrator started this (last) attempt, EliteBook local time. |
| `git_commit` | Repo commit on the Pi that ran the cell. |
| `n_samples` | Rows in `power_trace.parquet`. |
| `trace_s` | Trace length in seconds (`n_samples` / the fitted sample rate). |
| `baseline_W` | Idle power: mean of the baseline_A and baseline_B windows around the measured phases, W. |
| `mean_P_W_forward` | Mean power over the whole `forward` phase (all iterations, W). |
| `energy_total_J` | Energy of the whole trace, J. |
| `step_coverage` | Share of samples inside the measured phase that carry a `step` label. |
| `max_residual_ms` | Worst clock-alignment residual over the sync edges, ms (limit 1.0; every kept cell passed). |
| `drift_ppm` | Fitted clock drift between the Monsoon and the Pi, ppm. |
| `slip_warning` | True if samples were lost during the run (see Quality flags). |
| `slip_samples_total` | Samples lost in the run (about 0.2 ms each). |
| `n_slips` | Number of loss events (times in `alignment.json` `slips[].t_ns`). |
| `drift_warning` | True if the fitted clock drift is above 50 ppm. |
| `throttled_start` | `vcgencmd get_throttled` before the run (0 = nothing). |
| `throttled_end` | `vcgencmd get_throttled` after the run (0 = nothing; an under-voltage fails the cell). |
| `family` | Matrix family from the manifest: `A`, `A-input`, `B`, `C`, `D`, `soak`, `ref`. |
| `dataset_version` | 2 for every v2 cell (counters guaranteed); 1 for the `ref` cells (v1 command line, counters off). |
| `variant` | Model variant from `run_meta.json` (`yolo11n`, `yolo11n_int8s`, ...). |
| `quant` | Precision: `fp32` or `int8-static` (FX graph mode, qnnpack, 200 calibration pictures). |
| `holdout` | Holdout tag: `primary-s` (yolo11s, yolov8s), `secondary-m` (yolo11m, yolov8m), `none` (everything else). Empty for the `ref` cells, which carry no tag. |
| `threads` | PyTorch threads for the main workload. |
| `governor` | `pinned` (performance governor at `cpu_mhz`) or `ondemand` (free-running clock, DVFS). |
| `fan_mode` | `full`, `auto` or `off`. |
| `input_source` | `repeat` (one picture repeated) or `stream` (50 different pictures). |
| `input_image` | The single picture used (`bus.jpg`, `empty.jpg`, `crowded.jpg`), or `N files` for a stream. |
| `serve_fps` | Requests per second of the serve phase (1, 5, 15); empty for the default 10 requests with 200 ms gaps. |
| `soak_minutes` | Length of a soak run in minutes; empty otherwise. |
| `concurrent_mode` | `model` (a second yolo11n at the same time), `spinner` (a matrix-multiply spinner), or empty. |
| `energy_per_frame_mJ` | Energy of one measured iteration (frame 0 excluded), mJ. For a concurrent cell: the whole board, both workloads. |
| `dyn_energy_per_frame_mJ` | The same minus `baseline_W` times the iteration's duration, mJ. |
| `latency_ms_per_frame` | Mean duration of one measured iteration, ms. |
| `latency_ms_per_image` | The same per image (divided by `batch`), ms. |
| `map50_95` | COCO mAP50-95 of this model / variant / image size on 500 validation pictures (one number per model, not per cell). |
| `map50` | COCO mAP50 of the same measurement. |
| `map_measured_on_machine` | Machine the mAP was measured on (`aarch64` = the Pi; anything else is a provisional NUC number). |
| `counter_coverage` | Share of samples in the measured phase that have `perf` counter data (limit 0.95; empty if the cell has no counters). |
| `max_temp_C` | Highest SoC temperature read during the run, C (the run aborts at 85). |
| `baseline_drift_exempt` | Reason the idle-baseline drift check did not apply (soak, fan auto); empty otherwise. |
| `bench` | True if the per-layer-type power benchmarks ran (`perop_power.csv` has rows); only repeat 0 of A / A-input / C. |

Empty means "does not apply or was not recorded". The v1 index has the same first columns
(minus `seq_len`, `optimizer` and the train / prefill / decode power columns, which do not apply to
YOLO inference), so v1 and v2 indexes can be concatenated on the shared columns.

## `power_trace.parquet` columns

| Column | Meaning |
|---|---|
| `t_s` | seconds since the run's first event |
| `t_pi_ns` | sample time on the Pi's clock (GPS/PTP-disciplined wall clock), ns |
| `I_A`, `V_V`, `P_W` | input current, voltage and power at the Monsoon output |
| `phase` | `load_weights`, `first:<mode>`, `warmup`, `sync:start`, `baseline_A`, `forward`, `concurrent_drain`, `serve`, `baseline_B`, `gap`, `bench:<config_id>`, `sync:end`, `idle` |
| `mode`, `iter` | the mode and iteration number inside a measured phase (empty elsewhere) |
| `step` | `preprocess` / `forward` / `postprocess` (`idle_between_requests` in `serve`), or empty. In a concurrent cell: the main workload's step |
| `layer_path`, `depth` | the innermost network module running and its nesting depth (empty in soak and concurrent cells: no per-layer events) |
| `ctx_read` | `True` while the Pi's background sensor read (`vcgencmd`) ran; drop these rows for clean power |
| `ctr_<event>` | hardware counter for the 100 ms interval containing the sample: a **count per interval**, identical for every sample in it (use ratios, or one row per interval). Empty outside every interval |
| `pct_<event>` | share (0-100) of that interval in which the event was really counted; 100 = no counter sharing |
| `sys_cpu0_kHz` ... `sys_cpu3_kHz`, `sys_temp_C`, `sys_fan_state`, `sys_uv_alarm` | the latest 1 Hz reading (at most 2 s old): per-core clock, SoC temperature, fan state (0-4), live under-voltage alarm |
| `tag`, `step_m0`, `step_m1` / `step_spin` | only in `--concurrent` cells: which workloads were inside a step, and each one's own step |

## Loading

```python
import json, pandas as pd
idx = pd.read_csv("data/monsoon_lockbox2_2026-11/dataset_index.csv")
cell = "data/monsoon_lockbox2_2026-11/cells/" + idx.cell_id[0]
df = pd.read_parquet(cell + "/power_trace.parquet")
df = df[~df.ctx_read]
print(df[df.phase == "forward"].groupby("step").P_W.mean())
print(json.load(open(cell + "/derived.json"))["energy_per_frame_mJ"])
```

## The holdout

`holdout` marks cells to keep out of predictor training so that it can be tested on models it has
never seen. The tag follows the model, not the family, so every cell of an s-model (image sizes,
threads, INT8, ...) carries it:

| Model | `holdout` |
|---|---|
| yolo11s, yolov8s | `primary-s` (primary holdout: interpolation between the nano and medium models) |
| yolo11m, yolov8m | `secondary-m` (a second, harder holdout) |
| everything else | `none` |

The `ref` cells carry no tag (empty): they are v1 command lines for comparison, not training data.
Training rows: `idx[idx.holdout == "none"]` (this drops the `ref` cells, whose tag is empty); when
`secondary-m` is the second test set it is already excluded.

## v1 versus v2: the counter split

**The 224 v1 cells have no hardware counters and cannot get them afterwards.** Every
`dataset_version` 2 cell has `counters.csv`, the `ctr_*` / `pct_*` / `sys_*` columns in
`power_trace.parquet` and `perf_counters.enabled: true` in `run_meta.json`.

- A predictor that uses **counter features** trains on **v2 cells only**.
- A predictor that does **not** use them can use v1 and v2 together.

Both are legitimate; nobody should train on counter features with rows that silently have none.
The `ref` family (0 cells here) re-runs the v1 command lines (yolov8n, yolo11n;
1500 / 2000 / 2400 MHz) with counters off on the new rig. Between v1 and v2 two things changed
together, and the `ref` cells measure them together: the supply voltage (5.25 V to 5.40 V) and the
Pi kernel (6.18.50 to 6.18.60+rpt-rpi-2712). They stay `dataset_version` 1. Use them only to compare against the
224 v1 cells.

## Quality flags (read before training on it)

- **Alignment:** every one of the 15 cells passed. Maximum sync-edge residual 0.189 ms (limit 1.0; median 0.158).
- **Sample losses (`slip_warning`, `slip_samples_total`, `n_slips`).** The Monsoon occasionally loses a few milliseconds of samples without reporting it. 6 of 15 cells have at least one detected loss (median 65 samples, maximum 133). After a loss that run's labels are shifted by roughly the loss size; averages over phases and steps are unaffected. For sample-exact work use the cells with `slip_warning == False` or cut the trace at `alignment.json` `slips[].t_ns`.
- **`drift_warning`:** 0 of 15 cells have a fitted clock drift above 50 ppm (usually a side effect of a sample loss between the two sync trains). Their alignment still passed.
- **Under-voltage and throttling:** 15 of 15 cells have a non-zero `throttled_start` / `throttled_end` (e.g. lockbox2_mbv3l_224_t4_hoff_2400_r00); a new under-voltage fails the cell, so none is kept.
- **Baseline drift check:** cells with `baseline_drift_exempt` (none) are exempt from the 0.5 W idle-baseline drift limit by design.
- **Verify warnings** (the cell still passed): slip_warning: N samples lost (6 cells); mAP reference missing for yoloNm@N (3 cells).
- **Not included:** failed or retried attempts (their last stage and error line are in
  `excluded.csv`, the full log in `results.jsonl`), cells cut from the matrix (below), the raw
  continuous Monsoon recording, `counters.raw.txt` (`counters.csv` is the same data), and v1 data.
- **Layer table for INT8 cells:** `bytes_moved`, `macs` and `arith_intensity` in `layer_table.csv`
  assume 4-byte floats, so they are wrong for quantized cells. Do not use them there.
- **INT8 static is not a fully INT8 network:** 80 convolutions are quantized; the C2PSA attention
  blocks, the Detect decode and every SiLU stay FP32.
- **Concurrent cells:** `energy_per_frame_mJ` covers both workloads together, not only the main one.
- **Soaks** (the 30-minute ones have about 9 million samples) have no per-layer begin/end events (`layer_events: false`).

## What was cut and why

- **INT8 dynamic.** Dynamic quantization only replaces Linear-type layers and YOLO is almost all
  convolutions, so it quantizes 0 modules; `run.py` fails such a run rather than label float32
  data INT8. The 9 cells are buildable behind `--include-int8d` but were not run.
- **Pruning (30 % and 50 % of yolo11n).** Structured pruning without fine-tuning gives mAP50-95
  of about 0 even after re-estimating the BatchNorm statistics, and the detection check on
  `bus.jpg` finds nothing. The tool and numbers are kept in `tools/`; the 6 cells were not run
  (`--include-prune`).
- **FP16** is optional (`--include-fp16`): it executes in float16 on the Pi CPU but whole yolo11n
  at 640 was slower than FP32 in an informal check (about 553 ms against about 330 ms per frame).
- **Cost of the perf counters.** Sampling ten counters at 10 Hz costs little, but the overhead was
  not measured by a strict A/B, so treat it as indicative: the perf-off `ref` cell yolo11n at 2400 MHz
  (v1 arguments) drew 9.6 W at 291 ms per frame, the perf-on yolo11n cells at 640 px 9.3 W at
  295.5 ms (different recipes). A controlled single-cell A/B on 2026-11-20 gave -0.15 W and +6 %
  latency, taken under under-voltage, so also indicative.
- Probe outcomes: `docs/dataset-v2.md` section 6.

## Downloading a subset

The traces are in Git LFS (a full download is a large share of GitHub's LFS bandwidth quota, so
take a subset). Everything except `power_trace.parquet` and `events.csv.gz` is plain git and
arrives with a normal clone:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch <BRANCH> --single-branch https://github.com/macclab-stevens/power-measurement-pi.git
cd power-measurement-pi
# the index, README, coverage and every small file are now present; the traces are LFS pointers
git lfs pull --include="data/monsoon_lockbox2_2026-11/cells/yolo11s_640_t4_2400_r00/*"       # one cell
git lfs pull --include="data/monsoon_lockbox2_2026-11/cells/yolo11s_*"                      # one model
```

To select by feature, filter `dataset_index.csv` and pass the cells to `git lfs pull`:

```python
import pandas as pd, subprocess
idx = pd.read_csv("data/monsoon_lockbox2_2026-11/dataset_index.csv")
pick = idx[(idx.family == "C") & (idx.imgsz == 320)].cell_id
subprocess.run(["git", "lfs", "pull", "--include", ",".join(f"data/monsoon_lockbox2_2026-11/cells/{c}/*" for c in pick)], check=True)
```

## Open items (for the dataset owner)

- TODO(controller): mAP measured on the Pi for 3 of 3 YOLO cells (plan Task 16)
- TODO(controller): branch name for the download commands (rerun with --branch-name)
