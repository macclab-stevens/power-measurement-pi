# Analytics and energy prediction: design

Date: 2026-11-22. Status: approved in conversation (design), spec under review.
Source plan: `~/Desktop/pi-power-analysis-prediction-plan.pdf` (34 pp.), read critically, not adopted
wholesale. Section 2 records where this spec departs from it and why.

## 1. Goal

Predict how much energy a Raspberry Pi 5 will use for a specific ML deployment, before running it
(Mode A) and while or after running it without a power meter (Mode L, live), for models that are new
to the predictor but similar to something it has seen. Measure that adaptability honestly, including
where it fails.

Decisions by Ishan (2026-11-22):

| Question (plan section 15) | Answer |
|---|---|
| Q1 headline target | T3, per-inference energy: predict a deployment's energy when the scenario is known beforehand |
| Q2 mode | Both: a priori (Mode A) and live energy accounting during a deployment (Mode L) |
| Q3 holdout | Any type of model, given something similar was seen: a leave-one-model-out battery, not only the s-model holdout |
| Q4 machine | All analysis on the NUC (192.168.0.205). The EliteBook only serves a one-time copy of the packaged datasets |
| New data | Approved: a small lockbox collection after both predictors are frozen (section 9) |

## 2. Facts that shaped the design, and departures from the plan

Measured 2026-11-22 from the v1/v2 dataset indexes and sample cells (read-only):

- v2's 179 cells are 57 configurations; repeat-to-repeat CV of energy per frame is 0.02-1.7 %
  (about 0.5 % typical). v1 has 42 configurations, of which 15 (5 vision models x 3 clocks) are
  inference T3 material. The evidence unit is the configuration, and there are about 9 distinct
  vision models.
- Energy per image spans 0.8-42 J (52x); forward power only 5.4-11.5 W (2x); latency 100-3,870 ms
  (39x). Energy variance is mostly latency variance.
- v1 and the v2 `ref` cells (same command lines) agree within +/-1 % on forward power and 0-3 % on
  energy per frame. Pooling is defensible with a rig indicator.
- Input picture (empty / crowded / 50-picture stream) changes energy per frame by under 1 %.
- Serving at 5 and 15 fps exceeds yolo11n's about 3.4 fps capacity: only the 1 fps cells contain
  real idle time.
- 30-minute soaks: energy per frame 5-7 % above short cells (one soak per model, two in total).
- The repo's earlier PMIC-era work (`modeling/report.md`): per-layer scaling laws lost to
  `mean power x measured latency` on every family.
- `context.csv` in v1 and v2 holds 13 PMIC rail currents and voltages every 5 s.
- v1 and v2 `phase_energy.csv` differ in column order; v2 adds a `profiler` step inside the forward
  phase (a naive forward-phase sum over-counts latency by about 8 %).

| Plan says | This spec | Why |
|---|---|---|
| About 150M autocorrelated rows; trace-scale tooling; Optuna 50 trials; MLPs; TCN | Configuration-level tables; small fixed grids; no MLP, TCN or Optuna | Real n is about 100 configurations and about 9 models; large searches overfit inner folds |
| MAE primary | Relative error primary (median and mean absolute percentage error); MAE secondary | 52x range: MAE would only measure the largest models |
| Never predict power and latency separately | Score energy directly, but allow a factored `log E = log t + log P` candidate | Latency carries most variance; let held-out scores decide |
| Mode B: counters from the scored run | Mode L: causal 100 ms live model; probe mode = Mode L integrated over a different repeat | Same-run counters leak timing (task-clock is latency) |
| Primary holdout: s-models only, test touched once | Leave-one-model-out battery with similarity classes plus a fresh lockbox | Ishan's Q3; with LOMO every model is a fold and summaries were already viewed, so only new data is pristine |
| T1, T2, T3, T4 | T3 plus a deployment calculator; Mode L is a 100 ms power model scored on integrated energy | T3 is the headline; T2 / T4 deferred |
| Counters on/off ablation | Ladder: lookup, calibrated PMIC sum, task-clock/cycles, all counters (+ PMIC) | Asks whether perf beats a stopwatch and the Pi's own power readings |
| Slip interpolation with sensitivity analysis | Exclude slipped intervals and report them; revisit only if a scored segment loses over 1 % | Slips lose milliseconds out of minutes |
| Analysis on the training machine | NUC only | 15 GB RAM, no live instrument, dev machine; EliteBook runs the recorder |

## 3. Data

### 3.1 Copies

The packaged datasets are copied once from the EliteBook over the LAN (not from GitHub, to spare the
LFS bandwidth quota) to the NUC, outside the repo:

```
~/pi-power-data/v1/   <- EliteBook ~/dataset-push/data/monsoon_matrix_2026-10   (2.7 GB, 224 cells)
~/pi-power-data/v2/   <- EliteBook ~/dataset-v2/data/monsoon_v2_2026-11        (3.2 GB, 179 cells)
~/pi-power-data/lockbox/   (section 9, later)
```

These are the same bytes as branches `dataset-2026-10-matrix` and `dataset-2026-11-v2`, so every
result can be rebuilt from GitHub. Copies are verified against each dataset's manifest (sizes and
hashes) and treated as read-only. The data root is a config value (`ANALYTICS_DATA`, default
`~/pi-power-data`). v2 `events.csv` is stored as `events.csv.gz`; readers accept both.

### 3.2 Rig indicator

`rig` = collection campaign: `oct2026` (v1: 5.25 V, kernel 6.18.50) or `nov2026` (v2 and `ref`
cells: 5.40 V, kernel 6.18.60). It is not `dataset_version`: `ref` cells are `dataset_version` 1
but `nov2026` rig. Predictions are made for `nov2026`.

### 3.3 Scope of workloads

- Mode A training and evaluation: vision inference cells (YOLO and the three v1 classifiers),
  excluding serve, concurrency and soak cells (those validate the deployment calculator, section 6).
- Mode L: v2 cells (counters) of every family including serve, concurrency and soak; `ref` cells
  have no counters and are excluded.
- MiniGPT and training workloads: described in the exploration report only. With one language model
  there is never "something similar" to learn from.

## 4. Derivation and QC (Phase 1)

### 4.1 One derivation for both versions

`analytics.derive` computes, per cell and per measured mode, from `phase_energy.csv` (by column
name) and `run_meta.json`:

- `n_images` = frames done x batch (profiled frames excluded, as `derived.json` does);
- `latency_ms_per_image` = sum of measured step durations (preprocess, forward, postprocess) / n_images;
- `energy_mJ_per_image` = sum of measured step energies / n_images (gross, trapezoid over the meter
  samples; QC-excluded intervals removed on both duration and energy);
- `dyn_energy_mJ_per_image` = energy above the cell's own baseline power;
- `mean_P_W` = energy / duration over the same segment.

Acceptance: with QC exclusions switched off, it reproduces `derived.json` within 0.1 % on every v2
cell (energy and latency per frame); the size of the QC-exclusion effect is reported per cell. Unit tests cover the v1 layout, the v2 layout, the `profiler` exclusion and batch > 1.

### 4.2 QC rules

One script, `analytics.qc`, same rules for every cell, no rule informed by model results. Output:
`analytics/reports/qc.md` plus `qc_cells.csv` (one row per cell: decision, rules triggered).

| Rule | Check | Action |
|---|---|---|
| Q1 | `ctx_read` share inside scored segments (from `phase_energy.csv` `ctx_read_frac`) | Report; flag over 1 % |
| Q2 | Max alignment residual (`alignment.json`) | Include only below 1 ms |
| Q3 | Slips: count, slipped duration, overlap with scored segments | Exclude slipped intervals from scored segments, report; flag if over 1 % of a scored segment |
| Q4 | Throttle bits at start/end (`run_meta`, `context.csv`) | Current under-voltage or throttle bits: excluded from primary analysis, kept for a sensitivity row; history-only bits (0x50000) do not exclude |
| Q5 | `P_W` vs `V_V x I_A` per sample | Report median discrepancy per cell; bound set from the distribution across all cells before any model fit |
| Q6 | Trace integrity: monotonic `t_pi_ns`, sample count vs duration at the achieved rate, known phase vocabulary | Quarantine on failure, listed, never deleted |

Numeric bounds not given above are set in Phase 1 from the cross-cell distributions and recorded in
the protocol note before any model fit. Exit: every one of the 403 cells has a decision and a rule trail.

## 5. Exploration report (Phase 2)

`analytics/reports/eda.md` with figures, every number with n (configurations and cells) and the
dataset version:

1. Repeatability: CV per configuration (distribution), ICC; the noise floor used in every error table.
2. Clock scaling `P(f) = P_static + k f^alpha` per model as a description (three clocks, three
   parameters: no in-sample claim); idle baselines per clock, fan mode and governor.
3. Rig shift from `ref` vs v1 per configuration.
4. Interactions: threads x image size (1 thread is about 14 % more energy-efficient at 320 px),
   batch (per-image energy nearly flat), INT8 and FP16 against FP32.
5. Phase and step shares of energy and time; pre/post-processing discount.
6. Thermal: within-cell power vs temperature in long phases; soak trajectories (description, n = 1
   per model); cell-order residual check.
7. Counters (v2): rank and linear correlations of counter rates with 100 ms window power, raw and
   within-cell demeaned.
8. PMIC: rail-sum power vs Monsoon power at the 5 s reads, per family (v1 and v2).
9. Energy-accuracy trade-off: energy per image vs mAP50-95 (`map_reference`, measured on the Pi) for
   every YOLO configuration.
10. MiniGPT and training workloads, descriptive.

## 6. Mode A: a priori per-image predictor and the deployment calculator (Phase 3)

### 6.1 Target and unit

Target: `energy_mJ_per_image` (and, reported alongside, `latency_ms_per_image`) per configuration =
mean over its QC-included repeats. Configuration key: model, variant, image size, batch, threads,
clock, rig. Mode A trains on pinned-clock, fan-full cells only; governor and fan mode enter through the
deployment calculator (section 6.6), because only yolo11n (and yolo11s for ondemand) varies them.
Input source is not part of the key (section 2); cells that differ only in input merge into one
configuration.

### 6.2 Features (known before running anything)

- Static, from the FP32 base model at the run's image size: total MACs, parameters, weight bytes,
  activation bytes, layer counts and MAC fractions by layer type, depth. Read from FP32 cells'
  `layer_table.csv` (summing leaf rows). For a new model, `analytics/tools/static_features.py`
  runs one forward pass with the harness's own layer walker (`src/layers.py`) in the repo-root torch
  environment (NUC if available, else the Pi; it is not a measurement and does not need the Monsoon).
  Acceptance: for every model and image size in the data, it reproduces the Pi's `layer_table` MACs and
  parameters exactly.
- Variant: precision (fp32 / fp16 / int8), quantization mode, number of quantized modules. Variant
  cells take the FP32 base model's static features (the layer table's MACs, bytes and intensity are
  wrong for variants).
- Configuration: image size, batch, threads, clock (MHz), rig.
- Never: anything measured during the run (latency, temperature, baseline power, counters).

### 6.3 Splits

Groups are base models (all clocks, sizes and variants of a model together): yolov8n, yolov8s,
yolov8m, yolo11n, yolo11s, yolo11m, effb0, mbv3, resnet18. Split files are generated from
`run_meta.json`, written as data (configuration to fold), and hashed.

| Split | Folds | Similarity class |
|---|---|---|
| S-LOMO: leave one model out | 9 | `sibling-seen` for the YOLO folds (the same size class of the other YOLO generation is in training); `new-architecture` for the classifier folds |
| S-LSO: leave one YOLO size pair out | 3 (n pair, s pair, m pair) | `interpolation` (s pair, the original primary holdout) and `extrapolation` (n pair, m pair) |
| S-LVO: leave one variant out | yolo11s_int8s (yolo11n_int8s seen), yolo11n_int8s (yolo11s_int8s seen) | `variant-seen-on-other-model`; FP16 (one configuration, nothing similar) is reported as unsupported |
| S-LCO: leave one clock out | 1500, 2000, 2400; plus train 1500+2000, test 2400 | secondary |
| Random configuration split | 5-fold | pipeline sanity only, never a headline |

Each test configuration also gets a continuous novelty score: the distance in standardized static-
feature space to the nearest training model at the same image size. Error vs novelty is plotted
for every split.

### 6.4 Baselines and candidates (in order)

| # | Predictor |
|---|---|
| B0 | Median training energy per image |
| B1 | Energy proportional to MACs: `E = c(clock) x MACs`, `c` fitted on training configurations |
| B2 | Most similar seen model: energy of the nearest training model (novelty metric) at the same configuration, scaled by the MAC ratio |
| B3 | Log-linear: `log E` on log MACs, log params, clock, threads, batch, precision (OLS) |
| M1 | Ridge on all features, log target |
| M2 | Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`), log target |
| M3 | Factored: `log E = log t_hat + log P_hat`, each from the better of M1 and M2 on its own target (latency, mean power); scored on E |

Hyperparameters come from small fixed grids, written in the protocol note: ridge alpha over 7
values; trees over max depth {2, 3, 4}, learning rate {0.05, 0.1}, iterations {100, 300}, min
samples per leaf {3, 5}. Selection uses inner leave-one-model-out on the training models of each
outer fold. The model shipped for use is refitted on all data with the family and grid point most
often selected.

### 6.5 Metrics, intervals, comparisons

- Headline: S-LOMO, per fold and pooled over configurations: median absolute percentage error
  (MdAPE) of energy per image. Same table for latency. Then MAPE, MAE (mJ), signed bias.
- Every table carries the noise floor (median repeat CV) and n per subgroup (split, similarity
  class, model, variant, clock).
- Intervals: 90 % prediction intervals from inner-fold residual quantiles (cross-validation+ at
  model level); coverage reported per similarity class. With 9 models the coverage estimate is
  coarse, and every table says so.
- Confidence intervals on metrics: cluster bootstrap over models (2,000 resamples, seed from config).
- Paired comparisons: per test configuration, Wilcoxon signed-rank on absolute log errors between
  each candidate and the best baseline; Holm correction across that family. A difference below the
  noise floor is reported as indistinguishable.
- Applicability flag: a prediction is flagged `no-similar-model` when its novelty score exceeds the
  largest novelty seen among inner folds whose MdAPE stayed under 15 %. The threshold is computed
  inside each outer fold.
- Learning curve: S-LOMO MdAPE vs number of training YOLO models (2 to 5), and v2-only vs pooled
  v1+v2 training.

### 6.6 Deployment calculator

`analytics.deploy.predict(scenario)`. The scenario holds the model file or its static features,
variant, image size, batch, threads, clock or governor, fan mode, frame rate (fps, or `max`) and
duration (s).

```
t_img, E_img       <- Mode A model
rate_eff           = min(rate, 1 / t_img)            (images per second)
N                  = rate_eff x duration
E_total            = N x E_img + (duration - N x t_img) x P_idle(clock, governor, fan)
                     x soak_factor(duration)          (reported with and without)
```

- `P_idle` is the measured baseline table by clock, governor and fan mode (exploration report).
- Governor `ondemand`: active work is predicted at 2400 MHz (measured: ondemand forward energy equals
  pinned 2400 within 1 %), idle at the ondemand baseline.
- Fan `auto` / `off`: an active-power offset and an idle offset, both measured on yolo11n, are
  applied to `E_img` and `P_idle`.
- `soak_factor` comes from the soaks and is labelled as resting on one soak per model. It is
  validated leave-one-soak-out.
- Output: total energy (J), mean power (W), energy per image, the 90 % interval, the applicability
  flag and the energy-accuracy point where mAP is known.

Validation cells (never used to fit Mode A):
- yolo11n serve at 1, 5 and 15 fps: whole serve-phase energy including `idle_between_requests`;
- yolo11s ondemand: independent of the yolo11n cells the governor offset was read from;
- fan auto / off: offsets measured on yolo11n only, so they are an unvalidated assumption for other models;
- both soaks.

Concurrency cells are described only.

## 7. Mode L: live 100 ms power model (Phase 4)

### 7.1 Windows and causal join

Each window is one perf counter interval from `counters.csv`, about 100 ms, on the Pi clock; a trace
sample belongs to it when `t_start_ns < t_pi_ns <= t_end_ns` (the harness's own `merge_counters`
rule). The label is the mean of `P_W` over those samples (uniform sampling, so equal to the trapezoid
mean up to the slip handling). All samples are kept: live accounting must count what the meter saw, including sensor reads.
A window is dropped if more than 20 % of its duration is slipped or uncovered (count reported).

Features may only use data available at the window's end:
- the window's own counter interval and up to 3 previous intervals;
- the latest `sysfs.csv` row with timestamp at or before the window end (CPU clocks, temperature,
  fan state);
- the latest `context.csv` PMIC read that had finished by the window end, plus its age. A read's
  `t_ms` is taken before the read starts and a read takes 37-494 ms, so its availability time is the
  next read's time minus the 5 s logging period (the last read: `t_ms` + 500 ms). Amended 2026-11-23
  after review found the first version used reads up to 45 ms early;
- static clock, fan and governor settings.

A unit test proves no window uses a counter interval, sysfs row or PMIC read that ends after it
(including the perf start offset `offset_ns`), following each file's join contract in `run_meta.json`.

### 7.2 Ladder (in order)

| # | Model |
|---|---|
| L0 | Lookup: mean training power by (clock, governor, fan) x busy / idle, busy = task-clock share over 0.5 |
| L1 | Calibrated PMIC: `P = a + b x sum(I x V over rails)`, zero-order hold between 5 s reads; fitted on training folds |
| L2 | Ridge on task-clock rate x clock and cycles rate |
| L3 | Ridge and gradient-boosted trees on all counter rates (IPC, cache and branch miss rates, backend stall share, LLC misses, context switches, page faults), sysfs and settings |
| L4 | L3 plus the PMIC features |

L1 is also scored on v1 cells, its only cross-family test with current data.

### 7.3 Scoring

Same model groups as S-LOMO, restricted to the 6 YOLO models (v2 only), with inner leave-one-model-
out for selection. Headline: absolute percentage error of accumulated energy over a whole cell's
measured segment, median over held-out configurations. Also reported: horizons of 1 s and 10 s
(median absolute percentage error over all horizon blocks), bias, per-window MAE (W), per phase
(idle, load weights, warm-up, forward, serve, gaps) and per family (A, B, C, D, soak).

### 7.4 Probe mode

For a configuration with at least 2 repeats: run Mode L over repeat j's measured segment and divide by
its images to get energy per image, then score against repeat k's measured energy per image
(j != k), under S-LOMO. Probe mode is compared with Mode A on the same YOLO folds, which answers
whether a short unmetered run beats predicting from the model file.

### 7.5 Stated limits

- Counters cover the measured process's user-space code only (`perf_event_paranoid=2`). Other
  processes are invisible. Mode L claims hold for a dedicated single-process deployment.
- Counter data is YOLO-only until the lockbox (section 9) adds non-YOLO cells.
- Perf runs during deployment too, so its overhead is inside both training and use. The overhead
  figure from the v2 probes is quoted beside every Mode L result.

## 8. Reproducibility, code and safety

- Package `analytics/` at the repo root with its own `pyproject.toml` and `uv.lock` (polars,
  pyarrow, numpy, scipy, scikit-learn, matplotlib, pytest). No torch; `tools/static_features.py`
  runs in the repo-root environment.
- Modules: `data.py` (cell discovery, readers), `derive.py`, `qc.py`, `eda.py`, `features.py`,
  `splits.py`, `mode_a.py`, `deploy.py`, `windows.py`, `mode_l.py`, `evaluate.py` (metrics,
  bootstrap, paired tests, intervals), `report.py`, `cli.py`. Final names are fixed in the
  implementation plan.
- Built tables are cached under `~/pi-power-data/derived/<build-hash>/`; the build hash covers code
  commit, dataset manifests, the QC inclusion list and split files.
- Every model run appends one JSON line to `analytics/runs.jsonl`: run ID, time, commit, data and
  split hashes, mode, split, features version, family and hyperparameters, metrics per subgroup with
  intervals, coverage, seed, machine and wall time. Master seed in `analytics/config.toml`;
  headline results re-run under 3 seeds.
- Committed: reports (markdown and PNG figures), `runs.jsonl`, split files, QC tables, the protocol
  note, model cards. Not committed: built tables and model binaries (rebuilt by script).
- Tests run on the NUC (`cd analytics && uv run pytest`); fixtures are small synthetic cells and
  slices of real cells.
- Memory: reads stream one cell at a time with column projection; peak memory is reported in the run
  log. The NUC is not a measurement machine, so no lab safety rule applies to analysis runs.

## 9. Lockbox collection (Phase 5, after Phases 3 and 4 are frozen)

"Frozen" means: final Mode A and Mode L models refitted on all data and saved with hashes, the commit
tagged `analytics-frozen`. Only then:

| Cells | Purpose |
|---|---|
| yolo12n 640 px, 3 repeats | new model, `sibling-seen` (nano YOLO) |
| yolo12n 320 px, 3 repeats | new model at another image size |
| yolo11n-seg 640 px, 3 repeats | same backbone, different head (partly similar) |
| resnet18, 3 repeats, counters on | Mode L's only non-YOLO test (Mode A has seen resnet18 in v1) |
| mbv3, 3 repeats, counters on | same |

About 15 cells at 2400 MHz, fan full, 4 threads, about 1.5-2 h, run by the existing orchestrator on
the EliteBook with the usual lab rules. Pi Wi-Fi comes up only to download weights, then goes down
again before measuring. The power-cycle procedure is unchanged.
- **Feasibility gate:** a dry run of each new model through `src/run.py` before the matrix.
  yolo12n should load as an ultralytics model. yolo11n-seg may need a postprocess change for its
  mask output; if it cannot run without harness changes beyond a small, reviewed patch, it is
  replaced by yolov8n at 480 px (unseen size for that model).
- mAP for yolo12n is measured on the Pi with `tools/eval_map.py` if time allows (energy-accuracy
  point only).

The lockbox is scored exactly once by the frozen models. A failure is reported as the result; any
fix after scoring is labelled exploratory.

## 10. Phases and exit criteria

| Phase | Work | Exit |
|---|---|---|
| 0 | Copy and verify data; package skeleton; protocol note committed | Manifests verified; protocol note in git. Amendments allowed until the first model fit, each logged with a reason |
| 1 | Derivation and QC | `derive` matches `derived.json` within 0.1 % on all v2 cells; QC decision for all 403 cells |
| 2 | Exploration report | Every section of section 5 present, with n and dataset version |
| 3 | Mode A and deployment calculator | Baseline-ordered tables for every split, intervals and coverage, worst-10 configuration autopsy, calculator validation table |
| 4 | Mode L | Causal-join test passing; ladder table under S-LOMO; probe vs Mode A comparison |
| 5 | Freeze, lockbox, final report | Tag `analytics-frozen` before collection; lockbox scored once; final report and model cards; negative results present |

## 11. Out of scope

- T1 below 100 ms, T2 per-step energy, T4 per-layer energy attribution.
- Concurrency prediction, LM and training energy prediction.
- Other boards, meters, runtimes (ONNX, TFLite), system-wide counters.
- A live scheduler. Mode L ends as a validated model plus a function that turns a counter stream into
  power and energy.
