# Analytics and energy prediction — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** QC, exploration, an a priori per-image energy predictor with a deployment calculator, and a
live 100 ms power model for the Pi 5 Monsoon datasets, evaluated leave-one-model-out.

**Architecture:** A new `analytics/` subproject (own `pyproject.toml` / `uv.lock`, flat modules,
`tests/`), run only on the NUC against read-only dataset copies in `~/pi-power-data`. Config-level
tables (hundreds of rows) for Mode A; 100 ms windows built once from the traces for Mode L. A small
CLI writes reports to `analytics/reports/` and appends every model run to `analytics/runs.jsonl`.

**Tech Stack:** Python 3.12-3.13, pandas, pyarrow, numpy, scipy, scikit-learn, matplotlib, pytest, uv.
The static-feature tool lives in `tools/` (its env already has torch 2.14.0 and ultralytics 8.4.152).

**Spec:** `docs/superpowers/specs/2026-11-22-analytics-design.md` (read it; section numbers "spec §N"
below refer to it). Protocol note: `analytics/PROTOCOL.md` (written by the controller before Task 1).

**Branch:** `analytics` (from `main` a46b1ba plus the spec commit).

## Context every task needs

- **Data** (verified 2026-11-23 byte-for-byte against the GitHub dataset branches):
  `~/pi-power-data/v1/` (224 cells, `dataset_index.csv`, `README.md`, `cells/<cell_id>/`) and
  `~/pi-power-data/v2/` (179 cells; also `manifest.jsonl`, `results.jsonl`, `coverage.csv`).
  Later `~/pi-power-data/lockbox/` (same layout as v2). The root comes from env `ANALYTICS_DATA`
  (default `~/pi-power-data`). Treat it as read-only. Cells also contain `pi_console.log` and
  `pi_dmesg.log` (not published; ignore).
- **Per-cell files.** Both versions: `run_meta.json`, `alignment.json`, `phase_energy.csv`,
  `context.csv`, `layer_table.csv`, `op_table.csv`, `perop_power.csv`, `events.csv.gz`,
  `power_trace.parquet`. v2 only: `derived.json`, `sysfs.csv`, and `counters.csv` (absent or counters
  off in the 18 `ref` cells, whose ids start `v1ref_`). v1 has no `derived.json`, `sysfs.csv` or
  `counters.csv`.
- **`phase_energy.csv`**: one row per (phase, iter, step). Columns by name (v1 and v2 differ in order;
  v1 has no `n_finite`): `phase, iter, step, n_samples, [n_finite,] mean_P_W, ctx_read_frac,
  t_start_s, t_end_s, dur_ms, E_mJ`. Measured iterations of mode `m` are rows with `phase == m`
  (e.g. `forward`, `prefill`, `decode`, `train`) and numeric `iter`. Frame 0 is the cold first call
  and is excluded. A frame's energy/duration is the sum over all its rows (all steps, including an
  empty step and v2's `profiler` step if present in that frame). Baseline rows: `phase` in
  `baseline_A`, `baseline_B`. Serve rows: `phase == "serve"`, steps include `idle_between_requests`.
- **The reference derivation** is `host/verify_cell.py` `_frame_table` and `derived_metrics`
  (read them). `derived.json`: `energy_per_frame_mJ`, `latency_ms_per_frame`, `energy_per_image_mJ`
  (= per frame / batch), `dyn_energy_per_frame_mJ` (subtracts baseline x duration, baseline = mean of
  the `baseline_A` and `baseline_B` rows' `mean_P_W`), `n_frames`, `batch`, `by_mode`.
- **`run_meta.json`** keys used: `task` (`image`/`lm`/`train`), `modes`, `batch`, `imgsz`, `threads`,
  `variant`, `quant`, `executed_dtype`, `quantized_modules`, `serve_fps`, `serve_requests`,
  `soak_minutes`, `concurrent`, `governor`, `fan_mode`, `cpu_mhz_pinned`, `holdout`,
  `dataset_version`, `kernel` (v1 `6.18.50...`, v2 `6.18.60...`), `t0_real_ns` (Pi clock at trace
  start), `throttled_start`, `throttled_end`, `achieved_sample_hz`, `perf_counters` (dict with
  `enabled`, `interval_ms`, `offset_ns`, `events`), `max_temp_C`. Missing keys mean v1 defaults:
  governor `pinned`, fan `full`, threads 4, batch 1, quant `fp32`.
- **`dataset_index.csv`** (both versions) gives `model_key` (base model, e.g. `yolo11n` also for
  `yolo11n_int8s`), `family` (v2: `A`, `A-input`, `B`, `C`, `D`, `soak`, `ref`), `repeat`, `cpu_mhz`.
- **`alignment.json`**: `max_residual_ms`, `slips` (list of `{t_ns, step_ms, samples}`, `t_ns` on the
  Pi clock), `slip_samples_total`, `slip_rate_hz` (the sample rate; `dt_s = 1/slip_rate_hz`), `pass`.
- **`power_trace.parquet`** columns: `t_s, t_pi_ns, I_A, V_V, P_W, phase, mode, iter, step,
  layer_path, depth, ctx_read` and in counter cells `ctr_<event>`, `pct_<event>`, `sys_cpu{0..3}_kHz,
  sys_temp_C, sys_fan_state, sys_uv_alarm`. Always read with column projection
  (`pyarrow.parquet.read_table(path, columns=[...])`). A soak trace has about 9 M rows.
- **`counters.csv`**: `t_start_ns, t_end_ns` (Pi clock, already offset-corrected, about 100 ms
  apart, contiguous) then `ctr_cycles, ctr_instructions, ctr_cache_references, ctr_cache_misses,
  ctr_branch_misses, ctr_stalled_cycles_backend, ctr_LLC_load_misses, ctr_task_clock,
  ctr_context_switches, ctr_page_faults` and `pct_*` (percent of time counted). Counts are per
  interval. **`sysfs.csv`**: `t_real_ns, cpu0_kHz..cpu3_kHz, temp_C, fan_state, uv_alarm` (1 Hz).
  **`context.csv`**: `t_ms` (ms since `t0_real_ns`), `temp_C, arm_MHz, throttle_bits`, then 13 rail
  currents `rail_<NAME>_A current(<n>)` and voltages `rail_<NAME>_V volt(<n>)` (every 5 s).
- **`layer_table.csv`**: one row per (module path, distinct input signature): `path, class, leaf,
  params, macs, bytes_moved, count, ...`. `macs` is empty for some classes. `count` is calls over the
  profiled frames; per-inference multiplicity of a row is `round(count / min_count)` where
  `min_count` is the smallest positive `count` among leaf rows. MACs/bytes are wrong for INT8/FP16
  cells; static features always come from FP32 batch-1 cells.
- **Vision models** (Mode A groups): `yolov8n, yolov8s, yolov8m, yolo11n, yolo11s, yolo11m` (v2;
  yolov8n/yolo11n also v1) and `effb0, mbv3, resnet18` (v1, 224 px). Other v1 workloads
  (`lm16/64/128`, `train_*`) are described only.
- Tests: `cd analytics && uv run pytest -q`. Tests that need real data use the fixture
  `real_data` (Task 1) and are skipped when `~/pi-power-data/v2/cells` is absent.

## Global Constraints

1. Run only on the NUC. Never ssh to the Pi or EliteBook, never run a measurement, never write
   under `ANALYTICS_DATA` except `ANALYTICS_DATA/derived/`.
2. `analytics/pyproject.toml`: `requires-python = ">=3.12,<3.14"`, dependencies exactly `numpy,
   pandas, pyarrow, scipy, scikit-learn, matplotlib`; dev group `pytest`; `[tool.uv] package = false`;
   pytest `testpaths = ["tests"]`, `pythonpath = ["."]`. No other dependencies, no torch here.
3. Master seed `20261122` in `analytics/config.toml`; every stochastic step derives from it.
4. No fitting on test data: every model/scaler/lookup/threshold is fitted inside the training part
   of a fold. Functions that fit take only training rows.
5. Units and names: energy mJ, latency ms, power W, clock MHz; column names exactly as written here.
6. Commits: conventional prefix (`feat(analytics):`, `test(analytics):`, `docs(analytics):`), end the
   message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.
7. Matplotlib with the `Agg` backend; figures PNG at 150 dpi under `analytics/reports/fig/`.
8. Keep each module under about 400 lines; no notebooks.

## Review Focus

1. Optional files missing (v1 has no `derived.json`/`sysfs.csv`/`counters.csv`; `ref` cells no
   counters): every reader returns `None` / empty and callers skip, never crash. Tested in Task 1.
2. Batch > 1 and multi-mode cells: per-image = per frame / batch; `lm` cells give one row per mode and
   never enter Mode A. Tested in Task 2.
3. Concurrency cells: their energy covers both workloads; never Mode A eligible, labelled in every
   table that includes them. Tested in Tasks 1 and 5.
4. Slip `t_ns` outside every frame (e.g. during load_weights): excludes nothing, counted as
   "outside scored segments". Tested in Task 2.
5. Configurations with one cell (soaks, lockbox): CV is NaN and excluded from the noise floor.
   Tested in Task 5.

---

### Task 1: package skeleton, settings, cell catalogue, run log, CLI stub

**Files:** Create `analytics/pyproject.toml`, `analytics/config.toml`, `analytics/settings.py`,
`analytics/data.py`, `analytics/runlog.py`, `analytics/cli.py`, `analytics/tests/conftest.py`,
`analytics/tests/test_data.py`, `analytics/.gitignore` (`.venv/`). Generate `analytics/uv.lock`.

**Interfaces — Produces:**
- `data.VISION_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "effb0",
  "mbv3", "resnet18"]` (Mode A groups; `splits.VISION` re-exports it).
- `settings.data_root() -> Path` (env `ANALYTICS_DATA`, default `~/pi-power-data`, expanded);
  `settings.derived_dir() -> Path` (`data_root()/"derived"`, created on demand);
  `settings.seed() -> int` and `settings.cfg() -> dict` (parsed `config.toml`).
- `data.cell_dirs(root=None) -> list[Path]`: `root/{v1,v2,lockbox}/cells/*` directories, sorted.
- `data.read_meta(cell) -> dict`; `data.read_csv_opt(cell, name) -> pd.DataFrame | None` (accepts
  `name` or `name + ".gz"`); `data.read_json_opt(cell, name) -> dict | None`;
  `data.read_trace(cell, columns) -> pd.DataFrame` (pyarrow projection; string columns as plain str).
- `data.catalogue(root=None) -> pd.DataFrame`, one row per cell, columns exactly: `cell_id, source`
  (`v1`/`v2`/`lockbox`, from the parent folder), `path, task, model_key, base_model, variant,
  precision` (`fp32`/`fp16`/`int8` from `quant`/`executed_dtype`), `quant, n_quantized_modules`
  (len of `quantized_modules` or 0), `imgsz, batch, threads, cpu_mhz` (int: `cpu_mhz_pinned`, else the
  index `cpu_mhz`), `governor, fan_mode, serve_fps` (float, NaN if none), `soak_minutes` (float/NaN),
  `concurrent_mode` (str or None: `concurrent["mode"]`), `input_source, family` (index `family`; v1 ->
  `"v1"`), `repeat, rig` (`oct2026` if `kernel` starts `6.18.50`, else `nov2026`), `dataset_version`
  (int, v1 cells -> 1), `holdout, has_counters` (`counters.csv` present and `perf_counters.enabled`),
  `mode_a_eligible, mode_l_eligible`.
  `base_model = model_key`; `variant` defaults to `model_key`. `mode_a_eligible` = task `image` and
  base_model in the 9 vision models and serve_fps NaN and soak NaN and concurrent None and governor
  `pinned` and fan `full`. `mode_l_eligible = has_counters`.
- `runlog.append(record: dict, path=None)`: adds `run_id` (uuid4 hex), `ts` (UTC ISO),
  `commit` (`git rev-parse HEAD`), `dirty` (bool), `data_hash` (sha256 over the bytes of every
  `<root>/<source>/dataset_index.csv` present, sorted by path), `machine` (hostname), then appends one
  JSON line to `analytics/runs.jsonl`. `runlog.file_hash(path) -> str` (sha256 hex).
- `cli.py`: argparse with subcommands registered by later tasks; this task adds
  `catalogue` (writes `analytics/reports/catalogue.csv`, prints counts by source and family).

**Requirements / tests** (`tests/conftest.py` provides `make_cell(tmp_path, source, cell_id,
meta=..., phase_rows=..., alignment=..., derived=None, files=...)` writing a minimal synthetic cell,
and `real_data` fixture that skips if `data_root()/"v2"/"cells"` is missing):
- [ ] synthetic v1 cell (no derived/sysfs/counters) and v2 cell -> catalogue rows with the defaults
  above; `read_csv_opt` returns None for missing files and reads `events.csv.gz`.
- [ ] concurrency, serve_fps, soak, ondemand, fan auto cells are not `mode_a_eligible`; an INT8
  `yolo11n_int8s` cell is eligible with `base_model == "yolo11n"`, `precision == "int8"`.
- [ ] real data: 224 `v1` + 179 `v2` rows; every `v1ref_*` cell has `rig == "nov2026"` and
  `has_counters == False`; no `v1` cell has counters; every other v2 cell has `has_counters == True`
  (161 cells).
- [ ] `runlog.append` writes one valid JSON line with all keys (use a tmp path).
- [ ] Commit.

### Task 2: one derivation for v1 and v2 (`derive.py`)

**Files:** Create `analytics/derive.py`, `analytics/tests/test_derive.py`; add CLI `derive`.

**Interfaces — Consumes:** `data.read_meta`, `read_csv_opt`, `read_json_opt`, `catalogue`.
**Produces:**
- `derive.frame_table(pe: pd.DataFrame, mode: str) -> pd.DataFrame` indexed by frame (int >= 1):
  `E_mJ, dur_ms, t_start_s, t_end_s` (sum, sum, min, max over the frame's rows).
- `derive.slip_times_s(alignment: dict, meta: dict) -> list[tuple[float, float]]`: each slip as
  `(t_s, step_ms)` with `t_s = (t_ns - t0_real_ns) / 1e9`; empty if `t0_real_ns` or slips missing.
- `derive.derive_cell(cell: Path) -> list[dict]`, one dict per measured mode with keys:
  `cell_id, mode, batch, n_frames, n_frames_slip_excluded, n_images, energy_mJ_per_image,
  latency_ms_per_image, mean_P_W, dyn_energy_mJ_per_image, energy_mJ_per_image_raw,
  latency_ms_per_image_raw, baseline_W, baseline_A_W, baseline_B_W, serve_n_requests, serve_dur_ms,
  serve_E_mJ, serve_idle_dur_ms, serve_idle_E_mJ, trace_E_J, trace_dur_s, slips_total,
  slips_in_scored`.
  - `*_raw` reproduce `derived.json`: mean over frames of frame E (and dur), divided by batch.
  - Primary values drop every frame whose `[t_start_s, t_end_s]` overlaps a slip interval
    `[t_s, t_s + step_ms/1000]`; `mean_P_W = sum(E) / sum(dur)` over kept frames (W = mJ/ms).
  - `dyn_energy_mJ_per_image = mean(E - baseline_W * dur) / batch` over kept frames.
  - `baseline_W` = mean of the `baseline_A` / `baseline_B` rows' `mean_P_W` that exist (NaN if none).
  - serve fields from `phase == "serve"` rows (NaN/0 when no serve phase); `serve_n_requests` = number
    of distinct numeric `iter`; idle = rows with `step == "idle_between_requests"`.
  - `trace_E_J = sum(E_mJ)/1000` and `trace_dur_s = sum(dur_ms)/1000` over all rows.
  - `slips_in_scored` = slips overlapping any measured frame of any mode.
- `derive.derive_all(cat: pd.DataFrame) -> pd.DataFrame` (one row per cell and mode; cached as
  `derived_dir()/derive.parquet` keyed by `runlog.file_hash` of `derive.py` plus the data hash;
  `force=False` parameter).

**Tests:**
- [ ] synthetic v2 layout with a `profiler` row in frame 0 and an empty-step row in frame 2: raw
  values equal a hand computation; batch 2 halves per-image values.
- [ ] synthetic v1 layout (no `n_finite`, different column order) gives the same numbers as the same
  data in v2 layout.
- [ ] a slip inside frame 3 drops only frame 3 (`n_frames_slip_excluded == 1`); a slip during
  `load_weights` drops nothing and `slips_in_scored == 0`.
- [ ] `lm` cell with modes `prefill, decode` -> two rows.
- [ ] real data: for every v2 cell with one mode, `energy_mJ_per_image_raw` and
  `latency_ms_per_image_raw * batch` match `derived.json` `energy_per_image_mJ` and
  `latency_ms_per_frame` within 0.1 % (report the worst relative difference in the test output).
- [ ] Commit.

### Task 3: QC (`qc.py`)

**Files:** Create `analytics/qc.py`, `analytics/tests/test_qc.py`; add `[qc]` to `config.toml`;
add CLI `qc` writing `analytics/reports/qc_cells.csv` and `analytics/reports/qc.md`.

**Config `[qc]` initial values** (the controller may amend them in `PROTOCOL.md` after seeing the
distributions, before any model fit): `q1_ctx_frac_flag = 0.01`, `q2_max_residual_ms = 1.0`,
`q3_slip_frac_flag = 0.01`, `q5_pv_rel_flag = 0.02`, `q6_sample_ratio_lo = 0.98`,
`q6_sample_ratio_hi = 1.02`.

**Interfaces — Consumes:** catalogue row, `derive` row(s) of the cell, `data.read_trace`.
**Produces:** `qc.qc_cell(row: pd.Series, drows: pd.DataFrame, bounds: dict) -> dict` and
`qc.run_qc(cat, der) -> pd.DataFrame` with columns: `cell_id, include, primary, rules` (sorted
`;`-joined rule ids triggered, e.g. `Q1flag;Q4`), `max_residual_ms, n_slips, slip_samples,
slips_in_scored, slip_frames_excluded, slip_scored_frac` (excluded frames' duration / all measured
frames' duration), `ctx_read_frac_scored` (duration-weighted mean of `ctx_read_frac` over the measured
rows), `throttle_start, throttle_end, throttle_current` (bool), `pv_median_rel, n_samples,
expected_samples, sample_ratio, monotonic, dup_timestamps, unknown_phases, start_temp_C, max_temp_C`.
- Q1: `ctx_read_frac_scored > q1` -> `Q1flag` (flag only).
- Q2: `max_residual_ms >= q2` -> `Q2` (exclude).
- Q3: `slip_scored_frac > q3` -> `Q3flag`.
- Q4: current bits = `bits & 0xF` non-zero in `throttled_start`, `throttled_end`, any
  `context.csv` `throttle_bits`, or any `sysfs.csv` `uv_alarm == 1` -> `Q4` (kept, `primary` False).
  `0x50000`-style history bits alone do not trigger.
- Q5: on every 50th trace row, `median(|P_W - V_V * I_A| / P_W)` -> `pv_median_rel`; above q5 -> `Q5flag`.
- Q6: `t_pi_ns` strictly increasing (`monotonic`), `dup_timestamps` count, `sample_ratio = n_samples /
  (trace span s * achieved_sample_hz)` outside `[lo, hi]`, or a phase label outside the known
  vocabulary -> `Q6` (quarantine = exclude). Known vocabulary: `"", idle, load_weights, warmup,
  sync:start, sync:end, baseline_A, baseline_B, gap, serve, concurrent_drain, forward, prefill, decode,
  train` and any label starting `first:` or `bench:`. Before finalising, the implementer lists every
  distinct phase label in the real data in the report; any label not in this list is reported to the
  controller, not silently added.
- `include = not (Q2 or Q6)`; `primary = include and not Q4`.
- `qc.md`: counts per rule, the quarantined / excluded list with reasons, quantiles (min, p50, p95,
  max) of `max_residual_ms, slip_scored_frac, ctx_read_frac_scored, pv_median_rel, sample_ratio`.
- Trace reads only `t_pi_ns, P_W, V_V, I_A, phase`. Log peak RSS (`resource.getrusage`) and wall
  time to the CLI output.

**Tests:**
- [ ] synthetic cells trigger each rule exactly as specified, and a clean cell triggers none.
- [ ] history-only throttle `0x50000` does not trigger Q4; `0x50005` does.
- [ ] real data (`real_data`, marked slow): `run_qc` over 3 named cells (one v1, one v2, one soak:
  `yolo11n_soak30_640_t4_2400_r00`) returns a row each with no exception.
- [ ] Commit.

### Task 4: static model features (`static.py`, `tools/static_features.py`)

**Files:** Create `analytics/static.py`, `analytics/tests/test_static.py`, `tools/static_features.py`,
`tools/tests/test_static_features.py`.

**Interfaces — Produces:**
- `static.CLASS_GROUPS: dict[str, str]` mapping layer class to group: conv (`Conv2d`,
  `ConvTranspose2d`), linear (`Linear`), norm (`BatchNorm2d`, `BatchNorm1d`, `LayerNorm`,
  `GroupNorm`), act (`SiLU`, `ReLU`, `ReLU6`, `Hardswish`, `Hardsigmoid`, `Sigmoid`, `GELU`, `Tanh`),
  pool (`MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`), other (anything else).
  `static.GROUPS = ["conv", "linear", "norm", "act", "pool", "other"]`.
- `static.features_from_layer_table(lt: pd.DataFrame, batch: int = 1) -> dict` over leaf rows:
  `mult = round(count / min_count)`; `macs = sum(macs * mult) / batch` (NaN macs -> 0);
  `params = sum(params)` over unique leaf paths; `weight_bytes = 4 * params`; `act_bytes =
  sum((bytes_moved - 4 * params_row) * mult) / batch` (clip at 0); `depth = max(path.count(".")) + 1`;
  `n_leaf`; for each group `n_<g>` (leaf rows in group, each unique path once) and `frac_macs_<g>`
  (group MACs / total MACs).
- `static.static_table(cat: pd.DataFrame) -> pd.DataFrame` indexed by `(base_model, imgsz)` from FP32
  batch-1 Mode A eligible cells' `layer_table.csv` (first by cell_id; assert all candidates agree on
  `macs` within 0.1 %, else raise). Also `static.model_descriptor(st: pd.DataFrame) -> pd.DataFrame`
  indexed by `base_model`: `log_params, log_macs_per_px` (`log(macs / imgsz**2)`), `depth` and the
  six `frac_macs_*`, taken at imgsz 640 if present, else the model's only imgsz.
- `tools/static_features.py --model <yolo11n.pt | torchvision:resnet18> --imgsz N --out <csv>`:
  builds the FP32 model exactly as `src/tasks.py` does for `--task image` (import from `src/` by
  adding the repo's `src` to `sys.path`), wraps `task.module` in `src/layers.py` `LayerProfiler`, runs
  2 forward passes on a zeros input `(1, 3, N, N)` under `torch.no_grad()`, and writes
  `prof.aggregated(exclude_frame=-1)` rows in the `layer_table.csv` column format. It needs no Pi and
  no Monsoon.

**Tests:**
- [ ] synthetic layer table: multiplicity (a MaxPool called 3x per frame), NaN macs, batch 2 and group
  fractions summing to 1.
- [ ] real data: `static_table` has every (base_model, imgsz) of Mode A configs (yolo11n/s/m at
  320/480/640/960, yolov8n/s/m at 640, effb0/mbv3/resnet18 at 224) and yolo11n@640 `macs` is between
  2.5e9 and 4.5e9.
- [ ] `tools/tests`: for yolo11n.pt at 640 (skip if `yolo11n.pt` or the real cell is absent) the tool's
  leaf `macs` sum and `params` sum equal those of `~/pi-power-data/v2/cells/v1ref_yolo11n_2400_r00/
  layer_table.csv` exactly. Run with `cd tools && uv run pytest tests/test_static_features.py -q`
  (the repo-root `yolo11n.pt` exists on the NUC).
- [ ] Commit.

### Task 5: Mode A configuration table (`tables.py`)

**Files:** Create `analytics/tables.py`, `analytics/tests/test_tables.py`; add CLI `tables`
(writes `analytics/reports/mode_a_configs.csv`).

**Interfaces — Consumes:** catalogue, `derive_all`, `run_qc`, `static_table`.
**Produces:**
- `tables.CONFIG_COLS = ["base_model", "variant", "precision", "imgsz", "batch", "threads",
  "cpu_mhz", "rig"]`; `tables.config_key(row) -> str` = `"|".join(str(row[c]) for c in CONFIG_COLS)`.
- `tables.mode_a_table(cat, der, qc, st) -> pd.DataFrame`, one row per configuration over cells that
  are `mode_a_eligible` and QC `primary`, mode `forward`: `config_key`, CONFIG_COLS, `quant,
  n_quantized_modules, n_cells, cell_ids` (`;`-joined), `energy_mJ, latency_ms, mean_P_W,
  dyn_energy_mJ` (means of the per-image primary values), `energy_cv, latency_cv` (sample SD / mean;
  NaN when n_cells == 1), `family_set` (`;`-joined families), then the static features of
  `(base_model, imgsz)` (all `features_from_layer_table` keys) and the derived columns `log_macs,
  log_params, log_weight_bytes, log_act_bytes, cpu_ghz` (= cpu_mhz / 1000), `is_int8, is_fp16`.
- `tables.noise_floor(t) -> float` = median `energy_cv` over rows with `n_cells >= 2`.

**Tests:**
- [ ] synthetic: two cells differing only in `input_source` merge into one configuration; a
  one-cell configuration has NaN CV and is ignored by `noise_floor`; a QC-excluded cell is dropped; a
  concurrency cell never appears.
- [ ] real data: >= 60 configurations, no NaN `macs`, every configuration's base_model in the 9
  vision models, and `noise_floor` between 0.0001 and 0.03.
- [ ] Commit.

### Task 6: split battery (`splits.py`)

**Files:** Create `analytics/splits.py`, `analytics/tests/test_splits.py`; add CLI `splits`
(writes `analytics/splits/<split>.csv` and prints each file's sha256).

**Interfaces — Produces:**
- `splits.VISION = data.VISION_MODELS`, `splits.YOLO = VISION[:6]`.
- `@dataclass Fold: split: str; fold: str; similarity_class: str; train_keys: list[str];
  test_keys: list[str]`.
- `splits.make_splits(t: pd.DataFrame, seed: int) -> dict[str, list[Fold]]` with keys:
  - `S-LOMO`: one fold per base model present; test = its configs, train = the rest.
    Class `sibling-seen` for YOLO, `new-architecture` for effb0/mbv3/resnet18.
  - `S-LSO`: folds `n` (yolov8n+yolo11n), `s`, `m`; test = both models' configs, train = all configs
    of the other base models (classifiers included). Class `interpolation` for `s`, `extrapolation`
    for `n` and `m`.
  - `S-LVO`: one fold per non-FP32 variant (`yolo11n_int8s`, `yolo11s_int8s`, `yolo11n_fp16`); test =
    that variant's configs; train = everything else. Class `variant-seen-on-other-model` for INT8,
    `unsupported` for FP16.
  - `S-LCO`: folds `1500`, `2000`, `2400` (test = configs at that clock, train = the other clocks) and
    `1500+2000->2400` (train = configs at 1500 or 2000, test = 2400). Class `clock-interpolation` for
    2000, `clock-extrapolation` otherwise.
  - `RANDOM`: 5-fold `KFold(shuffle=True, random_state=seed)` over configs; class `sanity`.
- `splits.write_splits(sp, outdir) -> dict[str, str]` (CSV columns `split, fold, similarity_class,
  config_key, role` with role `train`/`test`; returns sha256 per file).
- `splits.inner_folds(train_t: pd.DataFrame) -> list[tuple[list[str], list[str]]]`:
  leave-one-base-model-out over the training configs (train_keys, val_keys); if fewer than 3 base
  models, leave-one-config-out.

**Tests:**
- [ ] every fold: train and test disjoint; in S-LOMO / S-LSO no base model appears on both sides;
  in S-LVO no test variant appears in train.
- [ ] each config is test exactly once in S-LOMO and RANDOM; S-LSO `s` test = all yolov8s and
  yolo11s configs.
- [ ] `inner_folds` never returns a key outside `train_t`.
- [ ] same seed -> identical files and hashes.
- [ ] Commit.

### Task 7: evaluation utilities (`evaluate.py`)

**Files:** Create `analytics/evaluate.py`, `analytics/tests/test_evaluate.py`.

**Interfaces — Produces:**
- `evaluate.metrics(y, yhat) -> dict` with `n, mdape, mape, mae, rmse, bias_pct` (median of signed %
  error `(yhat - y) / y`), `mdale` (median |log(yhat/y)|). Percent values as fractions (0.05 = 5 %).
- `evaluate.cluster_bootstrap(df, group_col, stat, n=2000, seed) -> tuple[float, float]`: resample
  groups with replacement, recompute `stat(df_resampled)`, 2.5 / 97.5 percentiles.
- `evaluate.paired_wilcoxon(err_a, err_b) -> dict(stat, p, median_diff)` (scipy
  `wilcoxon(zero_method="wilcox")`; when all differences are 0 return p = 1).
- `evaluate.holm(pvals: list[float]) -> list[float]` (step-down adjusted, monotone, capped at 1).
- `evaluate.coverage(y, lo, hi) -> float`.
- `evaluate.subgroup_table(pred: pd.DataFrame, by: list[str], y="y_true", yhat="y_pred") ->
  pd.DataFrame` (one row per group: metrics + coverage when `lo90`/`hi90` present).

**Tests:**
- [ ] metrics on hand-computed arrays; `holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]`.
- [ ] bootstrap deterministic for a fixed seed and the interval contains the point estimate on a
  simple example.
- [ ] Commit.

### Task 8: Mode A candidates and novelty (`mode_a_models.py`)

**Files:** Create `analytics/mode_a_models.py`, `analytics/tests/test_mode_a_models.py`.

**Interfaces — Consumes:** the `tables.mode_a_table` columns; `static.model_descriptor`.
**Produces:** candidates sharing `fit(train: pd.DataFrame, target: str, params: dict) -> self` and
`predict(test: pd.DataFrame) -> np.ndarray` (positive values in the target's units). `target` is one
of `energy_mJ`, `latency_ms`, `mean_P_W`. Learned candidates fit `log(target)`.
- `FEATURES = ["log_macs", "log_params", "log_weight_bytes", "log_act_bytes", "depth", "frac_macs_conv",
  "frac_macs_linear", "frac_macs_norm", "frac_macs_act", "frac_macs_pool", "frac_macs_other", "imgsz",
  "batch", "threads", "cpu_ghz", "is_int8", "is_fp16", "n_quantized_modules"]` plus `rig_nov`
  (1 if rig == nov2026).
- `B0Median` (median of train target), `B1MacsProportional` (`c = median(target / macs)` over train
  rows at the same `cpu_mhz`; nearest clock if none; prediction `c * macs`), `B2NearestModel`
  (candidate train rows matching the test row on `imgsz, batch, threads, cpu_mhz, precision`, relaxed
  in this order if empty: drop `threads`, then `batch`, then `precision`, then use the nearest `cpu_mhz`;
  among them the base model with the smallest novelty distance; prediction = that row's target scaled
  by `macs_test / macs_nn` for energy and latency, unscaled for power), `B3LogLinear` (OLS of log
  target on `log_macs, log_params, cpu_ghz, threads, batch, is_int8, is_fp16`), `M1Ridge`
  (StandardScaler + Ridge on FEATURES, grid `alpha` in `numpy.logspace(-2, 4, 7)`), `M2Trees`
  (`HistGradientBoostingRegressor`, grid `max_depth` {2, 3, 4} x `learning_rate` {0.05, 0.1} x
  `max_iter` {100, 300} x `min_samples_leaf` {3, 5}, `random_state` = seed), `M3Factored` (fits a
  latency model and a power model, each chosen from {M1 best, M2 best} by the caller; prediction =
  latency_hat x power_hat; energy only).
- `CANDIDATES: dict[str, type]` and `GRIDS: dict[str, list[dict]]` (empty grid = `[{}]`).
- `novelty_distance(desc: pd.DataFrame, test_model: str, train_models: list[str]) -> float`:
  standardize descriptor columns with mean/SD over the training models (SD floor 1e-6), return the
  minimum Euclidean distance from the test model to any training model (0 if it is one of them).

**Tests:**
- [ ] B1 is exact on data where energy = c(clock) x macs; B3 recovers log-linear coefficients within
  1e-6 on noiseless data; M1/M2 fit and predict positive values; M3 equals the product of its parts.
- [ ] B2 picks the matching row and applies the MAC ratio; relaxation order is followed.
- [ ] `novelty_distance` is 0 for a training model and grows with descriptor distance.
- [ ] no candidate reads a column outside FEATURES + CONFIG_COLS + target + `macs`.
- [ ] Commit.

### Task 9: Mode A nested evaluation, intervals, applicability, learning curve (`mode_a.py`)

**Files:** Create `analytics/mode_a.py`, `analytics/tests/test_mode_a.py`; add CLI `mode-a
[--split S] [--target energy_mJ|latency_ms|mean_P_W] [--seed N]` writing
`ANALYTICS_DATA/derived/mode_a/<run_id>.parquet` and one `runlog` line per (split, target).

**Interfaces — Consumes:** Tasks 5-8. **Produces:**
- `mode_a.run_split(t, folds: list[Fold], target: str, seed: int, desc: pd.DataFrame) ->
  pd.DataFrame` with one row per (fold, test config, candidate): `split, fold, similarity_class,
  config_key, base_model, candidate, y_true, y_pred, lo90, hi90, novelty, flagged, params` (JSON of the
  chosen grid point), plus a meta-candidate `AUTO` row per config.
  For each outer fold, using only `fold.train_keys` rows:
  1. For each candidate and grid point, predict every inner validation config with `inner_folds`;
     choose the grid point with the lowest pooled inner MdAPE. For M3, its latency and power parts are
     the best of M1/M2 on their own targets by the same rule.
  2. Refit the chosen point on all training rows; predict the test configs.
  3. Interval: inner residuals `r = log(y) - log(yhat)` of the chosen point;
     `lo90 = yhat * exp(quantile(r, 0.05))`, `hi90 = yhat * exp(quantile(r, 0.95))`.
  4. `AUTO` = whichever of M1, M2, M3 had the lowest inner MdAPE (for latency/power targets: M1 or M2).
  5. Novelty of the test model vs the fold's training models; applicability threshold = the largest
     inner-fold novelty (each inner held-out model vs its inner training models) whose inner MdAPE
     (AUTO) was below 0.15 (0 if none); `flagged = novelty > threshold`.
- `mode_a.fit_final(t, candidate: str, target: str, seed) -> object`: grid chosen by leave-one-model-
  out over all of `t`, refit on all rows; saved with `pickle` under `derived/models/` with the file's
  sha256 logged.
- `mode_a.learning_curve(t, target, seed, max_subsets=10) -> pd.DataFrame`: for each YOLO model held
  out, train on k of the remaining YOLO models (k = 2..5; at most `max_subsets` random subsets per k
  with the seed) plus all classifier configs; rows `heldout, k, subset, mdape`.

**Tests:**
- [ ] a spy candidate records the keys it was fitted on: no test key ever appears, and inner fits never
  see the inner validation keys.
- [ ] on a synthetic 6-model dataset with energy = k x macs^0.9 x clock factor x (1 + 1 % noise),
  `AUTO` MdAPE on S-LOMO is below 0.1 and 90 % coverage is between 0.6 and 1.0.
- [ ] every test config appears once per (fold, candidate); `flagged` is True for a deliberately alien
  model descriptor.
- [ ] Commit.

### Task 10: deployment calculator and its validation (`deploy.py`)

**Files:** Create `analytics/deploy.py`, `analytics/tests/test_deploy.py`; add CLI `deploy-validate`
writing `analytics/reports/deploy_validation.csv`.

**Interfaces — Consumes:** catalogue, derive, QC, Mode A predictions/final models. **Produces:**
- `deploy.idle_table(cat, der, qc) -> pd.DataFrame` keyed by `(cpu_key, fan_mode)` where `cpu_key` is
  `str(cpu_mhz)` for pinned cells and `"ondemand"` for ondemand cells: `P_idle_W` (median baseline_W
  over primary cells), `n, iqr_W`. Only `nov2026` rig cells.
- `deploy.fan_offsets(t_cells) -> dict`: for fan `auto` and `off`: `active_W` = mean_P_W(fan mode) -
  mean_P_W(full) and `idle_W` = idle difference, both from yolo11n 640 px, 4 threads, 2400 MHz,
  `nov2026` cells. Labelled `assumption: measured on yolo11n only`.
- `deploy.soak_factor(der, cat, exclude_cell=None) -> float`: mean over available soaks (minus the
  excluded one) of soak energy per image / energy per image of the matching non-soak configuration at
  the same base model, 640 px, 4 threads, 2400 MHz, fan full; applied as
  `1 + (f - 1) * min(1, duration_s / 1800)` (assumption, stated in the output).
- `deploy.predict(scenario: dict, e_img_mJ: float, t_img_ms: float, idle: pd.DataFrame,
  offsets: dict, soak: float | None) -> dict`. Scenario keys: `cpu_key, fan_mode, duration_s` and
  either `rate_fps` (float or `"max"`) or `n_images`. Formula (spec §6.6):
  `rate_eff = min(rate, 1000 / t_img_ms)`, `N = rate_eff * duration_s` (or given `n_images`),
  `busy_s = N * t_img_ms / 1000` (must not exceed duration: clip and set `saturated=True`),
  `E_J = (N * e_img_mJ / 1000 + (duration_s - busy_s) * P_idle) * soak_factor`. Ondemand: e_img at
  2400 MHz, P_idle at `"ondemand"`. Fan offsets add `active_W * busy_s` and adjust P_idle. Returns
  `E_J, E_J_no_soak, mean_P_W, n_images, saturated, assumptions` (list of strings).
- `deploy.validate(...) -> pd.DataFrame` rows `case, cell_id, measured_J, predicted_J, ape, source,
  notes`, where predictions use the S-LOMO `AUTO` fold prediction for the cell's configuration
  (`source = "lomo"`), never an in-sample fit:
  1. `serve-default`: every Mode A primary cell's serve phase (`n_images = serve_n_requests * batch`,
     `duration_s = serve_dur_ms / 1000`), measured = `serve_E_mJ / 1000`.
  2. `serve-fps`: the yolo11n serve 1 / 5 / 15 fps cells, same mapping.
  3. `ondemand`: yolo11s ondemand cells (forward phase: per-image energy vs prediction with ondemand
     rules; and the serve phase).
  4. `soak`: each soak's forward phase energy with the soak factor from the other soak only.
  5. `fan`: fan auto / off cells, `notes = "offsets measured on these cells: not a validation"`.

**Tests:**
- [ ] `predict` hand-computed cases: saturation at 15 fps with 290 ms latency; idle-only (rate 0);
  ondemand uses the ondemand idle row; soak factor ramps with duration.
- [ ] `idle_table` ignores `oct2026` cells and non-primary cells.
- [ ] Commit.

### Task 11: 100 ms windows and PMIC windows (`windows.py`)

**Files:** Create `analytics/windows.py`, `analytics/tests/test_windows.py`; add CLI `windows`
(builds and caches `derived/windows.parquet` and `derived/pmic_windows.parquet`).

**Interfaces — Produces:**
- `windows.cell_windows(cell: Path, meta: dict) -> pd.DataFrame | None` (None without counters). One
  row per `counters.csv` interval: `cell_id, w_idx, t_start_ns, t_end_ns, dur_s, n_samples, cover_frac`
  (= n_samples x dt_s / dur_s), `P_W` (mean of trace `P_W` with `t_start_ns <= t_pi_ns < t_end_ns`),
  `E_mJ` (= sum P_W x dt_s x 1000), `phase` (majority label), `phase_frac`, counter features:
  `busy` (= ctr_task_clock / (dur_s x 4 x U) where U is the task-clock unit factor; the implementer
  verifies on real data that the 99th percentile of busy over forward-phase windows lies in
  [0.85, 1.05] with U for nanoseconds, and reports if not), `cycles_rate, instr_rate` (per s),
  `ipc, cache_miss_rate, branch_mpki, stall_frac, llc_mpki, ctx_sw_rate, pf_rate` (zero denominators ->
  NaN; count of guarded windows returned in `df.attrs["guards"]`), lags `busy_l1..l3, cycles_rate_l1..l3,
  ipc_l1..l3` (previous intervals of the same cell; NaN at the start), `pct_min` (min of the `pct_*`).
  sysfs as of `t_end_ns` (latest `t_real_ns <= t_end_ns`): `freq_mhz` (mean of cpu0-3 kHz / 1000),
  `temp_C, fan_state, uv_alarm, sysfs_age_s`. PMIC as of `t_end_ns` (latest context read with
  `t0_real_ns + t_ms x 1e6 <= t_end_ns`): `pmic_W` (sum over the 13 rails of current x voltage, rails
  paired by name), `pmic_age_s`. Settings: `cpu_key, fan_mode, governor` from the catalogue row.
  Windows with `cover_frac < 0.8` are dropped; the count is in `df.attrs["dropped"]`.
- `windows.pmic_windows(cell: Path, meta: dict) -> pd.DataFrame`: for every cell (v1 and v2), one row
  per pair of consecutive context reads `[t_i, t_{i+1})`: `cell_id, t_start_ns, t_end_ns, dur_s,
  P_W` (trace mean), `E_mJ, phase, pmic_W` (from read i, known at the window start), `cpu_key,
  fan_mode, governor`.
- `windows.build_windows(cat, force=False)` and `build_pmic_windows(cat, force=False)` (cache under
  `derived/`, one cell at a time; log peak RSS).

**Tests:**
- [ ] synthetic cell with a sysfs row 1 ns after a window end and a PMIC read exactly at the window end:
  the sysfs row is not used for that window, the PMIC read is (`<=`).
- [ ] lags never come from a later interval; first window lags are NaN.
- [ ] `P_W` equals the trace mean on a synthetic trace; a window with half its samples missing is
  dropped.
- [ ] real data: on one v2 cell, every window has `sysfs_age_s >= 0`, `pmic_age_s >= 0`, and the
  window energy sum is within 1 % of the trace energy between the first and last window.
- [ ] Commit.

### Task 12: exploration report, configuration level (`eda.py`, part 1)

**Files:** Create `analytics/eda.py`, `analytics/plotstyle.py` (one shared style: font size, colour
cycle, `save(fig, name)`), `analytics/tests/test_eda.py`; add CLI `eda` writing
`analytics/reports/eda.md` and figures.

**Contents** (spec §5 items 1-5, 9, 10; every number with n configurations / cells and dataset
version): repeatability (CV distribution per configuration, ICC(1,1) from a one-way random-effects
ANOVA over configurations with >= 2 cells, the noise floor); clock scaling fit
`P = P_static + k * f^alpha` per model with scipy `curve_fit` (description only, say so), and the same
for idle; idle table by clock, governor, fan; rig shift `ref` vs v1 per configuration (energy, latency,
power, % difference); interactions (threads x imgsz energy grid for yolo11n/yolo11s, batch per-image
energy, INT8/FP16 vs FP32 ratios); phase and step shares of energy and time per workload (from
`phase_energy.csv`); energy per image vs mAP50-95 (`run_meta["map_reference"]`, YOLO configs);
MiniGPT prefill/decode energy per token and training step energy (descriptive).
Functions return DataFrames; `write_eda_config(...)` renders markdown tables and figures.

**Tests:**
- [ ] ICC on synthetic data with known between/within variance; scaling fit recovers parameters on
  noiseless data.
- [ ] `write_eda_config` on a small synthetic catalogue writes a markdown file containing every section
  heading listed above.
- [ ] Commit.

### Task 13: exploration report, trace level (`eda.py`, part 2)

**Files:** Modify `analytics/eda.py`, `analytics/tests/test_eda.py`.

**Contents** (spec §5 items 6-8, appended to `eda.md` by `write_eda_trace(...)`): within-cell power vs
temperature slope in forward-phase windows per configuration (OLS slope W/°C with its standard error,
per family, not pooled); soak trajectories (power and temperature vs time, figure; time to reach 95 % of
the final 5-minute mean power); cell-order residual check (residual of energy per image after
configuration means vs start order within each collection day, Spearman); counter correlations with
window power: Spearman and Pearson, raw and within-cell demeaned, per phase (forward, idle, serve)
and pooled; PMIC: `pmic_W` vs `P_W` scatter and linear fit per family over `pmic_windows` (v1 and v2);
the per-cell autocorrelation time of window power (lag where ACF < 0.1) for 5 named cells.

**Tests:**
- [ ] demeaned correlation removes a pure between-cell effect on synthetic data.
- [ ] time-to-95 % on a synthetic exponential trajectory is within 1 % of the analytic value.
- [ ] Commit.

### Task 14: Mode L ladder and probe mode (`mode_l.py`)

**Files:** Create `analytics/mode_l.py`, `analytics/tests/test_mode_l.py`; add CLI `mode-l [--seed N]`
writing `derived/mode_l/<run_id>.parquet` and runlog lines.

**Interfaces — Consumes:** windows, pmic windows, catalogue, `splits.YOLO`, Mode A predictions.
**Produces:**
- Window models with `fit(train_w, params, sample_weight)` / `predict(w) -> np.ndarray` (W):
  `L0Lookup` (mean P by `cpu_key, fan_mode, busy > 0.5`, falling back to `cpu_key` then global),
  `L1Pmic` (OLS `P = a + b * pmic_W`), `L2Busy` (Ridge on `busy, busy * freq_mhz, cycles_rate`),
  `L3Counters` (Ridge and HGB variants on all counter features + lags + `freq_mhz, temp_C, fan_state`
  + one-hot `fan_mode, governor`; missing values median-imputed inside the training fold for Ridge;
  grids as Task 8 M1/M2), `L4CountersPmic` (L3 + `pmic_W, pmic_age_s`).
  `sample_weight = 1 / n_windows_of_its_cell` so every cell counts equally.
- `mode_l.run_lomo(w, cat, seed) -> pd.DataFrame`: outer folds = leave one YOLO base model out
  (all its cells); inner leave-one-model-out on the training models chooses each rung's grid point by
  the median whole-cell energy APE. Output one row per (fold, test cell, rung):
  `E_true_J, E_pred_J` (whole cell), `E_true_meas_J, E_pred_meas_J` (windows whose phase is the
  cell's measured mode), `ape_1s, ape_10s` (median over consecutive non-overlapping blocks of 10 and
  100 windows), `mae_window_W, bias_pct`, plus per-phase energy APE columns `ape_phase_<p>` for
  `idle, load_weights, warmup, forward, serve`.
- `mode_l.run_pmic_v1(pw, cat) -> pd.DataFrame`: L1 scored leave-one-workload-out (all 14 v1 workloads
  + v2 models) on `pmic_windows`, whole-cell energy APE per cell.
- `mode_l.probe(w, der, cat, mode_a_pred, seed) -> pd.DataFrame`: for each YOLO configuration with
  >= 2 primary cells and each ordered pair (j, k), j != k: probe energy per image = mean predicted
  power over cell j's measured-phase windows (outer-fold model of the fold holding that model) x
  `latency_ms_per_image` of cell j; truth = cell k's `energy_mJ_per_image`. Columns `config_key, j, k,
  rung, probe_mJ, true_mJ, ape, mode_a_mJ, mode_a_ape` (Mode A = the S-LOMO `AUTO` prediction).

**Tests:**
- [ ] synthetic windows where P = 4 + 5 x busy: L2 recovers it; L0 lookup falls back correctly.
- [ ] spy: no outer test cell's windows reach any fit; sample weights sum to the number of training
  cells.
- [ ] probe never uses the same cell for j and k.
- [ ] Commit.

### Task 15: results report and model cards (`report.py`)

**Files:** Create `analytics/report.py`, `analytics/tests/test_report.py`; add CLI `report` writing
`analytics/reports/results.md`, `analytics/reports/model_card_mode_a.md`,
`analytics/reports/model_card_mode_l.md`, figures.

**Contents:** for each split and target: baseline-ordered table (B0, B1, B2, B3, M1, M2, M3, AUTO) of
MdAPE, MAPE, MAE, bias with cluster-bootstrap 95 % intervals (over base models; over configs for
RANDOM), 90 % coverage, n, the noise floor; per-similarity-class and per-model subgroup tables;
paired Wilcoxon of each learned candidate vs the best baseline with Holm-adjusted p and median
paired difference (marked "indistinguishable" when below the noise floor); error vs novelty figure;
worst-10 configurations by AUTO APE under S-LOMO joined to their QC row (the controller writes the
autopsy text into `analytics/reports/autopsy.md`); learning curve figure; v2-only vs pooled comparison
(re-run S-LOMO YOLO folds with `rig == "nov2026"` rows only); seed spread of the headline over the 3
seeds in `runs.jsonl`; deployment validation table; Mode L ladder table (whole cell, measured segment,
1 s, 10 s, per phase, per family), L1-on-v1 table, probe vs Mode A table, counter overhead note.
Every table states dataset version(s), split, target and n; every number cites a `run_id`.
Model cards follow spec §11 (target and mode, intended and out-of-scope use, data and split,
features, headline metrics with intervals, coverage, limitations, worst subgroups, hashes).

**Tests:**
- [ ] on synthetic prediction frames, every required table heading is present and each table carries
  `n` and a `run_id`.
- [ ] Commit.

### Task 16 (controller): run, write up, freeze

Not dispatched as one implementer task. The controller runs, in order: `catalogue`, `derive`, `qc`
(then amends `PROTOCOL.md` with the final QC bounds), `tables`, `splits`, `eda`, `windows`, then the
model runs (`mode-a` all splits and targets for seeds 20261122, 20261123, 20261124; `deploy-validate`;
`mode-l` for the same seeds), `report`, writes `autopsy.md`, fits the final models (`fit_final` for the
AUTO choice per target and the best Mode L rung), commits everything, and tags `analytics-frozen`.
Fixes found here go through an implementer and a reviewer like any task.

### Task 17 (controller): lockbox collection and single scoring

Spec §9. After the `analytics-frozen` tag:
1. Download `yolo12n.pt` and `yolo11n-seg.pt` on the NUC (tools env), copy them to the Pi project root
   (`scp`, no Pi Wi-Fi needed). Write static features with `tools/static_features.py`, record Mode A
   predictions for every lockbox configuration in `analytics/reports/lockbox_predictions.csv`, and
   commit that file before any lockbox cell is measured.
2. Dry run each new model on the Pi without the Monsoon. If yolo11n-seg needs more than a small,
   reviewed patch, replace it with yolov8n at 480 px.
3. Run the lockbox cells with the existing orchestrator on the EliteBook (usual lab rules: Pi Wi-Fi
   down, recorder healthy, recorder restart if older than 7 days, power-cycle procedure unchanged).
   Package with `host/package_dataset.py` to the EliteBook, copy to `~/pi-power-data/lockbox/`.
4. Score once with the frozen models (Mode A, deployment calculator, Mode L, probe), write
   `analytics/reports/lockbox.md`, update the model cards. Any later fix is labelled exploratory.
