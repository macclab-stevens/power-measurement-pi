# Dataset v2 harness — implementation plan (Phase 1 of the expansion plan)

**Spec:** `~/Downloads/pi-power-dataset-expansion-plan.pdf` ("Dataset Expansion Plan", 13 pages).
This plan implements its Section 8 (harness changes) so Phases 0-3 can run. Section numbers below
("§5", "§8") refer to that PDF.

**Branch:** `dataset-v2-harness` (from `origin/main` bc85c3e, which contains all v1 work).

## Context every task needs

- Three machines. **NUC** (this checkout, `~/Desktop/power-measurement-pi`): development + the new
  `tools/` (quantize, prune, mAP). **Pi 5** (`maccpi@192.168.0.243`, project at
  `~/Desktop/power-measurement-pi`): runs `src/run.py`. **EliteBook** (`macc2026@192.168.0.11`,
  project at `~/power-measurement-pi`): Monsoon recorder on port 8051 plus the orchestrator
  (`host/experiment.py`, `host/lab.py`, `host/align.py`, `host/verify_cell.py`).
- `src/run.py` runs one measurement ("cell") on the Pi. In `--sampler monsoon` mode it logs events
  (`src/eventlog.py`, `t_real_ns = time.time_ns()`, the lab PTP-disciplined wall clock) and emits
  sync pulses. `host/align.py` aligns events to the Monsoon trace and writes
  `power_trace.parquet` with a `t_pi_ns` column (the Pi wall clock in ns) per 5 kHz sample.
  `host/verify_cell.py` checks a cell; `host/experiment.py` runs a manifest of cells;
  `host/matrix.py` builds v1 manifests; `host/lab.py` (`./lab`) runs single presets.
- Root helper `tools/pi-power-pin` is installed root-owned at `/usr/local/sbin/pi-power-pin` and is
  the ONLY thing `maccpi` may run with passwordless sudo. `src/pinning.py` wraps it.
- Pi facts (probed 2026-11-19): kernel 6.18, Debian 13, torch 2.14.0, ultralytics 8.4.152,
  `torch.backends.quantized.supported_engines` = qnnpack, onednn, x86, fbgemm. cpufreq: a single
  `policy0` covering cpus 0-3, per-core `/sys/devices/system/cpu/cpu{0..3}/cpufreq/scaling_cur_freq`
  (kHz). Governors available: conservative ondemand userspace powersave performance schedutil.
  `/sys/class/thermal/thermal_zone0/temp` (milli-C, type cpu-thermal), trip 0 critical 110 C,
  trips 1-4 active (fan) 50/60/67.5/75 C, fan = `cooling_device0` (pwm-fan). `perf` 6.18.60 is
  installed; `perf_event_paranoid=2` so counting is user-space only (events print as `cycles:u`).
  The A76 PMU multiplexes: with 11 hardware events, `cycles` was counted 42% of the time and
  `L1-dcache-load-misses`, `LLC-loads`, `LLC-load-misses`, `bus-cycles` printed `<not counted>` in a
  0.2 s run. Event names that exist: `cycles` (alias `cpu_cycles`), `instructions`,
  `cache-references`, `cache-misses`, `branch-misses`, `stalled-cycles-backend`,
  `L1-dcache-loads`, `L1-dcache-load-misses`, `LLC-loads`, `LLC-load-misses`, `bus-cycles`,
  `task-clock`, `context-switches`, `page-faults`.
- Weights on the Pi project root: yolo11n/s/m.pt, yolov8n/s/m.pt (`*.pt` is gitignored).

## Global Constraints

1. **v1 must not change.** Every v1 command line (the `host/matrix.py` `MODELS` presets with the
   v1 flags) must produce the same behaviour, outputs and checks as today. New flags default to v1
   behaviour. `schema_version` stays 4 for v1-shaped runs; runs that use any v2 feature write
   `"dataset_version": 2` in `run_meta.json` (v1 runs write `"dataset_version": 1`).
2. **Quality gates are not waived.** Alignment pass stays `max_residual_ms < 1.0` (`PASS_MS`).
   New modes get adapted checks, never skipped checks.
3. **Thermal limits:** abort at **85 C** (`THERMAL_ABORT_C = 85.0`), hard stop at **90 C**
   (`THERMAL_HARD_STOP_C = 90.0`), enforced inside `run.py` on the Pi from the 1 Hz sysfs sampler.
   Exit codes: thermal abort = **3**, hard stop = **4**.
4. **Counters:** `perf stat -I <ms> -x,` attached to the run.py process, default interval
   **100 ms** (10 Hz), option to use 200 ms (5 Hz). sysfs at **1 Hz**. All timestamps are
   `time.time_ns()` on the Pi, so they join to `power_trace.parquet` on `t_pi_ns` with
   `pandas.merge_asof`. Raw `counters.csv` and `sysfs.csv` stay in the cell folder.
   Counter coverage threshold for a v2 cell: **≥ 95 %** of the measured-phase samples.
5. **Holdout tags** (written into `run_meta.json["holdout"]`): `"primary-s"` for yolo11s/yolov8s,
   `"secondary-m"` for yolo11m/yolov8m, `null` otherwise.
6. **No passwords** in code, docs, tests or commits. Never run `sudo` (other than through the
   existing whitelisted helper) and never switch power. Never touch the Monsoon recorder process.
7. **Never run a measurement.** Tasks are code + unit tests only. Do not run `src/run.py` with
   `--sampler monsoon`, do not run `host/experiment.py run`, `./lab run` or `./lab test`.
8. **Testing.** The NUC has no project venv. Run tests on scratch copies with
   `/home/macc-2026/.claude/jobs/45f7226a/tmp/remote-test.sh pi|host|both [pytest args]`.
   It rsyncs the NUC working tree (uncommitted changes included) to `~/sdd-v2/power-measurement-pi`
   on the Pi and/or EliteBook and runs pytest there (Pi: `tests/`; host: `host/tests`). Pytest
   args are paths relative to `tests/`' parent for `pi` (e.g. `tests/test_pinning.py`) and relative
   to `host/` for `host` (e.g. `tests/test_align.py`). Baseline before this plan: Pi 50 passed
   (1 deselected), host 145 passed. Both suites must stay green. Never edit the deployed
   checkouts on the Pi/EliteBook.
9. **NUC tools** (`tools/*.py`) run in their own uv project `tools/pyproject.toml` (Python 3.12,
   CPU torch **2.14.0**, `ultralytics==8.4.152`, plus what the tool needs), with uv at
   `~/.local/bin/uv`. Generated artifacts (weights, datasets, images) are never committed;
   small JSON results that the Pi needs (e.g. `data/map_reference.json`, `inputs/MANIFEST.json`)
   are committed.
10. Match the existing code style: module docstring explaining purpose and usage, small pure
    functions that tests can drive, subprocess/sysfs access behind injectable parameters (see how
    `src/pinning.py` takes `run=subprocess.run` and `host/experiment.py` takes a `Shell`).
11. Commit each task on the branch with a conventional message (`feat(run): ...`). End each commit
    message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Do not push.

## Cell naming (used by Tasks 11-12)

`{model}[_{variant}][_{condition}]_{imgsz}_t{threads}[_b{batch}]_{clock}_r{repeat:02d}` where
`clock` is the pinned MHz (`2400`) or `dvfs`, `variant` ∈ `int8d int8s fp16 p30 p50`,
`condition` ∈ `fanauto fanoff soak30 serve1 serve5 serve15 stream empty crowded conc2 concspin`.
Examples: `yolo11n_int8s_640_t4_2400_r00`, `yolo11n_640_t4_dvfs_r00` (DVFS: the clock field
says `dvfs`, no separate condition token), `yolo11n_fanoff_640_t4_2400_r01`,
`yolo11m_soak30_640_t4_2400_r00`, `yolo11s_480_t4_2400_r02`, `yolo11n_640_t4_b8_2400_r00`.

---

### Task 1: pi-power-pin helper — unpin, fan auto, fan off

**Files:** `tools/pi-power-pin`, `src/pinning.py`, `tests/test_pinning.py`.

Add three helper commands, keeping the existing four unchanged:
- `pi-power-pin unpin` — governor `ondemand`, min 1500000 kHz, max 2400000 kHz, **fan trips
  untouched** (unlike `restore`, which also resets the fan).
- `pi-power-pin fan auto` — stock fan trips `50000 60000 67500 75000` (what `restore` sets).
- `pi-power-pin fan off` — fan trips `85000 86000 87000 88000` (fan stays off below 85 C; the fan
  comes back as a backstop at 85 C, the same point as the run.py abort). Comment why.
- Usage line and header comment updated. Still prints the status line after every command.

`src/pinning.py`:
- `apply(cpu_mhz, fan_full, run=...)` keeps its signature and behaviour. Add
  `apply_mode(cpu_mhz: int | None, fan: str, governor: str, run=subprocess.run) -> str` where
  `fan` ∈ `("full", "auto", "off", "none")` (`"none"` = do not touch the fan) and `governor` ∈
  `("pinned", "ondemand")`. `pinned` + cpu_mhz → `pin <mhz>`; `ondemand` → `unpin`. Fan command
  is issued before the clock command (as `apply` does). Validates values (ValueError).
  `apply(cpu_mhz, fan_full)` must be equivalent to
  `apply_mode(cpu_mhz, "full" if fan_full else "none", "pinned")`.
- Add `pin(mhz, run=...)` and `unpin(run=...)` thin wrappers (used by Task 4 around sync pulses).
- Add `clock_range_errors(sysfs_rows, lo_mhz=1500, hi_mhz=2400, tol_mhz=25) -> list[str]`: one
  error per sysfs row whose any `cpuN_kHz` value is outside `[lo-tol, hi+tol]` MHz, plus
  `"no cpu frequency readings"` if no row has a value. (Rows are dicts as produced by Task 2:
  keys `t_real_ns`, `cpu0_kHz`..`cpu3_kHz`, `temp_C`, `fan_state`; values may be None.)

Tests: fake `run` capturing argv for every new command path; the equivalence of `apply` and
`apply_mode`; validation errors; `clock_range_errors` cases. Also a test that parses
`tools/pi-power-pin` text and asserts the `fan off` trips are `85000 86000 87000 88000` and that
`unpin` does not call `set_trips` (grep the case branch).

Note for the report: the installed helper on the Pi must be re-installed by the user with sudo
(README §4.4 command) before Phase 0; do not attempt it.

### Task 2: 1 Hz sysfs sampler with thermal watchdog

**Files:** new `src/sysfs_sampler.py`, `src/run.py`, new `tests/test_sysfs_sampler.py`.

`SysfsSampler(period_s=1.0, root="/", abort_c=85.0, hard_stop_c=90.0, clock=time.time_ns,
interrupt=_thread.interrupt_main)`:
- Thread reads, each period: `cpu{0..3}/cpufreq/scaling_cur_freq` (kHz, int) under
  `{root}/sys/devices/system/cpu/`, `{root}/sys/class/thermal/thermal_zone0/temp` (→ `temp_C`
  float, milli-C / 1000), `{root}/sys/class/thermal/cooling_device0/cur_state` (`fan_state` int).
  Unreadable value → None for that field (never raises in the thread). Row keys exactly
  `t_real_ns, cpu0_kHz, cpu1_kHz, cpu2_kHz, cpu3_kHz, temp_C, fan_state`.
- `read_once()` public (tests call it without threads). `start()`, `stop()`, `rows()`,
  `write_csv(path)` (header = the row keys, in that order), `max_temp_C()`.
- Thermal: when a reading has `temp_C >= abort_c` the first time, set `self.thermal_abort =
  {"t_real_ns", "temp_C", "level": "abort"}`. When `temp_C >= hard_stop_c`, set level
  `"hard_stop"` and call `interrupt()` once (raises KeyboardInterrupt in the main thread).
- No subprocesses (that is the point: §7 says sysfs reads are cheaper than vcgencmd).

`src/run.py` integration (monsoon AND pmic mode; v1 runs get it too — it is passive and adds
`sysfs.csv`):
- Start the sampler right after `sampler.start()`, stop it in the `finally`. Write
  `sysfs.csv` next to `context.csv` (also on failure paths that reach the finally: write it in
  the finally if the outdir exists).
- Add a check function `_check_thermal()` called at every frame boundary in the measured-mode
  loop, every warmup pass, every serve request, and (Task 4) every soak iteration; if
  `thermal_abort` is set it raises `ThermalAbort`. `main()` catches `ThermalAbort` → writes
  `thermal_abort.json` (the dict + `"phase"` it happened in) to outdir, logs an error, and exits
  with code **3**. A `KeyboardInterrupt` while `thermal_abort["level"] == "hard_stop"` → write
  the same file and exit **4**. Any other KeyboardInterrupt behaves as today.
- `run_meta.json` gains `"sysfs_period_s"`, `"max_temp_C"` (from the sampler) and
  `"thermal_abort": null` on success.
- The existing 5 s vcgencmd context thread is unchanged (it carries throttle bits and the PMIC
  dump that v1 acceptance uses).

Tests: a temp-dir fake sysfs tree (write the files, then `read_once()`); missing files → None;
abort and hard-stop transitions with a fake `interrupt`; `write_csv` header/order; a test of the
exit-code mapping by calling a small pure helper (factor one out, e.g.
`thermal_exit_code(thermal_abort) -> int | None`).

### Task 3: perf stat hardware counters

**Files:** new `src/perfcounters.py`, `src/run.py`, new `tests/test_perfcounters.py`,
fixture `tests/fixtures/perf_interval_sample.txt`.

`DEFAULT_EVENTS = ("cycles", "instructions", "cache-references", "cache-misses",
"branch-misses", "stalled-cycles-backend", "LLC-load-misses", "task-clock",
"context-switches", "page-faults")` — 7 hardware events fit the A76 PMU (cycle counter + 6
programmable) without multiplexing (measured 100 % running on the Pi). Not in the default set:
`L1-dcache-loads` / `L1-dcache-load-misses` (identical counts to `cache-references` /
`cache-misses` on the A76), `LLC-loads`, `bus-cycles` (an 8th hardware event multiplexes all of
them to 71-90 %). (Ruling R13; changed after the Task 3 first pass.)

- `PerfStat(pid, events=DEFAULT_EVENTS, interval_ms=100, out_path=..., popen=subprocess.Popen,
  clock=time.time_ns, which=shutil.which)`. `available()` → False if `perf` is not on PATH.
  `start()` records `t0_ns = clock()` immediately before spawning
  `perf stat -I <interval_ms> -x , -o <raw_path> -e <comma-joined events> -p <pid>`
  (`raw_path` = `counters.raw.txt` in the cell folder). `stop()` sends SIGINT, waits ≤ 5 s,
  then kills.
- `parse_interval_csv(text, t0_ns) -> list[dict]`: perf `-I -x,` lines are
  `time,count,unit,event,run_time_ns,pct_running[,metric,metric_unit]`. Skip comments/blank
  lines. Strip a `:u`/`:k` modifier from the event name. `count` may be `<not counted>` or
  `<not supported>` → None. Output one row per (interval, event):
  `t_end_ns = t0_ns + round(time_s * 1e9)`, `interval_s`, `event`, `count`, `pct_running`
  (float 0-100, None if missing).
- `to_wide(rows) -> list[dict]`: one row per interval with `t_end_ns`, `t_start_ns`
  (previous interval's end, or `t0_ns` for the first), and per event two columns
  `ctr_<event>` (count, None if not counted) and `pct_<event>`; event names normalized with
  `-`→`_`. Write as `counters.csv` (columns sorted: `t_start_ns, t_end_ns`, then `ctr_*` in
  DEFAULT_EVENTS order, then `pct_*` in the same order).
- Build the fixture from the real Pi output format (example lines, `-x,` without `-I` shown here;
  with `-I` a leading `time` field is added, e.g. `0.100183117,125585,,cycles:u,212278,42.00,0.253,GHz`
  and `0.100183117,<not counted>,,LLC-loads:u,0,0.00,,`).

`src/run.py`:
- Flags `--perf-counters {on,off}` (default **on**) and `--perf-interval-ms` (default 100,
  choices 100 or 200). In `--sampler pmic` mode counters are forced off.
- Start perf **after warm-up, before `sync_pulse("start")`** (torch's thread pool exists by
  then; `-p` only attaches to threads that exist), stop it after `sync_pulse("end")`. If
  `perf` is unavailable or fails to start, log an error, record it and continue (verify will
  fail the cell on coverage — do not crash a measurement over counters).
- Write `counters.csv` (wide) next to the raw file. `run_meta.json` gains
  `"perf_counters": {"enabled", "available", "interval_ms", "events", "t0_ns",
  "error": str|None, "paranoid": <int read from /proc/sys/kernel/perf_event_paranoid or None>,
  "scope": "user" if paranoid >= 2 else "user+kernel"}`.

Tests: parse fixture (counts, None for not counted, `:u` stripped, timestamps from t0);
`to_wide` columns and ordering; `available()` with fake `which`; command line built by `start()`
with a fake `popen` (assert exact argv); stop sends SIGINT.

### Task 4: run.py workload flags (threads, batch, inputs, serve rate, soak, governor, fan, holdout)

**Files:** `src/run.py`, `src/tasks.py`, new `tests/test_run_v2_flags.py` (and extend
`tests/test_tasks_steps.py` if natural).

- `--threads N` (existing, default 4) now sets all three: `torch.set_num_threads(N)`,
  `cv2.setNumThreads(N)` and `os.environ["OMP_NUM_THREADS"]=str(N)` (set the env var before the
  first torch parallel op; record `"threads": {"torch": torch.get_num_threads(),
  "opencv": cv2.getNumThreads(), "omp_env": ...}` under a new key `"threads_detail"`; keep the
  existing `"threads"` int).
- `--batch` default becomes `None`: train → 8 (unchanged), image → 1. For `--task image`,
  batch B stacks B copies of the preprocessed frame (stream mode: B consecutive stream
  images) into `Bx3xSxS`; postprocess runs on the whole batch. `run_meta["batch"]` reports the
  real batch for image too. `host/lab.py`'s `int(d.get("--batch", 8))` default stays valid.
- `--input-image PATH`: frame fed to the model (default: the `--check-image`, as today). The
  `--check-image` oracle is unchanged and independent. Lets the empty/crowded-scene cells keep
  `bus.jpg` as the oracle.
- `--frames-source {repeat,stream}` (default repeat) and `--stream-dir` (default
  `inputs/stream50`): stream cycles through the sorted `*.jpg` files, decoded once at load
  (decode outside the measured loop so I/O is not measured). Error if fewer than 1 image.
- Record input identity in `run_meta["inputs"]`: `{"source": "repeat"|"stream"|"camera",
  "files": [{"path", "sha256"}]}` (stream: all files).
- `--serve-fps F` (float, default None): when given with `--serve N`, request k is scheduled at
  `t_serve0 + k / F`; the gap is the remaining time (`step "idle_between_requests"` when > 0);
  a late request starts immediately and counts as a missed deadline. Without `--serve-fps`,
  serve keeps today's fixed `--serve-gap-ms` gap. `run_meta`: `"serve_fps"`,
  `"serve_missed_deadlines"`, `"serve_wall_s"`.
- `--soak-minutes M` (float, default None): the measured `forward` mode loops until M minutes
  have elapsed (instead of `--frames`), with `prof.set_frame(f)` as usual, frame 0 still
  profiled, `_check_thermal()` every iteration. `run_meta["soak_minutes"]`,
  `"frames_done"` per mode. Benches must be skipped when soaking (force `--no-bench`, record
  it) to bound the cell.
- `--governor {pinned,ondemand}` (default pinned) and `--fan {full,auto,off}` (default None =
  legacy: `--fan-full` decides). `--fan-full` stays and means `--fan full`; giving both with
  different values is an argparse error. Apply via `pinning.apply_mode`. With `ondemand`,
  `--cpu-mhz` is required and is the clock pinned **only during the sync-pulse trains**:
  pin before each `sync_pulse(...)` call, sleep 0.5 s to settle, emit pulses, then `unpin`
  (the plan's "pin during sync, unpin for the workload"). Log markers stay as today.
  Acceptance: with ondemand, replace `pinning.clock_errors(ctx, cpu_mhz)` by
  `pinning.clock_range_errors(sysfs_rows)` + `pinning.throttle_errors(ctx)`.
  With fan auto/off: skip `wait_fan_full`, skip the "fan not at max" check; in the `finally`,
  if fan was `off`, call `pinning.apply_mode(None, "auto", "pinned")` best-effort so a fan-off
  cell never leaves the fan off. `run_meta`: `"governor"`, `"fan_mode"` (`full|auto|off|none`),
  keep `"fan_full"` (true only for full).
- `--holdout {primary-s,secondary-m,none}` (default none) → `run_meta["holdout"]` (null for
  none). `run_meta["dataset_version"] = 2` if any v2 flag is non-default (threads/batch/imgsz do
  not count by themselves; holdout, governor, fan, serve-fps, soak, frames-source,
  input-image, quant, prune, concurrent, perf-counters do), else 1 — make the rule a small pure
  function with tests. Perf counters default on, so: a run counts as v2 if `--holdout` is given
  or any other v2 flag listed is non-default; plain v1 command lines stay `dataset_version` 1 even
  though counters are collected.
- `run_meta["latency_ms_per_frame"]`: mean/median/p95 of the measured-mode forward iteration
  wall times (time each `task.run_one(mode)` call in the measured loop, frame 0 excluded) and
  `"latency_ms_per_image"` = per-frame / batch. Energy per frame is derived on the host (Task 10).

Tests: pure helpers for serve scheduling (missed deadlines, sleep durations), dataset_version
rule, batch stacking shape (`ImageTask.preprocess` path with a fake frame), stream loading from
a temp dir with 3 tiny jpgs written by cv2, argparse conflicts (`--fan-full --fan off` error,
`--governor ondemand` without `--cpu-mhz` error). Do not run a real model or the monsoon sampler.

### Task 5: precision and compression variants in run.py

**Files:** `src/run.py`, `src/tasks.py`, new `src/yolo_variants.py`, new
`tests/test_variants.py`, new `data/map_reference.json` (initially `{}`).

- `--quant {fp32,int8-dynamic,int8-static,fp16}` (default fp32), `--quant-weights PATH`
  (required for int8-static), `--prune PATH` (a pruned model file from Task 9; mutually
  exclusive with `--quant` other than fp32 — combinations are out of scope).
- `int8-dynamic`: `torch.ao.quantization.quantize_dynamic(module, {nn.Linear}, dtype=qint8)` at
  load, engine `qnnpack`. Count the modules that were actually replaced. **If the count is 0,
  acceptance fails** with "int8-dynamic quantized 0 modules" — YOLO is conv-heavy and dynamic
  quantization only covers Linear-type layers, so it may be a no-op; FP32 data must never be
  labelled INT8 (same rule as the FP16 probe, §4). Record `"quantized_modules"`.
- `int8-static`: `torch.backends.quantized.engine = "qnnpack"`; load the artifact from
  `--quant-weights` with `torch.load(..., weights_only=False)`; the artifact format is a dict
  `{"format": "pi-power-int8s/1", "model": <nn.Module>, "base_model": "yolo11n.pt",
  "fp32_modules": [...], "calibration": {...}}` (Task 8 writes it). The module replaces
  `yolo.model` for forward; keep the ultralytics `YOLO` object for nothing else.
- `fp16`: `module.half()` and the input tensor `.half()`. **Verify the executed dtype at
  runtime**: from the LayerProfiler `layer_rows` (`input_dtypes` of `Conv2d` leaves) assert
  float16 was actually used; record `"executed_dtype"`; acceptance fails if any Conv2d leaf ran
  in float32 ("fp16 requested but conv ran in float32"). Postprocess casts outputs to float32
  before NMS.
- `--prune PATH`: `torch.load(PATH, weights_only=False)`; format `{"format":
  "pi-power-pruned/1", "model": <nn.Module>, "base_model", "ratio", "params_before",
  "params_after"}`. `src/yolo_variants.py` holds any module classes the pruned/quantized
  artifacts need to unpickle (Task 9 will add a split-free `C2f` replacement there; create the
  module now with a docstring and a `register()` no-op hook if nothing is needed yet).
- Oracle for every non-fp32 / pruned variant: run the **executed** module on the check image
  through the same preprocess → forward → NMS path and require max conf ≥ `--conf` (the
  v1 `yolo.predict` oracle would test the FP32 model, not the variant). fp32 keeps the v1
  oracle exactly.
- Variant id: `variant_id(model_path, quant, prune_ratio) -> str`, defined in
  `src/yolo_variants.py` (import-light, so the NUC tools import it too), e.g. `yolo11n`,
  `yolo11n_int8d`, `yolo11n_int8s`, `yolo11n_fp16`, `yolo11n_p30`. `run_meta["variant"]`,
  `run_meta["quant"]`, `run_meta["prune"]` (`{"path", "ratio", "sha256"}` or null),
  `run_meta["quant_weights_sha256"]`.
- mAP reference: `data/map_reference.json` maps `"<variant_id>@<imgsz>"` →
  `{"map50_95", "map50", "n_images", "dataset", "tool_version", "measured"}`. run.py looks up
  its key and writes `run_meta["map_reference"]` (the dict, or null when missing; never an
  error on the Pi — verify warns, Task 10).
- Microbench note: per-op isolation benches deep-copy leaves and synthesize inputs; they are not
  meaningful for quantized/fp16 leaves. If `--quant != fp32`, force `--no-bench` and record
  `"bench_forced_off": "quant"`.

Tests: `variant_id` table; quant-count-0 acceptance message via a pure function on a tiny
`nn.Sequential(nn.Conv2d(...))`; int8-dynamic on a tiny `nn.Sequential(nn.Linear)` really
replaces a module (count 1); fp16 dtype check function on synthetic `layer_rows`;
map_reference lookup (hit/miss); artifact format validation errors (wrong `format` string).

### Task 6: NUC tools environment and COCO data subsets

**Files:** new `tools/pyproject.toml` (+ `tools/uv.lock`), new `tools/coco_subset.py`, new
`tools/README.md`, `.gitignore`, new `inputs/MANIFEST.json`, `tools/tests/test_coco_subset.py`.

- uv project in `tools/`: python `>=3.12,<3.13`, `torch==2.14.0` from the PyTorch CPU index
  (`[tool.uv.sources]` + `[[tool.uv.index]] url = "https://download.pytorch.org/whl/cpu"`,
  explicit), `torchvision` matching, `ultralytics==8.4.152`, `opencv-python-headless`,
  `numpy`, `torch-pruning`, `pytest` (dev). Run with `cd tools && ~/.local/bin/uv run ...`.
  Tests here run on the NUC: `cd tools && ~/.local/bin/uv run pytest tests`.
- `coco_subset.py` downloads (into `tools/data/`, gitignored) COCO val2017 images
  (`http://images.cocodataset.org/zips/val2017.zip`) and the YOLO-format labels
  (`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip`; use only
  `labels/val2017`). Resumable (skip if present, verify zip integrity).
  Deterministic subsets from the image ids sorted ascending:
  - **eval**: the first **500** images (by sorted file name) → `tools/data/coco500/images` +
    `labels` (+ an ultralytics dataset YAML `tools/data/coco500.yaml` with the 80 COCO names).
  - **calib**: the next **200** images (disjoint from eval) → `tools/data/calib200/images`.
  - **stream**: 50 images from the remaining ones (sorted, every k-th so they span the set) →
    `inputs/stream50/*.jpg`.
  - **empty**: the first remaining image with zero label lines → `inputs/empty.jpg`.
  - **crowded**: the remaining image with the most label lines → `inputs/crowded.jpg`.
  - Writes `inputs/MANIFEST.json`: per file `{coco_file, sha256, n_labels}` and the rule used.
- `.gitignore`: `tools/data/`, `inputs/*` except `inputs/MANIFEST.json`, `models/`.
- `tools/README.md`: one page — what each tool does, commands, where outputs go, and how to copy
  `inputs/` and `models/` to the Pi (`rsync -a inputs models maccpi@192.168.0.243:Desktop/power-measurement-pi/`).
- Actually run `coco_subset.py` on the NUC (internet is available there) and commit the
  manifest. Tests cover the subset selection logic on a synthetic label directory (pure
  function taking a list of (file, n_labels)).

### Task 7: tools/eval_map.py — one-time mAP per model/variant

**Files:** new `tools/eval_map.py`, `tools/tests/test_eval_map.py`, `data/map_reference.json`.

- Evaluates a model or variant artifact on `tools/data/coco500` at a given imgsz and writes
  /merges `data/map_reference.json` entries keyed `"<variant_id>@<imgsz>"` (Task 5 format and
  `variant_id` — import it from `src/yolo_variants.py`; do not copy logic). Fields: `map50_95`, `map50`, `n_images`, `dataset: "coco-val2017-first500"`,
  `tool_version` (git short hash), `measured` (ISO date).
- FP32 ultralytics weights: may use `YOLO(path).val(data=coco500.yaml, imgsz=..., batch=1,
  device="cpu", conf=0.001, iou=0.7, plots=False)`. Variant artifacts (Task 8/9 dict formats,
  fp16): run your own loop with the **same** letterbox preprocess as `src/tasks.py`
  `ImageTask.preprocess`, NMS `conf_thres=0.001, iou_thres=0.7`, and ultralytics'
  `ultralytics.utils.metrics` (`ap_per_class` / `box_iou`, the standard COCO IoU thresholds
  0.50:0.05:0.95) so all variants use one metric. Include an FP32 sanity mode that evaluates an
  FP32 model through the custom loop; within ±0.01 mAP50-95 of `.val()` is the acceptance check
  you report (state the measured difference).
- CLI: `uv run python eval_map.py --model ../yolo11n.pt [--quant fp16|int8-dynamic]
  [--artifact ../models/yolo11n_int8s.pt] --imgsz 640` and `--all` that runs the plan's set
  (FP32 n/s/m of both generations at 640; yolo11n/s/m FP32 at 320, 480, 960; fp16 yolo11n@640;
  int8-dynamic yolo11n/yolo11s/yolov8n@640; and any `models/*.pt` artifacts present @640).
  Weights: download ultralytics weights into the repo root if missing (`*.pt` gitignored).
- Run `--all` for what exists now (FP32 + fp16 + int8-dynamic) and commit
  `data/map_reference.json`. Report the table in the report file. int8-dynamic: report the
  quantized-module count (expected 0 for YOLO → mAP equals FP32; that is a finding, not a bug).

Tests: the metric wrapper on a tiny synthetic case (one image, one perfect box → mAP 1.0; one
wrong class → 0); key format; merge preserves other keys.

### Task 8: tools/quantize_yolo.py — INT8 static PTQ

**Files:** new `tools/quantize_yolo.py`, `tools/tests/test_quantize_yolo.py`.

- Static post-training quantization of yolo11n and yolo11s for the **qnnpack** engine (the Pi
  runs it; set `torch.backends.quantized.engine = "qnnpack"` on the NUC too), calibrated on
  `tools/data/calib200` with the same letterbox preprocess at 640.
- Use whatever PTQ path works in torch 2.14 (check what exists first: FX graph mode
  `torch.ao.quantization.quantize_fx.prepare_fx/convert_fx`, or PT2E). The YOLO modules C2f /
  C3k2 / SPPF / Detect / C2PSA may resist tracing or lose accuracy: quantize what traces cleanly
  and keep the rest FP32 (e.g. the Detect head, DFL, attention), recording **which modules stayed
  FP32 and why** in the artifact's `fp32_modules`. A partially quantized model with a known mAP is
  the goal; a silently broken one is not.
- Output `models/<stem>_int8s.pt` in the Task 5 dict format (`"format": "pi-power-int8s/1"`,
  `model`, `base_model`, `fp32_modules`, `calibration: {"n_images": 200, "subset":
  "calib200", "engine": "qnnpack", "method": ...}`, `torch_version`).
- Verify after conversion: the artifact reloads with `torch.load(weights_only=False)` in a fresh
  process, runs a forward at 640 with qnnpack, output shape matches FP32, and count quantized
  modules > 0. Then run `eval_map.py` on it and commit the updated `data/map_reference.json`.
- Report: method used, quantized vs FP32 module counts, mAP vs FP32, artifact size, and any
  operator that failed under qnnpack.

Tests: small pure pieces (artifact dict validation, module-count helper on a toy quantized
model). The full PTQ run is evidence in the report, not a unit test.

### Task 9: tools/prune_yolo.py — structured channel pruning

**Files:** new `tools/prune_yolo.py`, `src/yolo_variants.py`, `tools/tests/test_prune_yolo.py`.

- Prune yolo11n at ratios **0.30 and 0.50** with `torch-pruning` (magnitude importance, L1 or
  L2 — state which), global channel ratio, Detect head output convs ignored (`ignored_layers`),
  **no fine-tuning** (§4).
- Ultralytics C2f/C3k2 split the tensor with `chunk`/`split` on a fixed `self.c`, which breaks
  channel pruning. Replace those modules with a split-free equivalent before pruning (load the
  same weights so outputs are identical before pruning — test this). Put the replacement
  class(es) in `src/yolo_variants.py` so the Pi can unpickle the artifact (tools import from
  `../src`). Keep the class import-light (torch + ultralytics only).
- Output `models/yolo11n_p30.pt`, `models/yolo11n_p50.pt` in the Task 5 dict format
  (`"format": "pi-power-pruned/1"`, `model` (FP32, eval), `base_model`, `ratio`,
  `params_before`, `params_after`, `macs_before`, `macs_after` at 640).
- Verify: reload in a fresh process with only `src/` on the path, forward at 640, output shape
  equal to FP32; run `eval_map.py` on both and commit `data/map_reference.json`.

Tests: split-free replacement equals the original C2f/C3k2 output on random input (tolerance
1e-5); artifact validation.

### Task 10: host side — merge counters/sysfs, v2 checks, derived metrics

**Files:** `host/align.py`, `host/verify_cell.py`, `host/tests/test_align.py`,
`host/tests/test_verify_cell.py`.

`align.py`:
- After writing the labelled frame, if `counters.csv` exists in the Pi run dir, `merge_asof` it
  onto the samples by `t_pi_ns`: each sample takes the counter interval that **contains** it
  (`t_start_ns < t_pi_ns <= t_end_ns`; use the interval end as the key with
  `direction="forward"` and then null out samples with `t_pi_ns <= t_start_ns`), columns
  `ctr_*` (as float64 counts per interval) and `pct_*`. If `sysfs.csv` exists, merge with
  `direction="backward"`, `tolerance` 2 s, columns prefixed `sys_` (`sys_cpu0_kHz` …
  `sys_temp_C`, `sys_fan_state`). Missing files → no columns (v1 runs unchanged).
- Keep the alignment thresholds unchanged (Global Constraint 2). The DVFS design pins during sync
  pulses, so no edge-tolerance change is needed; add a comment saying so.

`verify_cell.py`:
- `fan_full` check: required only when `meta.get("fan_mode", "full" if meta.get("fan_full") else
  "none")` is `full`; for `auto`/`off` check the mode is recorded instead.
- `verify(run_dir, cpu_mhz=None, ...)` with `meta["governor"] == "ondemand"`: keep the
  `cpu_mhz_pinned` equality (it is the sync clock) and add a frequency-range check over
  `sys_cpu*_kHz` in the mode phases (within 1500-2400 MHz ± 25) — error if no readings.
- Counter coverage (only when `meta.get("dataset_version") == 2`): fraction of mode-phase
  samples whose `ctr_instructions` is non-null must be **≥ 0.95**, else error
  `"counter coverage X < 0.95"`; if `perf_counters.enabled` is false for a v2 cell → error.
  Report `counter_coverage` in the result. v1 cells: no counter check.
- Temperature: `thermal_abort.json` present or `meta["thermal_abort"]` non-null → error
  ("thermal abort at T C"); `max_temp_C >= 85` → error; otherwise report `max_temp_C`.
- mAP: for YOLO models, `meta["map_reference"]` null → **warning** "mAP reference missing for
  <variant>@<imgsz>".
- Expected phases: soak cells and serve-fps cells keep the same phase names (soak only lengthens
  `forward`), but `bench:*` is not required when `bench_skipped` or `bench_forced_off`.
- Derived metrics in the result and in a new `derived.json` in the cell folder:
  `energy_per_frame_mJ` = mean over measured-mode iterations (frame ≥ 1) of the per-iteration
  energy from `phase_energy` (sum of E_mJ for that mode's iters) and the same minus baseline
  (`dyn_energy_per_frame_mJ` = E − baseline_P × duration), `latency_ms_per_frame` (from the
  trace: mean iteration duration), `energy_per_image_mJ` (÷ batch), `map_reference` copied
  through, `variant`, `holdout`.

Tests: synthetic dataframes / temp dirs for each new check (existing tests show the pattern);
merge_asof edge cases (sample exactly at an interval end belongs to that interval; sample before
the first interval → NaN); v1 meta fixture still verifies exactly as before.

### Task 11: experiment.py — timeouts, thermal stage, disk and recorder pre-checks

**Files:** `host/experiment.py`, `host/tests/test_experiment.py`.

- Per-cell timeout: `cell.get("timeout_s")` overrides `RUN_TIMEOUT_S` (v1 cells have none).
- run.py exit code 3/4 → result `stage "thermal_abort"` with the Pi's `thermal_abort.json`
  content (it is rsynced with the run dir) in `error`. Not retried (`attempts` still counted),
  and the matrix continues with the next cell after the normal start-of-cell thermal gate.
- Thermal gate: unchanged (start of cell only; 50 C) — soaks are never gated mid-run.
- Disk pre-check: before `run()` starts, require free space on the EliteBook runs dir ≥
  `MIN_FREE_GB = 10` and on the Pi project dir ≥ 5 GB (`df -Pk` over ssh); before every cell
  whose `family` is `"soak"`, require ≥ 2 GB more than that. Failure → stage `"disk"`, which
  aborts the matrix like `recorder`.
- **Recorder stall check** (found 2026-11-19: the recorder silently stops at n_total =
  4294967295 while `/status` still says recording): before each cell, read `/status` twice
  2 s apart; if `n_total` did not increase, or `n_total >= 4294967295 - 5000 * 3600` (less than
  an hour of samples left before the 32-bit ceiling), fail stage `"recorder"` with a message that
  says to restart the recorder per README §8. Make it a pure function
  `recorder_stalled(st_a, st_b) -> str | None` plus the wiring; the `sleep` is injectable.

Tests: each new path with the existing fake Shell/status patterns.

### Task 12: v2 matrix families

**Files:** new `host/matrix_v2.py`, `host/experiment.py` (`plan` subcommand),
new `host/tests/test_matrix_v2.py`.

Build manifest rows with the cell-naming scheme above. Every row: `cell_id`, `family`
(`A`, `A-input`, `B`, `C`, `D`, `soak`), `model_key` (e.g. `yolo11s`), `cpu_mhz` (int — also
for DVFS cells: the sync-pin clock 2400), `repeat`, `round`, `holdout`, `timeout_s` (only when
not default), `args` (run.py args, same style as `host/matrix.py` `_cell`).

Common args for every v2 cell: `--task image --model <m>.pt --check-image bus.jpg --sampler
monsoon --perf-counters on --holdout <tag> --threads <t> --imgsz <s> --cpu-mhz <mhz>` plus
`--fan full` unless the cell sets another fan mode. Frames: n/s models 30, m models 20; for batch
B use `max(5, ceil(frames / B))`; warmup 10; `--serve 10 --serve-gap-ms 200` as in v1 inference
cells unless the cell is a serve-rate cell. Benches: `--bench 2` for families A and C; `--no-bench`
for B, D and soak.

- **A (44)**: yolov8s, yolov8m, yolo11s, yolo11m at 640/t4/b1 fp32: 2400 MHz × 5 repeats, 2000 ×
  3, 1500 × 3.
- **A-input (6)**: yolo11n @2400 × 3 with `--input-image inputs/empty.jpg` (condition `empty`)
  and × 3 with `inputs/crowded.jpg` (`crowded`).
- **B** — after Rulings R17 (int8-dynamic quantizes 0 modules on YOLO) and R23 (pruned yolo11n
  detects nothing without fine-tuning, even after BN recalibration: mAP 0.0 at 30 %/50 %), the
  default B is int8-static only: **12 cells** (+3 fp16 = 15). int8-dynamic (`include_int8d`) and
  pruning (`include_prune`) stay buildable behind flags, default False. Full original list:
  int8-dynamic yolo11n/yolo11s/yolov8n × 3; int8-static yolo11n/yolo11s × 3
  (`--quant-weights models/<m>_int8s.pt`); fp16 yolo11n × 3 (only with `include_fp16=True`;
  default False — §4 makes it probe-gated); pruned yolo11n p30/p50 × 3 (`--prune
  models/yolo11n_p30.pt`); int8-static yolo11n at 1500 and 2000 × 3. All 2400 MHz unless stated.
- **C (60, +6 optional)**: imgsz {320,480,960} × {yolo11n,yolo11s,yolo11m} × 3; batch {2,4,8}
  yolo11n@640 × 3; threads {1,2} × {yolo11n,yolo11s}@640 × 3; threads {1,2} × imgsz {320,960}
  yolo11n × 3; optional (`include_optional=True`) threads {1,2} yolo11m × 3.
- **D (34)** (the PDF's D table says 36, but its rows add up to 10+6+9+3+6 = 34): DVFS (`--governor ondemand`, cpu_mhz 2400 for sync) yolo11n, yolo11s × 5;
  fan auto / fan off yolo11n × 3 each (pinned 2400); serve-rate `--serve-fps` {1,5,15} yolo11n ×
  3 with `--serve <round(20 * fps)>` and no `--serve-gap-ms`; stream
  (`--frames-source stream --frames 50`) yolo11n × 3; concurrency yolo11n+yolo11n × 3 and
  yolo11n+spinner × 3 (args from Task 13: `--concurrent yolo11n.pt` and `--concurrent spinner`;
  build them behind `include_concurrent=True`, default True only after Task 13 lands — make the
  flag default False in this task).
- **soak (2)**: `--soak-minutes 30` yolo11n, yolo11m, pinned 2400, `--fan auto`,
  `timeout_s = 30*60 + 20*60`.
- **pilot**: exactly one cell per distinct configuration group: A per model @2400 (4) + yolo11s
  @1500 + @2000 (2); A-input empty + crowded (2); B int8s yolo11n, int8s yolo11s, int8s yolo11n@1500 (3, + fp16 if included; + int8d / p30 / p50
  only when their flags are on); C imgsz 320 + 960 yolo11n, 960 yolo11m, batch 2, batch
  8, threads 1, threads 2, threads 1 × 960 (8); D DVFS yolo11n × 2 repeats, fan auto, fan off,
  serve 1 fps, serve 15 fps, stream (7, + 2 concurrency when included); and one 5-minute soak
  of yolo11n (`soak5`, `--soak-minutes 5`). Cell ids get prefix `pilot_`.
- Rounds: like v1, round r holds repeat r of every config; shuffle within a round with
  `random.Random(seed)`.
- `experiment.py plan --v2 {A,A-input,B,C,D,soak,all,pilot} [--include-fp16]
  [--include-optional] [--include-concurrent] --out manifest.jsonl` writes the rows. v1 `plan`
  options unchanged.

Tests: exact counts per family (A 44, A-input 6, B 12 without fp16 / 15 with (27/30 with
`include_int8d` and `include_prune`), C 60 / 66,
D 28 without concurrency / 34 with, soak 2), cell-id uniqueness and format (regex), holdout tags,
DVFS cells carry `--governor ondemand` and `--cpu-mhz 2400`, every cell has `--perf-counters on`,
B cells have `--no-bench`, the args of an A cell parse with the Task 4/5 argparse (import
`src/run.py`'s parser builder if it is factored out — if not, factor `build_parser()` out of
`run.py` in this task).

### Task 13: concurrency mode — one process, one sync envelope

**Files:** `src/run.py`, `src/eventlog.py`, `src/tasks.py` (if needed), `host/align.py`,
`host/verify_cell.py`, tests on both sides.

- `--concurrent {<model>.pt,spinner}` adds a second workload **in the same run.py process**:
  - `<model>.pt`: a second `ImageTask` (same imgsz/batch/input) run in its own thread.
  - `spinner`: one thread doing a fixed 1-thread busy workload that releases the GIL (e.g. a
    small `torch.mm` loop under `torch.set_num_threads(1)` semantics for that thread, or a
    separate process pinned with `os.sched_setaffinity` to core 3 that emits no events — choose
    and justify; it must not emit sync pulses).
  - Thread split: choose and document a policy (e.g. each model gets `threads // 2` torch
    threads; the spinner uses 1 core) and record it in `run_meta["concurrent"]`.
  - Exactly one sync envelope (`sync_pulse("start")`/`("end")` once, by the main thread).
- EventLog: add a `tag` column (default `""`; v1 events.csv gains an empty column) so step
  events carry the model tag (`m0`, `m1`, `spin`). `EventLog._depth` is not thread-safe: in
  concurrent mode disable per-layer events (LayerProfiler `events=None`) and keep step events,
  which are depth-free. Record `"layer_events": false`.
- Measured phase: both workloads run concurrently for the same `--frames` count of the primary
  (secondary loops until the primary finishes); per-tag iteration counts and latencies in
  `run_meta["concurrent"]`.
- align.py: tolerate the new `tag` column (v1 files without it still work); add a `tag` column
  to the trace only when events have non-empty tags — label each sample with the tags whose step
  spans contain it (e.g. `"m0+m1"`), per-tag step columns optional.
- verify_cell.py: concurrent cells require step coverage per tag, skip the single-step
  expectations that assume one workload. Add `include_concurrent` default True in matrix_v2.
- Tests on both sides with synthetic events.

If the design can't keep one sync envelope and correct labels, stop and report BLOCKED with the
reason; the plan allows deferring concurrency (§13 Decision 4).

### Task 14: lab.py presets and checks for v2

**Files:** `host/lab.py`, `host/tests/test_lab.py`.

- `./lab presets` and `./lab run <preset>` know every v2 configuration: add the v2 presets from
  `matrix_v2` (one preset per distinct configuration group, named like the cell id without
  `_r..` and without the clock, e.g. `yolo11n_int8s_640_t4`, `yolo11n_fanoff_640_t4`,
  `yolo11n_640_t4_dvfs`), shown in their own "Dataset v2" section of `./lab presets`.
- `build_args` must keep v2 flags (`--fan`, `--governor`, `--quant`, `--quant-weights`,
  `--prune`, `--holdout`, `--perf-counters`, `--serve-fps`, `--soak-minutes`,
  `--frames-source`, `--input-image`, `--concurrent`) and must not append `--fan-full` when
  the preset sets `--fan`.
- `./lab check`: add the recorder stall check (Task 11's `recorder_stalled`) as an error row,
  a Wi-Fi check on the Pi (`cat /sys/class/net/wlan0/operstate` must not be `up` — a radio adds
  power noise — warning level), `perf` present on the Pi (warning), and the installed helper
  supports `unpin` and `fan off` (`grep -c unpin /usr/local/sbin/pi-power-pin` > 0, warning
  with the reinstall hint).
- Supervised fan-off: `./lab run yolo11n_fanoff_640_t4` prints a one-line warning that the first
  fan-off cell must be watched in person, and requires `--yes-supervised`.

Tests: preset listing contains the v2 section; `build_args` for a fan-off and a DVFS preset
(exact argv); check rows with fake shell output; `--yes-supervised` gate.

### Task 15: documentation

**Files:** `README.md` (§5.4 matrix, §6 output files, §11 reference/flags), `INSTRUCTIONS.md`
(the new presets), `tools/README.md` (already exists from Task 6 — update), new
`docs/dataset-v2.md` (short: what v2 adds, the v1-vs-v2 counter split, holdout definition,
Phase 0 probe checklist with the go/no-go questions from §9 and the commands to run each probe,
the one-time setup steps: reinstall pi-power-pin with sudo, copy `inputs/` and `models/` to the
Pi, turn Pi Wi-Fi off before measuring, the perf_event_paranoid note).

Every new run.py flag, every new output file (`sysfs.csv`, `counters.csv`, `counters.raw.txt`,
`thermal_abort.json`, `derived.json`) and every new `run_meta.json` field is documented once.
Plain language, same tone as the existing README. No measurement numbers that were not measured.

### Task 16: measure mAP on the Pi (the target device)

**Decided by the user 2026-11-20:** accuracy numbers come from the Pi, not the NUC — INT8
(qnnpack on ARM) and FP16 run different kernels on the Pi than on x86. The NUC keeps only offline
weight building (quantize, prune) and data prep. Runs after Tasks 8-9 (it evaluates their
artifacts too). Never during a measurement session.

**Files:** `tools/eval_map.py`, `tools/tests/test_eval_map.py`, `tools/README.md`,
`data/map_reference.json`.

- Make `tools/eval_map.py` runnable with the **Pi project environment** (repo root
  `pyproject.toml`: torch 2.14.0, ultralytics 8.4.152, opencv — no tools-only imports on the eval
  path), e.g. `cd ~/Desktop/power-measurement-pi && ~/.local/bin/uv run --frozen --offline python
  tools/eval_map.py ...`. Add `--data-dir` so the eval set location is explicit.
- Every entry records where it was measured: `"measured_on": {"hostname", "machine"
  (platform.machine()), "torch": torch.__version__, "quantized_engine"}`. run.py's lookup is
  unchanged (it only needs the key); verify (Task 10) warns when a cell's `map_reference` was not
  measured on an `aarch64` machine.
- Copy `tools/data/coco500` (images + labels + yaml) and `models/` to the Pi
  (`~/Desktop/power-measurement-pi/tools/data/coco500`, `models/`), and the six weights are already
  there.
- Run `--all` (plus every `models/*.pt` artifact) **on the Pi** in the background with `nohup`,
  logging to `runs/map_eval.log`, with the fan forced full (`sudo -n /usr/local/sbin/pi-power-pin
  fan full` before, `fan auto` after) and resumable per entry (`--skip-existing` semantics that only
  skip entries already measured on aarch64). The Pi has had intermittent hard lockups under
  sustained load (see README §9 / Pi lockup notes): if it locks up, wait for it, then resume.
- Replace all NUC entries in `data/map_reference.json` with the Pi results; keep the NUC numbers
  in the report as a cross-check table (Pi vs NUC per key, difference).
- Do not run anything while `pgrep -af '[s]rc/run.py'` finds a measurement on the Pi.
