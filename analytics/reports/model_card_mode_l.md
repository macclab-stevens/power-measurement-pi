# Model card: Mode L (live 100 ms power model)

Predicts board power (W) for each 100 ms window from the process's perf counters, sysfs state and the PMIC read, and accumulates it to energy. Mode: live, causal (features available at the window's end only); a ladder of rungs from a lookup (L0) to counters plus PMIC (L4).

Built 2026-11-23 from commit `ad0ab1cdc1` (dirty working tree), data hash `09890ae9fab489b9`, run seed 20261122. Full results: `results.md`.

## Intended use

- Live energy accounting of a dedicated single-process image-inference deployment on this Raspberry Pi 5, with the same perf event set and sampling interval as the training cells.
- Probe mode: a short unmetered run of a new model, scored against Mode A on the same configurations (`results.md`, section 11).


## Out of scope

- Other boards than the one Raspberry Pi 5 measured here, other meters than the Monsoon rig, other runtimes or kernels.
- Claims about the SoC alone: every energy figure is gross board energy at the Monsoon input (SoC plus the rest of the board).
- Workloads other than single-model image inference: language-model, training and concurrent multi-workload energy are described only.
- System-wide or multi-process loads: counters cover the measured process's user-space code only (`perf_event_paranoid=2`), other processes are invisible.
- Windows shorter than 100 ms, per-step or per-layer attribution, a live scheduler.
- Non-YOLO models: counter data is YOLO-only until the lockbox.


## Data and split

v2 counter cells of the YOLO models (161 held-out cells over folds yolo11m, yolo11n, yolo11s, yolov8m, yolov8s; dataset v2, nov2026 rig). Split: leave one YOLO base model out; the grid point of each rung is chosen by inner leave-one-model-out on the training models; every window is weighted 1 / (windows in its cell). Windows before a cell's first available PMIC read have no PMIC feature; L1 predicts them with the fold's L0 lookup.

`L1-cell` is exploratory: it was added after a scored smoke fold (protocol amendment of 2026-11-23, `analytics/PROTOCOL.md`). `L1` is the pre-registered PMIC rung.

## Features

L0: clock, governor, fan and busy / idle (task-clock share over 0.5). L1: PMIC rail power with zero-order hold. L2: task-clock and cycle rates. L3: all counter rates (IPC, cache and branch miss rates, backend stall share, LLC misses, context switches, page faults), sysfs and settings. L4: L3 plus the PMIC features. All features use data available at the window's end.

## Headline metrics

*Split: leave one YOLO base model out. Target: whole-cell energy. n = 161 held-out cells in 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Interval: cluster bootstrap over the 5 held-out base models.

| rung | n | whole-cell MdAPE % [95 % CI] | whole-cell bias % [95 % CI] | 1 s block APE % | 10 s block APE % |
|---|---|---|---|---|---|
| L0 | 161 | 5.8 [3.4, 7.4] | 3.4 [-4.9, 6.6] | 7.0 | 6.1 |
| L1 | 161 | 6.7 [4.7, 13.2] | 1.6 [-7.7, 4.5] | 17.8 | 8.6 |
| L1-cell (exploratory) | 161 | 5.6 [4.2, 6.8] | 0.9 [-3.8, 3.0] | 17.0 | 8.5 |
| L2 | 161 | 5.1 [4.3, 8.7] | 2.3 [-4.4, 4.4] | 5.4 | 5.5 |
| L3-ridge | 161 | 2.9 [2.0, 4.1] | 0.3 [-2.9, 1.1] | 3.9 | 3.2 |
| L3-trees | 161 | 1.2 [0.8, 1.5] | -0.0 [-1.5, 0.7] | 1.3 | 1.4 |
| L4-ridge | 161 | 3.1 [2.2, 4.3] | 0.5 [-3.0, 1.1] | 3.9 | 3.4 |
| L4-trees | 161 | 1.2 [0.6, 1.5] | 0.1 [-1.5, 0.9] | 1.6 | 1.4 |


Counter overhead: counters-off (`ref`) cells drew +0.320 W (+3.34 %) against the same 1 configuration with counters on (`reports/eda.md`, section 7.2; family, order and the counters switch are confounded, so this bounds the overhead only loosely). Perf was running in every v2 counter cell, so the overhead is inside every number here.

## Coverage

Mode L produces point predictions only: no prediction intervals, so there is no coverage figure. The stated uncertainty is the bootstrap interval of the metric above.

## Worst subgroups

*Split: leave one YOLO base model out. Target: whole-cell energy. n = 161 held-out cells in 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Per held-out base model, median whole-cell APE of the rung with the lowest overall MdAPE among the non-exploratory rungs.

| rung | held-out base model | n | MdAPE % |
|---|---|---|---|
| L3-trees | yolov8m | 11 | 1.9 |
| L3-trees | yolo11m | 21 | 1.5 |
| L3-trees | yolo11n | 84 | 1.4 |
| L3-trees | yolov8s | 11 | 0.8 |
| L3-trees | yolo11s | 34 | 0.7 |


*Split: leave one YOLO base model out. Target: phase energy. n = 161 held-out cells in 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Median APE per phase.

| rung | n | idle | load_weights | warmup | forward | serve |
|---|---|---|---|---|---|---|
| L0 | 161 | 12.9 (n=161) |  | 12.6 (n=161) | 5.8 (n=161) | 6.8 (n=161) |
| L1 | 161 | 56.3 (n=161) |  | 60.8 (n=161) | 13.6 (n=161) | 11.1 (n=161) |
| L1-cell (exploratory) | 161 | 42.6 (n=161) |  | 47.3 (n=161) | 12.2 (n=161) | 10.9 (n=161) |
| L2 | 161 | 3.1 (n=161) |  | 3.7 (n=161) | 7.0 (n=161) | 5.4 (n=161) |
| L3-ridge | 161 | 3.5 (n=161) |  | 5.5 (n=161) | 3.3 (n=161) | 3.1 (n=161) |
| L3-trees | 161 | 0.5 (n=161) |  | 0.5 (n=161) | 1.8 (n=161) | 1.4 (n=161) |
| L4-ridge | 161 | 3.7 (n=161) |  | 6.2 (n=161) | 3.8 (n=161) | 3.2 (n=161) |
| L4-trees | 161 | 0.6 (n=161) |  | 0.7 (n=161) | 1.9 (n=161) | 1.9 (n=161) |


## Limitations

- A single device and a single measurement rig pair (the oct2026 and nov2026 rigs): between-board and between-meter variation is not measured.
- The lockbox (about 15 new cells collected after the `analytics-frozen` tag) is pending: nothing here is a pristine out-of-sample test, and the protocol note discloses that configuration-level summaries were viewed before any model was fitted.
- Mode A and Mode L fold counts are small (9 and 5 base models); intervals and coverage estimates are coarse.
- Counters cover the measured process's user-space code only (`perf_event_paranoid=2`). Other processes are invisible, so Mode L claims hold for a dedicated single-process deployment.
- Counter data is YOLO-only until the lockbox adds non-YOLO cells; Mode L folds are the five YOLO models with counter cells (yolov8n has none: its v2 cells are the counters-off `ref` family).
- Perf runs during deployment too, so its overhead is inside both the training data and any later use.
- One Raspberry Pi 5 and one Monsoon rig; nothing here transfers to other boards or meters.
- No Mode L model file is frozen yet: the final rung is refitted on all data when the lockbox is frozen.


## Hashes and runs

Mode L run `acc4f1bf2b524febb2565e1d8fe3e79b` (parts lomo, pmic_v1, probe), seed 20261122, commit `ad0ab1cdc1`, data hash `09890ae9fab489b9`. Probe mode used Mode A run(s) `aa634083491146caa1898cb53dd5b2dc`.
