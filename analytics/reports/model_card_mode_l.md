> **R1, the pre-registered analysis (frozen).** The post-review revision R2 (2026-11-24, exploratory) is summarised in [`r2/README.md`](r2/README.md); the measurement checks are in [`measurement_audit.md`](measurement_audit.md). This file is unchanged apart from this note; an exact copy is in `r1_frozen/`.

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

*Split: leave one YOLO base model out. Target: whole-cell energy. n = 161 held-out cells in 38 configurations and 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Interval: cluster bootstrap over the 5 held-out base models.

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

*Split: leave one YOLO base model out. Target: whole-cell energy. n = 161 held-out cells in 38 configurations and 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Per held-out base model, median whole-cell APE of the rung with the lowest overall MdAPE among the non-exploratory rungs.

| rung | held-out base model | n | MdAPE % |
|---|---|---|---|
| L3-trees | yolov8m | 11 | 1.9 |
| L3-trees | yolo11m | 21 | 1.5 |
| L3-trees | yolo11n | 84 | 1.4 |
| L3-trees | yolov8s | 11 | 0.8 |
| L3-trees | yolo11s | 34 | 0.7 |


*Split: leave one YOLO base model out. Target: phase energy. n = 161 held-out cells in 38 configurations and 5 folds (dataset v2, nov2026 rig). run_id: `acc4f1bf2b524febb2565e1d8fe3e79b`. Mode A's energy noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %, reused here as the indistinguishability margin.* Median APE per phase.

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
- The eight rungs are frozen (`final-mode-l`): each was fitted once on 161 cells (198,619 windows) of YOLO v8/11 models; non-YOLO models are an extrapolation the lockbox tests. A refit is refused from `lockbox-predict` onward (once `lockbox_predictions.csv` or `lockbox_frozen_models.json` exists), and always once lockbox cells exist.


## Hashes and runs

Pickles are not committed. Each is identified by the sha256 recorded in `runs.jsonl` (command `mode-a-extras`, `final-mode-a-secondary` or `final-mode-l`); `lockbox-predict` and `lockbox-score` refuse a pickle whose hash differs, and the fit commands refuse to run once `lockbox_predictions.csv` or `lockbox_frozen_models.json` exists. The `file` column compares the pickle on the machine that built this card with the recorded hash.

*Split: all training data. Target: frozen Mode A models. n = 7 frozen Mode A models. run_id: `2543bfac27da45c8bfaf5c92fbd1e31e`, `a730e59128f34439a920bd73dfe2b696`, `14fe0b4726384159b316722b224172c3`, `b4a2507e14b64bb883685408481acc74`, `329a9d1abd9543e68044a4449ad811f5`, `9130b293320c4c9899231458cc1135d9`, `953612c6e9c44922839f9b62cc2bb400`.* `mode-a-extras` and `final-mode-a-secondary`: trained on all 58 Mode A configurations, grid chosen by leave-one-model-out. Roles: `primary` is the AUTO-final candidate of the target (`mode-a-extras`), `secondary` the pre-declared extras (`final-mode-a-secondary`). 

| target | candidate | role | params | sha256 | run_id | command | trained on | AUTO chose (folds) | S-LOMO run | file |
|---|---|---|---|---|---|---|---|---|---|---|
| energy_mJ | M1Ridge | primary | {"alpha": 10.0} | 0d7b2cc339c242b06e5a7733cf1fb4a6429c30271081577c91452df6c904dea2 | 2543bfac27da45c8bfaf5c92fbd1e31e | mode-a-extras | 58 configurations | {"M1Ridge": 6, "M3Factored": 3} | aa634083491146caa1898cb53dd5b2dc | matches |
| energy_mJ | B3LogLinear | secondary | {} | cb0c40502b3b395bb0326d5c18e61b5be128f5fd4f9367a325f8e6781630b599 | a730e59128f34439a920bd73dfe2b696 | final-mode-a-secondary | 58 configurations |  |  | matches |
| energy_mJ | M3Factored | secondary | {"latency": {"name": "M1Ridge", "params": {"alpha": 10.0}}, "power": {"name": "M1Ridge", "params": {"alpha": 0.1}}} | b15b9ce067454687a96a27eadf31c5a70f4723e61fb9882f0c85fbe37f269950 | 14fe0b4726384159b316722b224172c3 | final-mode-a-secondary | 58 configurations |  |  | matches |
| latency_ms | M1Ridge | primary | {"alpha": 10.0} | d558746cd4c550a1c94c5255ce867a3e9a5f1a7d3a185131446141134a70b561 | b4a2507e14b64bb883685408481acc74 | mode-a-extras | 58 configurations | {"M1Ridge": 9} | d8d7ac3750c442d292c0a01511667235 | matches |
| latency_ms | B3LogLinear | secondary | {} | aee0cf2e2054290e13e4693e8b67d503a129bba7426607795977885f222638a9 | 329a9d1abd9543e68044a4449ad811f5 | final-mode-a-secondary | 58 configurations |  |  | matches |
| mean_P_W | M1Ridge | primary | {"alpha": 0.1} | 1180cf9f1c277014e575ba60f3ef7ddc84e7f37ce9d80fd907f5a45419adc961 | 9130b293320c4c9899231458cc1135d9 | mode-a-extras | 58 configurations | {"M1Ridge": 6, "M2Trees": 3} | a2b6e23cd011443f94994a250137159f | matches |
| mean_P_W | B3LogLinear | secondary | {} | 7e1d0bdbbf7f0cb24050ec04b7342ba8681b355ca823496d66223c6758f218da | 953612c6e9c44922839f9b62cc2bb400 | final-mode-a-secondary | 58 configurations |  |  | matches |


*Split: all training data. Target: frozen Mode L rungs. n = 8 frozen Mode L rungs. run_id: `66e8d6a51ac44595b0c46475b02a8c1b`, `5a06c7fda7914b8ebe8e040def4e9a2b`, `8238906e9a514389b4e60883272c652d`, `1dcfeb0b52d84bb2b8669c81637a6e36`, `97282373d11c468d9ebeb9c878a1762f`, `5dd3c845fc954b228b5849ed59488b50`, `95b3c3818cab45e8995863d1997b3346`, `b7f5c785c1f34e4489ee9519dd61b58f`.* `final-mode-l`: grid by leave-one-YOLO-model-out, refit on every QC-primary counter cell of the YOLO base models. 

| rung | role | params | sha256 | run_id | cells | windows | file |
|---|---|---|---|---|---|---|---|
| L0 | secondary (pre-declared) | {} | a4c593764baf99d9a085b92bbd80d760feed4747275225ee0703d29568252649 | 66e8d6a51ac44595b0c46475b02a8c1b | 161 | 198619 | matches |
| L1 | secondary (pre-declared) | {} | 00b9d92dc4bda699467a351635e96020e1df3b1484f3b02dc5d87ac6b352726c | 5a06c7fda7914b8ebe8e040def4e9a2b | 161 | 198619 | matches |
| L1-cell | exploratory | {} | d0913ceb68774f68b36c06aeeb35a6e1d6bb747f7ed1e5e6361878af9c618413 | 8238906e9a514389b4e60883272c652d | 161 | 198619 | matches |
| L2 | reported | {"alpha": 10.0} | 3a8ee3ca3ba1944fb428347d7b6834c8503ec473d4158820519eb3bd9871c6e5 | 1dcfeb0b52d84bb2b8669c81637a6e36 | 161 | 198619 | matches |
| L3-ridge | reported | {"alpha": 0.1} | 6d6308ad41c04b9e94ac5f6194173e901120a26bf87c42d9b51feb153fcf5b78 | 97282373d11c468d9ebeb9c878a1762f | 161 | 198619 | matches |
| L3-trees | secondary (pre-declared) | {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50} | 0d3aee5cfc383dd5307953d08c14f8d9378ab8929f455545fd3304f40e3c9ea5 | 5dd3c845fc954b228b5849ed59488b50 | 161 | 198619 | matches |
| L4-ridge | reported | {"alpha": 0.01} | e4960733492f3142bef2e3244a36623583c126235a064d33f8ea9ebb613ac211 | 95b3c3818cab45e8995863d1997b3346 | 161 | 198619 | matches |
| L4-trees | secondary (pre-declared) | {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50} | 9a711735eb3c665063f319e3c43f9f0b4a8d4a24432e57482fddf65ce58314fc | b7f5c785c1f34e4489ee9519dd61b58f | 161 | 198619 | matches |


Evaluation: Mode L run `acc4f1bf2b524febb2565e1d8fe3e79b` (parts lomo, pmic_v1, probe), seed 20261122, commit `ad0ab1cdc1`, data hash `09890ae9fab489b9`. Probe mode used Mode A run(s) `aa634083491146caa1898cb53dd5b2dc`.
