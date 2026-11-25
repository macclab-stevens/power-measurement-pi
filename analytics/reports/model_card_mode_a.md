> **R1, the pre-registered analysis (frozen).** The post-review revision R2 (2026-11-24, exploratory) is summarised in [`r2/README.md`](r2/README.md); the measurement checks are in [`measurement_audit.md`](measurement_audit.md). This file is unchanged apart from this note; an exact copy is in `r1_frozen/`. Two top-level data files were regenerated under R2: `deploy_validation.csv` (gap scenario added) and `mode_a_configs.csv` (R2 columns added). Their R1 versions are `r1_frozen/deploy_validation.csv` and `r1_frozen/mode_a_configs.csv`.

# Model card: Mode A (a priori per-image predictor)

Predicts energy per image (mJ), latency per image (ms) and mean power (W) of an image-inference configuration from the model file's static features and the run configuration, before anything is run. Mode: a priori; nothing measured during the run is an input.

Built 2026-11-23 from commit `ad0ab1cdc1` (dirty working tree), data hash `09890ae9fab489b9`, run seed 20261122. Full results: `results.md`.

## Intended use

- Planning the energy and latency of a vision model (YOLO family and classifiers like those in the data) on this Raspberry Pi 5 at a given image size, thread count, clock and precision, through `analytics.deploy.predict` and its 90 % interval.
- Reading the `no-similar-model` applicability flag: predictions flagged by novelty are outside what the validation supports.


## Out of scope

- Other boards than the one Raspberry Pi 5 measured here, other meters than the Monsoon rig, other runtimes or kernels.
- Claims about the SoC alone: every energy figure is gross board energy at the Monsoon input (SoC plus the rest of the board).
- Workloads other than single-model image inference: language-model, training and concurrent multi-workload energy are described only.
- Predicting latency or energy of models with no similar training model without the flag.
- Governor and fan effects other than through the deployment calculator's measured offsets (rest on yolo11n).


## Data and split

Training data: the Mode A configuration table, 58 configurations from 226 QC-primary cells; n = 58 configurations / 226 cells (dataset v1 (oct2026 rig) 80, v2 (nov2026 rig) 146 cells). Headline split S-LOMO (leave one base model out, 9 folds); other splits S-LSO, S-LVO, S-LCO, RANDOM are in `results.md`. Hyperparameters and AUTO are chosen by inner leave-one-model-out on each outer fold's training models only.

## Features

Known before running anything: `log_macs`, `log_params`, `log_weight_bytes`, `log_act_bytes`, `depth`, `frac_macs_conv`, `frac_macs_linear`, `frac_macs_norm`, `frac_macs_act`, `frac_macs_pool`, `frac_macs_other`, `imgsz`, `batch`, `threads`, `cpu_ghz`, `is_int8`, `is_fp16`, `n_quantized_modules`, `rig_nov`. `rig_nov` is 1 for the nov2026 rig and 0 for the oct2026 rig (derived from the `rig` column); M1 and M2 use all of them, B3 uses `log_macs`, `log_params`, `cpu_ghz`, `threads`, `batch`, `is_int8`, `is_fp16`. Never used: latency, temperature, baseline power or counters of the run being predicted.

## Headline metrics

*Split: S-LOMO. Target: energy_mJ, latency_ms, mean_P_W. n = 58 configurations / 226 cells (dataset v1 (oct2026 rig) 80, v2 (nov2026 rig) 146 cells). run_id: `aa634083491146caa1898cb53dd5b2dc`, `d8d7ac3750c442d292c0a01511667235`, `a2b6e23cd011443f94994a250137159f`. Noise floors (median repeat CV, configurations with at least 2 cells): energy 0.6 %, latency 0.2 %, mean power not defined.* Pooled over test configurations; intervals are cluster bootstrap over base models.

| target | predictor | n | MdAPE % [95 % CI] | bias % [95 % CI] | coverage90 | run_id |
|---|---|---|---|---|---|---|
| energy_mJ | AUTO | 58 | 17.5 [13.8, 22.8] | -16.4 [-18.4, 13.1] | 0.9138 | aa634083491146caa1898cb53dd5b2dc |
| energy_mJ | B3 log-linear | 58 | 15.8 [8.9, 41.6] | -3.6 [-28.8, 36.5] | 0.8793 | aa634083491146caa1898cb53dd5b2dc |
| latency_ms | AUTO | 58 | 18.3 [12.6, 20.9] | -17.8 [-20.8, 9.8] | 0.8621 | d8d7ac3750c442d292c0a01511667235 |
| latency_ms | B3 log-linear | 58 | 16.8 [8.9, 47.8] | -4.6 [-31.5, 35.4] | 0.8966 | d8d7ac3750c442d292c0a01511667235 |
| mean_P_W | AUTO | 58 | 3.6 [2.9, 4.9] | 2.7 [1.0, 4.7] | 0.8621 | a2b6e23cd011443f94994a250137159f |
| mean_P_W | B3 log-linear | 58 | 2.9 [2.4, 3.3] | 1.3 [-1.8, 2.9] | 0.9138 | a2b6e23cd011443f94994a250137159f |


Energy headline (AUTO MdAPE, S-LOMO) over the protocol seeds: seed 20261122: 17.5 %; seed 20261123: 17.5 %; seed 20261124: 17.5 %; see `results.md`, section 3.

## Coverage

The 90 % prediction interval comes from inner leave-one-model-out residual quantiles. Coverage of AUTO on S-LOMO energy by similarity class (coarse with 9 models):

*Split: S-LOMO. Target: energy_mJ. n = 58 configurations / 226 cells (dataset v1 (oct2026 rig) 80, v2 (nov2026 rig) 146 cells). run_id: `aa634083491146caa1898cb53dd5b2dc`. Noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %.* AUTO rows only.

| similarity class | n | MdAPE % | MAPE % | bias % | coverage90 |
|---|---|---|---|---|---|
| new-architecture | 9 | 94.9 | 1487.8 | -16.9 | 0.6667 |
| sibling-seen | 49 | 16.8 | 17.6 | -16.2 | 0.9592 |


## Worst subgroups

AUTO on S-LOMO energy, the three base models and the similarity classes with the largest MdAPE:

*Split: S-LOMO. Target: energy_mJ. n = 58 configurations / 226 cells (dataset v1 (oct2026 rig) 80, v2 (nov2026 rig) 146 cells). run_id: `aa634083491146caa1898cb53dd5b2dc`. Noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %.* By base model.

| base_model | n | AUTO MdAPE % | bias % |
|---|---|---|---|
| effb0 | 3 | 4340.8 | 4340.8 |
| resnet18 | 3 | 94.9 | -94.9 |
| yolo11m | 6 | 22.8 | -22.8 |


*Split: S-LOMO. Target: energy_mJ. n = 58 configurations / 226 cells (dataset v1 (oct2026 rig) 80, v2 (nov2026 rig) 146 cells). run_id: `aa634083491146caa1898cb53dd5b2dc`. Noise floor (median repeat CV, configurations with at least 2 cells) 0.6 %.* By similarity class.

| similarity_class | n | AUTO MdAPE % | bias % |
|---|---|---|---|
| new-architecture | 9 | 94.9 | -16.9 |
| sibling-seen | 49 | 16.8 | -16.2 |


## Limitations

- A single device and a single measurement rig pair (the oct2026 and nov2026 rigs): between-board and between-meter variation is not measured.
- The lockbox (about 15 new cells collected after the `analytics-frozen` tag) is pending: nothing here is a pristine out-of-sample test, and the protocol note discloses that configuration-level summaries were viewed before any model was fitted.
- Mode A and Mode L fold counts are small (9 and 5 base models); intervals and coverage estimates are coarse.
- Mode A trains on pinned-clock, fan-full cells; the deployment calculator adds governor, fan and soak effects from single measured offsets (one soak per model, fan offsets from yolo11n).
- Static features of INT8 and FP16 variants come from the FP32 base model; FP16 (one configuration) is unsupported.


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


Table sha256 `8a1c44ad7cc16fb93b7da809836f574a365f0857250a416403ff32eadb221061`. Split sha256: S-LOMO `08440d25646af8354635b78a206cd3320eff7562bfa10650e972aed9149a6d36`, S-LSO `d01194ab1e8d865e04be94dc3fec1d7c1897659a62d9a7238810371ec4bc3767`, S-LVO `be7b1e6e3f0b5b845641f148d87a1a91a04230badaf1abe90c6f30211bca687a`, S-LCO `b9951066ab730b716e5bfa72a42a485cbbde9f1537c89c72272924464e8a5cea`, RANDOM `406ede1bdaed0b4cb1720325802a0884e652d0ee0f1e67f95fb2305b68d9c422`.
