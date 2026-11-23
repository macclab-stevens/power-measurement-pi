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

Known before running anything: `log_macs`, `log_params`, `log_weight_bytes`, `log_act_bytes`, `depth`, `frac_macs_conv`, `frac_macs_linear`, `frac_macs_norm`, `frac_macs_act`, `frac_macs_pool`, `frac_macs_other`, `imgsz`, `batch`, `threads`, `cpu_ghz`, `is_int8`, `is_fp16`, `n_quantized_modules`. Never used: latency, temperature, baseline power or counters of the run being predicted.

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

*Split: all data. Target: energy_mJ, latency_ms, mean_P_W. n = 3 final models trained on 58 configurations. run_id: `2543bfac27da45c8bfaf5c92fbd1e31e`, `b4a2507e14b64bb883685408481acc74`, `9130b293320c4c9899231458cc1135d9`.* Final models are refitted on every configuration, grid chosen by leave-one-model-out; binaries are not committed (rebuilt by `cli.py mode-a-extras`).

| target | candidate | params | AUTO chose (folds) | sha256 | run_id |
|---|---|---|---|---|---|
| energy_mJ | M1Ridge | {"alpha": 10.0} | {"M1Ridge": 6, "M3Factored": 3} | 0d7b2cc339c242b06e5a7733cf1fb4a6429c30271081577c91452df6c904dea2 | 2543bfac27da45c8bfaf5c92fbd1e31e |
| latency_ms | M1Ridge | {"alpha": 10.0} | {"M1Ridge": 9} | d558746cd4c550a1c94c5255ce867a3e9a5f1a7d3a185131446141134a70b561 | b4a2507e14b64bb883685408481acc74 |
| mean_P_W | M1Ridge | {"alpha": 0.1} | {"M1Ridge": 6, "M2Trees": 3} | 1180cf9f1c277014e575ba60f3ef7ddc84e7f37ce9d80fd907f5a45419adc961 | 9130b293320c4c9899231458cc1135d9 |


Table sha256 `8a1c44ad7cc16fb93b7da809836f574a365f0857250a416403ff32eadb221061`. Split sha256: S-LOMO `08440d25646af8354635b78a206cd3320eff7562bfa10650e972aed9149a6d36`, S-LSO `d01194ab1e8d865e04be94dc3fec1d7c1897659a62d9a7238810371ec4bc3767`, S-LVO `be7b1e6e3f0b5b845641f148d87a1a91a04230badaf1abe90c6f30211bca687a`, S-LCO `b9951066ab730b716e5bfa72a42a485cbbde9f1537c89c72272924464e8a5cea`, RANDOM `406ede1bdaed0b4cb1720325802a0884e652d0ee0f1e67f95fb2305b68d9c422`.
