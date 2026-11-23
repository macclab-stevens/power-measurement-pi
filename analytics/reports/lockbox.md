# Lockbox results

**EXPLORATORY. The scoring code is not the code of `lockbox-predict` and was scored with --allow-code-change; changed files: analytics/data.py.** This is not the pre-registered lockbox result.

Scored once; any later change is exploratory. Nothing was fitted on lockbox data: Mode A predictions are the committed `lockbox_predictions.csv`, Mode L uses the frozen window models.

## Provenance

- predictions file `analytics/reports/lockbox_predictions.csv` sha256 `2bb7a2606f3b42d9862de0b2e2b16866bad40a3418bcbb735d62812fb1304993` (written by `lockbox-predict` run `3f1736a7d94548fbb1b0a1a8f12cd060`)
- frozen-models manifest `analytics/reports/lockbox_frozen_models.json` (checked against the runlog line); predictions written 2026-11-23T20:03:51+00:00, earliest lockbox cell started 2026-11-23T20:04:13+00:00
- scoring run `bfff786a637b4030ade0f37ee1cf9163`; earlier scoring runs: 0
- scoring code DIFFERS from `lockbox-predict`: analytics/data.py; recorded commit `76673341dbd9bd2db3130b9eaa7232217ab0ea1f`: descendant of the recorded commit
- Static features: the layer tables of the measured cells equal the static features the predictions used (15 QC-primary cells in 5 configurations; macs and params exactly, other features to a relative 1e-09)
- idle table matches the one used at `lockbox-predict`: yes
- lockbox cells 15 (15 QC-primary): lockbox_mbv3_224_t4_2400_r00, lockbox_mbv3_224_t4_2400_r01, lockbox_mbv3_224_t4_2400_r02, lockbox_resnet18_224_t4_2400_r00, lockbox_resnet18_224_t4_2400_r01, lockbox_resnet18_224_t4_2400_r02, lockbox_yolo12n_320_t4_2400_r00, lockbox_yolo12n_320_t4_2400_r01, lockbox_yolo12n_320_t4_2400_r02, lockbox_yolo12n_640_t4_2400_r00, lockbox_yolo12n_640_t4_2400_r01, lockbox_yolo12n_640_t4_2400_r02, lockbox_yolov8n_480_t4_2400_r00, lockbox_yolov8n_480_t4_2400_r01, lockbox_yolov8n_480_t4_2400_r02
- frozen Mode A models (candidate / target: sha256, runlog run):
  - M1Ridge / energy_mJ: `0d7b2cc339c242b06e5a7733cf1fb4a6429c30271081577c91452df6c904dea2` (2543bfac27da45c8bfaf5c92fbd1e31e)
  - B3LogLinear / energy_mJ: `cb0c40502b3b395bb0326d5c18e61b5be128f5fd4f9367a325f8e6781630b599` (a730e59128f34439a920bd73dfe2b696)
  - M3Factored / energy_mJ: `b15b9ce067454687a96a27eadf31c5a70f4723e61fb9882f0c85fbe37f269950` (14fe0b4726384159b316722b224172c3)
  - M1Ridge / latency_ms: `d558746cd4c550a1c94c5255ce867a3e9a5f1a7d3a185131446141134a70b561` (b4a2507e14b64bb883685408481acc74)
  - B3LogLinear / latency_ms: `aee0cf2e2054290e13e4693e8b67d503a129bba7426607795977885f222638a9` (329a9d1abd9543e68044a4449ad811f5)
  - M1Ridge / mean_P_W: `1180cf9f1c277014e575ba60f3ef7ddc84e7f37ce9d80fd907f5a45419adc961` (9130b293320c4c9899231458cc1135d9)
  - B3LogLinear / mean_P_W: `7e1d0bdbbf7f0cb24050ec04b7342ba8681b355ca823496d66223c6758f218da` (953612c6e9c44922839f9b62cc2bb400)
- frozen Mode L models (rung: sha256, runlog run):
  - L0: `a4c593764baf99d9a085b92bbd80d760feed4747275225ee0703d29568252649` (66e8d6a51ac44595b0c46475b02a8c1b)
  - L1: `00b9d92dc4bda699467a351635e96020e1df3b1484f3b02dc5d87ac6b352726c` (5a06c7fda7914b8ebe8e040def4e9a2b)
  - L1-cell: `d0913ceb68774f68b36c06aeeb35a6e1d6bb747f7ed1e5e6361878af9c618413` (8238906e9a514389b4e60883272c652d)
  - L2: `3a8ee3ca3ba1944fb428347d7b6834c8503ec473d4158820519eb3bd9871c6e5` (1dcfeb0b52d84bb2b8669c81637a6e36)
  - L3-ridge: `6d6308ad41c04b9e94ac5f6194173e901120a26bf87c42d9b51feb153fcf5b78` (97282373d11c468d9ebeb9c878a1762f)
  - L3-trees: `0d3aee5cfc383dd5307953d08c14f8d9378ab8929f455545fd3304f40e3c9ea5` (5dd3c845fc954b228b5849ed59488b50)
  - L4-ridge: `e4960733492f3142bef2e3244a36623583c126235a064d33f8ea9ebb613ac211` (95b3c3818cab45e8995863d1997b3346)
  - L4-trees: `9a711735eb3c665063f319e3c43f9f0b4a8d4a24432e57482fddf65ce58314fc` (b7f5c785c1f34e4489ee9519dd61b58f)

## Measured configurations

| base_model | imgsz | n_cells | cell_ids | energy_mJ | latency_ms | mean_P_W |
|---|---|---|---|---|---|---|
| mbv3 | 224 | 3 | lockbox_mbv3_224_t4_2400_r00;lockbox_mbv3_224_t4_2400_r01;lockbox_mbv3_224_t4_2400_r02 | 289.4 | 31.95 | 9.059 |
| resnet18 | 224 | 3 | lockbox_resnet18_224_t4_2400_r00;lockbox_resnet18_224_t4_2400_r01;lockbox_resnet18_224_t4_2400_r02 | 859.7 | 79.27 | 10.85 |
| yolo12n | 320 | 3 | lockbox_yolo12n_320_t4_2400_r00;lockbox_yolo12n_320_t4_2400_r01;lockbox_yolo12n_320_t4_2400_r02 | 1189 | 126.2 | 9.418 |
| yolo12n | 640 | 3 | lockbox_yolo12n_640_t4_2400_r00;lockbox_yolo12n_640_t4_2400_r01;lockbox_yolo12n_640_t4_2400_r02 | 3700 | 373.8 | 9.898 |
| yolov8n | 480 | 3 | lockbox_yolov8n_480_t4_2400_r00;lockbox_yolov8n_480_t4_2400_r01;lockbox_yolov8n_480_t4_2400_r02 | 1655 | 163.3 | 10.13 |

## Mode A: pooled

`new models` are those with novelty above 0 (outside the 9 training models); training models (resnet18, mbv3, yolov8n) have novelty 0; `new size of a training model` is a training model at an image size the training table has not seen (e.g. yolov8n at 480 px), and resnet18 and mbv3 at 224 px are the closest to in-sample (same size, other rig). Novelty is per base model: a model's 320 px configuration uses its 640 px architecture descriptor (as in S-LOMO), so its novelty equals that of its 640 px configuration. The applicability threshold comes from S-LOMO, where novelty is standardised over the 8 training models of each fold; the lockbox novelty is standardised over all 9 training models, so the two scales are close but not identical.

| target | role | candidate | group | n | mdape | mape | bias_pct | inside_90 | n_flagged |
|---|---|---|---|---|---|---|---|---|---|
| energy_mJ | primary | M1Ridge | all | 5 | 7.4% | 8.0% | 7.2% | 100.0% | 2 |
| energy_mJ | primary | M1Ridge | new models | 2 | 6.6% | 6.6% | -0.8% | 100.0% | 2 |
| energy_mJ | primary | M1Ridge | new size of a training model | 1 | 10.9% | 10.9% | 10.9% | 100.0% | 0 |
| energy_mJ | secondary | B3LogLinear | all | 5 | 31.5% | 44.2% | -2.8% | 80.0% | 2 |
| energy_mJ | secondary | B3LogLinear | new models | 2 | 11.8% | 11.8% | -11.8% | 100.0% | 2 |
| energy_mJ | secondary | B3LogLinear | new size of a training model | 1 | 45.4% | 45.4% | 45.4% | 100.0% | 0 |
| energy_mJ | secondary | M3Factored | all | 5 | 12.7% | 11.1% | 12.7% | 100.0% | 2 |
| energy_mJ | secondary | M3Factored | new models | 2 | 6.9% | 6.9% | 6.9% | 100.0% | 2 |
| energy_mJ | secondary | M3Factored | new size of a training model | 1 | 15.7% | 15.7% | 15.7% | 100.0% | 0 |
| latency_ms | primary | M1Ridge | all | 5 | 11.8% | 10.2% | 11.8% | 100.0% | 0 |
| latency_ms | primary | M1Ridge | new models | 2 | 5.3% | 5.3% | 2.0% | 100.0% | 0 |
| latency_ms | primary | M1Ridge | new size of a training model | 1 | 14.4% | 14.4% | 14.4% | 100.0% | 0 |
| latency_ms | secondary | B3LogLinear | all | 5 | 32.5% | 48.1% | -6.0% | 80.0% | 0 |
| latency_ms | secondary | B3LogLinear | new models | 2 | 14.0% | 14.0% | -14.0% | 100.0% | 0 |
| latency_ms | secondary | B3LogLinear | new size of a training model | 1 | 47.0% | 47.0% | 47.0% | 100.0% | 0 |
| mean_P_W | primary | M1Ridge | all | 5 | 1.7% | 2.8% | 1.7% | 80.0% | 0 |
| mean_P_W | primary | M1Ridge | new models | 2 | 4.8% | 4.8% | 4.8% | 50.0% | 0 |
| mean_P_W | primary | M1Ridge | new size of a training model | 1 | 1.2% | 1.2% | 1.2% | 100.0% | 0 |
| mean_P_W | secondary | B3LogLinear | all | 5 | 1.6% | 2.6% | 1.4% | 100.0% | 0 |
| mean_P_W | secondary | B3LogLinear | new models | 2 | 2.4% | 2.4% | 2.4% | 100.0% | 0 |
| mean_P_W | secondary | B3LogLinear | new size of a training model | 1 | 1.1% | 1.1% | -1.1% | 100.0% | 0 |

## Mode A: per configuration

| model | imgsz | target | role | candidate | pred | lo90 | hi90 | truth | ape | inside | flagged | novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo12n | 640 | energy_mJ | primary | M1Ridge | 3427 | 1533 | 8128 | 3700 | 7.4% | yes | yes | 2.872 |
| yolo12n | 320 | energy_mJ | primary | M1Ridge | 1257 | 562.5 | 2982 | 1189 | 5.8% | yes | yes | 2.872 |
| yolov8n | 480 | energy_mJ | primary | M1Ridge | 1835 | 821 | 4352 | 1655 | 10.9% | yes | no | 0 |
| resnet18 | 224 | energy_mJ | primary | M1Ridge | 921.6 | 412.3 | 2186 | 859.7 | 7.2% | yes | no | 0 |
| mbv3 | 224 | energy_mJ | primary | M1Ridge | 315.3 | 141.1 | 747.8 | 289.4 | 8.9% | yes | no | 0 |
| yolo12n | 640 | energy_mJ | secondary | B3LogLinear | 2929 | 1843 | 6430 | 3700 | 20.8% | yes | yes | 2.872 |
| yolo12n | 320 | energy_mJ | secondary | B3LogLinear | 1155 | 727.3 | 2536 | 1189 | 2.8% | yes | yes | 2.872 |
| yolov8n | 480 | energy_mJ | secondary | B3LogLinear | 2406 | 1514 | 5282 | 1655 | 45.4% | yes | no | 0 |
| resnet18 | 224 | energy_mJ | secondary | B3LogLinear | 1896 | 1193 | 4162 | 859.7 | 120.5% | no | no | 0 |
| mbv3 | 224 | energy_mJ | secondary | B3LogLinear | 198.4 | 124.9 | 435.5 | 289.4 | 31.5% | yes | no | 0 |
| yolo12n | 640 | energy_mJ | secondary | M3Factored | 3738 | 3155 | 9329 | 3700 | 1.0% | yes | yes | 2.872 |
| yolo12n | 320 | energy_mJ | secondary | M3Factored | 1340 | 1131 | 3344 | 1189 | 12.7% | yes | yes | 2.872 |
| yolov8n | 480 | energy_mJ | secondary | M3Factored | 1914 | 1615 | 4778 | 1655 | 15.7% | yes | no | 0 |
| resnet18 | 224 | energy_mJ | secondary | M3Factored | 964.5 | 813.9 | 2407 | 859.7 | 12.2% | yes | no | 0 |
| mbv3 | 224 | energy_mJ | secondary | M3Factored | 329.1 | 277.7 | 821.2 | 289.4 | 13.7% | yes | no | 0 |
| yolo12n | 640 | latency_ms | primary | M1Ridge | 361.3 | 305.3 | 879.2 | 373.8 | 3.4% | yes | no | 2.872 |
| yolo12n | 320 | latency_ms | primary | M1Ridge | 135.4 | 114.4 | 329.5 | 126.2 | 7.3% | yes | no | 2.872 |
| yolov8n | 480 | latency_ms | primary | M1Ridge | 186.8 | 157.8 | 454.6 | 163.3 | 14.4% | yes | no | 0 |
| resnet18 | 224 | latency_ms | primary | M1Ridge | 90.4 | 76.4 | 220 | 79.27 | 14.1% | yes | no | 0 |
| mbv3 | 224 | latency_ms | primary | M1Ridge | 35.71 | 30.18 | 86.91 | 31.95 | 11.8% | yes | no | 0 |
| yolo12n | 640 | latency_ms | secondary | B3LogLinear | 291.9 | 178.1 | 660.1 | 373.8 | 21.9% | yes | no | 2.872 |
| yolo12n | 320 | latency_ms | secondary | B3LogLinear | 118.6 | 72.39 | 268.2 | 126.2 | 6.0% | yes | no | 2.872 |
| yolov8n | 480 | latency_ms | secondary | B3LogLinear | 240.2 | 146.6 | 543.1 | 163.3 | 47.0% | yes | no | 0 |
| resnet18 | 224 | latency_ms | secondary | B3LogLinear | 184.7 | 112.7 | 417.6 | 79.27 | 133.0% | no | no | 0 |
| mbv3 | 224 | latency_ms | secondary | B3LogLinear | 21.56 | 13.16 | 48.75 | 31.95 | 32.5% | yes | no | 0 |
| yolo12n | 640 | mean_P_W | primary | M1Ridge | 10.35 | 9.875 | 11.35 | 9.898 | 4.5% | yes | no | 2.872 |
| yolo12n | 320 | mean_P_W | primary | M1Ridge | 9.898 | 9.446 | 10.86 | 9.418 | 5.1% | no | no | 2.872 |
| yolov8n | 480 | mean_P_W | primary | M1Ridge | 10.25 | 9.781 | 11.24 | 10.13 | 1.2% | yes | no | 0 |
| resnet18 | 224 | mean_P_W | primary | M1Ridge | 10.67 | 10.18 | 11.7 | 10.85 | 1.6% | yes | no | 0 |
| mbv3 | 224 | mean_P_W | primary | M1Ridge | 9.214 | 8.794 | 10.11 | 9.059 | 1.7% | yes | no | 0 |
| yolo12n | 640 | mean_P_W | secondary | B3LogLinear | 10.03 | 9.535 | 10.73 | 9.898 | 1.4% | yes | no | 2.872 |
| yolo12n | 320 | mean_P_W | secondary | B3LogLinear | 9.74 | 9.256 | 10.42 | 9.418 | 3.4% | yes | no | 2.872 |
| yolov8n | 480 | mean_P_W | secondary | B3LogLinear | 10.02 | 9.52 | 10.71 | 10.13 | 1.1% | yes | no | 0 |
| resnet18 | 224 | mean_P_W | secondary | B3LogLinear | 10.26 | 9.754 | 10.98 | 10.85 | 5.4% | yes | no | 0 |
| mbv3 | 224 | mean_P_W | secondary | B3LogLinear | 9.202 | 8.744 | 9.842 | 9.059 | 1.6% | yes | no | 0 |

## Deployment calculator (serve phase)

Frozen Mode A AUTO-final predictions with the measured serve duration and request count; idle power from the training cells. Skipped: {'no_serve': 0, 'no_idle': 0, 'no_prediction': 0}.

| new_model | n | median_ape | inside_90 |
|---|---|---|---|
| no | 9 | 0.9% | 100.0% |
| yes | 6 | 2.9% | 100.0% |

| cell_id | model | imgsz | n_images | duration_s | measured_J | predicted_J | lo90_J | hi90_J | ape | inside | flagged | new_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lockbox_mbv3_224_t4_2400_r00 | mbv3 | 224 | 10 | 2.319 | 12.79 | 12.84 | 11.1 | 17.17 | 0.4% | yes | no | no |
| lockbox_mbv3_224_t4_2400_r01 | mbv3 | 224 | 10 | 2.33 | 12.78 | 12.89 | 11.15 | 17.22 | 0.9% | yes | no | no |
| lockbox_mbv3_224_t4_2400_r02 | mbv3 | 224 | 10 | 2.32 | 12.7 | 12.84 | 11.1 | 17.17 | 1.2% | yes | no | no |
| lockbox_resnet18_224_t4_2400_r00 | resnet18 | 224 | 10 | 2.789 | 18.65 | 18.52 | 13.43 | 31.16 | 0.7% | yes | no | no |
| lockbox_resnet18_224_t4_2400_r01 | resnet18 | 224 | 10 | 2.79 | 18.51 | 18.53 | 13.44 | 31.17 | 0.1% | yes | no | no |
| lockbox_resnet18_224_t4_2400_r02 | resnet18 | 224 | 10 | 2.789 | 18.42 | 18.52 | 13.43 | 31.17 | 0.6% | yes | no | no |
| lockbox_yolo12n_320_t4_2400_r00 | yolo12n | 320 | 10 | 3.536 | 23.57 | 23.35 | 16.4 | 40.6 | 0.9% | yes | yes | yes |
| lockbox_yolo12n_320_t4_2400_r01 | yolo12n | 320 | 10 | 3.527 | 23.24 | 23.31 | 16.36 | 40.55 | 0.3% | yes | yes | yes |
| lockbox_yolo12n_320_t4_2400_r02 | yolo12n | 320 | 10 | 3.515 | 23 | 23.24 | 16.29 | 40.49 | 1.1% | yes | yes | yes |
| lockbox_yolo12n_640_t4_2400_r00 | yolo12n | 640 | 10 | 6.001 | 48.57 | 46.06 | 27.12 | 93.07 | 5.2% | yes | yes | yes |
| lockbox_yolo12n_640_t4_2400_r01 | yolo12n | 640 | 10 | 6.007 | 48.48 | 46.09 | 27.15 | 93.1 | 4.9% | yes | yes | yes |
| lockbox_yolo12n_640_t4_2400_r02 | yolo12n | 640 | 10 | 6.014 | 48.47 | 46.13 | 27.19 | 93.14 | 4.8% | yes | yes | yes |
| lockbox_yolov8n_480_t4_2400_r00 | yolov8n | 480 | 10 | 3.649 | 26.53 | 27.14 | 17 | 52.32 | 2.3% | yes | no | no |
| lockbox_yolov8n_480_t4_2400_r01 | yolov8n | 480 | 10 | 3.641 | 26.41 | 27.11 | 16.96 | 52.28 | 2.7% | yes | no | no |
| lockbox_yolov8n_480_t4_2400_r02 | yolov8n | 480 | 10 | 3.638 | 26.4 | 27.09 | 16.95 | 52.27 | 2.6% | yes | no | no |

## Mode L

Whole-cell APE (`median_ape`), measured-segment APE, 1 s and 10 s blocks, medians per frozen rung. Rungs were trained on YOLO v8/11 cells only; resnet18 and mbv3 are non-YOLO extrapolations and are reported separately. Exploratory rungs are marked.

| family | rung | median_ape | median_ape_meas | median_ape_1s | median_ape_10s | n_cells | exploratory |
|---|---|---|---|---|---|---|---|
| non-YOLO | L0 | 4.5% | 9.0% | 8.3% | 6.3% | 6 | no |
| YOLO family | L0 | 3.7% | 1.8% | 5.7% | 2.8% | 9 | no |
| non-YOLO | L1 | 6.0% | 16.0% | 20.0% | 9.0% | 6 | no |
| YOLO family | L1 | 5.4% | 18.4% | 21.8% | 4.4% | 9 | no |
| non-YOLO | L1-cell | 3.9% | 11.8% | 17.6% | 9.0% | 6 | yes |
| YOLO family | L1-cell | 5.7% | 28.1% | 26.8% | 9.5% | 9 | yes |
| non-YOLO | L2 | 5.8% | 7.5% | 6.2% | 6.6% | 6 | no |
| YOLO family | L2 | 1.3% | 1.4% | 1.6% | 1.2% | 9 | no |
| non-YOLO | L3-ridge | 4.8% | 6.5% | 4.6% | 6.0% | 6 | no |
| YOLO family | L3-ridge | 2.3% | 1.7% | 3.4% | 2.8% | 9 | no |
| non-YOLO | L3-trees | 4.9% | 7.4% | 5.6% | 4.8% | 6 | no |
| YOLO family | L3-trees | 0.4% | 1.6% | 1.0% | 0.9% | 9 | no |
| non-YOLO | L4-ridge | 4.8% | 6.5% | 4.5% | 5.6% | 6 | no |
| YOLO family | L4-ridge | 2.3% | 1.7% | 3.8% | 2.7% | 9 | no |
| non-YOLO | L4-trees | 4.9% | 7.3% | 5.7% | 4.9% | 6 | no |
| YOLO family | L4-trees | 0.4% | 1.8% | 1.0% | 1.0% | 9 | no |

### Per cell

| cell_id | rung | model | imgsz | E_true_J | E_pred_J | ape | ape_meas | ape_1s | ape_10s |
|---|---|---|---|---|---|---|---|---|---|
| lockbox_mbv3_224_t4_2400_r00 | L0 | mbv3 | 224 | 2343 | 2446 | 4.4% | 10.0% | 8.8% | 5.0% |
| lockbox_mbv3_224_t4_2400_r01 | L0 | mbv3 | 224 | 233 | 256.6 | 10.1% | 10.8% | 10.6% | 10.8% |
| lockbox_mbv3_224_t4_2400_r02 | L0 | mbv3 | 224 | 232.7 | 255.9 | 10.0% | 10.7% | 10.6% | 10.6% |
| lockbox_resnet18_224_t4_2400_r00 | L0 | resnet18 | 224 | 929.1 | 934.7 | 0.6% | 8.0% | 7.7% | 4.7% |
| lockbox_resnet18_224_t4_2400_r01 | L0 | resnet18 | 224 | 321.4 | 306.6 | 4.6% | 7.5% | 7.9% | 6.4% |
| lockbox_resnet18_224_t4_2400_r02 | L0 | resnet18 | 224 | 319.8 | 306.5 | 4.2% | 7.1% | 7.4% | 6.2% |
| lockbox_yolo12n_320_t4_2400_r00 | L0 | yolo12n | 320 | 2779 | 2902 | 4.4% | 3.4% | 6.4% | 4.9% |
| lockbox_yolo12n_320_t4_2400_r01 | L0 | yolo12n | 320 | 113.4 | 120.9 | 6.7% | 3.9% | 7.5% | 5.9% |
| lockbox_yolo12n_320_t4_2400_r02 | L0 | yolo12n | 320 | 112.8 | 121.2 | 7.4% | 4.5% | 7.8% | 6.7% |
| lockbox_yolo12n_640_t4_2400_r00 | L0 | yolo12n | 640 | 2939 | 3054 | 3.9% | 0.4% | 5.7% | 3.7% |
| lockbox_yolo12n_640_t4_2400_r01 | L0 | yolo12n | 640 | 215 | 221.2 | 2.9% | 0.3% | 2.8% | 1.9% |
| lockbox_yolo12n_640_t4_2400_r02 | L0 | yolo12n | 640 | 214 | 221.3 | 3.4% | 1.0% | 2.4% | 2.6% |
| lockbox_yolov8n_480_t4_2400_r00 | L0 | yolov8n | 480 | 2031 | 2107 | 3.7% | 2.1% | 5.8% | 2.8% |
| lockbox_yolov8n_480_t4_2400_r01 | L0 | yolov8n | 480 | 129.4 | 134.1 | 3.6% | 1.1% | 5.0% | 2.0% |
| lockbox_yolov8n_480_t4_2400_r02 | L0 | yolov8n | 480 | 128.8 | 133.1 | 3.3% | 1.8% | 4.8% | 1.3% |
| lockbox_mbv3_224_t4_2400_r00 | L1 | mbv3 | 224 | 2343 | 2507 | 7.0% | 12.4% | 19.7% | 9.3% |
| lockbox_mbv3_224_t4_2400_r01 | L1 | mbv3 | 224 | 233 | 234.9 | 0.8% | 13.1% | 13.2% | 8.8% |
| lockbox_mbv3_224_t4_2400_r02 | L1 | mbv3 | 224 | 232.7 | 225.7 | 3.0% | 12.4% | 13.5% | 7.0% |
| lockbox_resnet18_224_t4_2400_r00 | L1 | resnet18 | 224 | 929.1 | 881.6 | 5.1% | 21.9% | 20.2% | 5.9% |
| lockbox_resnet18_224_t4_2400_r01 | L1 | resnet18 | 224 | 321.4 | 294.4 | 8.4% | 19.0% | 21.6% | 13.7% |
| lockbox_resnet18_224_t4_2400_r02 | L1 | resnet18 | 224 | 319.8 | 264.7 | 17.2% | 28.9% | 27.4% | 23.5% |
| lockbox_yolo12n_320_t4_2400_r00 | L1 | yolo12n | 320 | 2779 | 2776 | 0.1% | 18.8% | 21.0% | 4.4% |
| lockbox_yolo12n_320_t4_2400_r01 | L1 | yolo12n | 320 | 113.4 | 128 | 12.9% | 18.4% | 26.7% | 1.5% |
| lockbox_yolo12n_320_t4_2400_r02 | L1 | yolo12n | 320 | 112.8 | 130 | 15.2% | 17.8% | 29.5% | 0.8% |
| lockbox_yolo12n_640_t4_2400_r00 | L1 | yolo12n | 640 | 2939 | 2878 | 2.0% | 17.9% | 18.3% | 3.0% |
| lockbox_yolo12n_640_t4_2400_r01 | L1 | yolo12n | 640 | 215 | 240.3 | 11.8% | 8.7% | 21.8% | 3.1% |
| lockbox_yolo12n_640_t4_2400_r02 | L1 | yolo12n | 640 | 214 | 217.4 | 1.6% | 14.1% | 14.2% | 5.2% |
| lockbox_yolov8n_480_t4_2400_r00 | L1 | yolov8n | 480 | 2031 | 1993 | 1.9% | 21.8% | 22.2% | 6.4% |
| lockbox_yolov8n_480_t4_2400_r01 | L1 | yolov8n | 480 | 129.4 | 136.4 | 5.4% | 22.1% | 21.8% | 8.1% |
| lockbox_yolov8n_480_t4_2400_r02 | L1 | yolov8n | 480 | 128.8 | 141.1 | 9.5% | 20.5% | 25.1% | 6.6% |
| lockbox_mbv3_224_t4_2400_r00 | L1-cell | mbv3 | 224 | 2343 | 2431 | 3.7% | 10.5% | 17.5% | 5.9% |
| lockbox_mbv3_224_t4_2400_r01 | L1-cell | mbv3 | 224 | 233 | 237.5 | 1.9% | 12.7% | 12.6% | 12.3% |
| lockbox_mbv3_224_t4_2400_r02 | L1-cell | mbv3 | 224 | 232.7 | 218.6 | 6.1% | 11.0% | 10.2% | 8.5% |
| lockbox_resnet18_224_t4_2400_r00 | L1-cell | resnet18 | 224 | 929.1 | 891.2 | 4.1% | 14.8% | 17.7% | 9.6% |
| lockbox_resnet18_224_t4_2400_r01 | L1-cell | resnet18 | 224 | 321.4 | 322.2 | 0.2% | 9.1% | 20.0% | 5.9% |
| lockbox_resnet18_224_t4_2400_r02 | L1-cell | resnet18 | 224 | 319.8 | 259.4 | 18.9% | 30.3% | 26.2% | 25.4% |
| lockbox_yolo12n_320_t4_2400_r00 | L1-cell | yolo12n | 320 | 2779 | 2719 | 2.2% | 30.4% | 24.0% | 8.5% |
| lockbox_yolo12n_320_t4_2400_r01 | L1-cell | yolo12n | 320 | 113.4 | 124 | 9.4% | 30.3% | 33.5% | 12.4% |
| lockbox_yolo12n_320_t4_2400_r02 | L1-cell | yolo12n | 320 | 112.8 | 127.6 | 13.1% | 29.7% | 38.1% | 11.7% |
| lockbox_yolo12n_640_t4_2400_r00 | L1-cell | yolo12n | 640 | 2939 | 2863 | 2.6% | 18.4% | 20.7% | 7.3% |
| lockbox_yolo12n_640_t4_2400_r01 | L1-cell | yolo12n | 640 | 215 | 274.3 | 27.6% | 1.1% | 39.9% | 9.5% |
| lockbox_yolo12n_640_t4_2400_r02 | L1-cell | yolo12n | 640 | 214 | 226.2 | 5.7% | 10.9% | 11.2% | 5.0% |
| lockbox_yolov8n_480_t4_2400_r00 | L1-cell | yolov8n | 480 | 2031 | 1972 | 2.9% | 28.1% | 26.8% | 8.3% |
| lockbox_yolov8n_480_t4_2400_r01 | L1-cell | yolov8n | 480 | 129.4 | 132.5 | 2.4% | 29.5% | 17.3% | 18.0% |
| lockbox_yolov8n_480_t4_2400_r02 | L1-cell | yolov8n | 480 | 128.8 | 143.1 | 11.1% | 26.1% | 38.4% | 14.0% |
| lockbox_mbv3_224_t4_2400_r00 | L2 | mbv3 | 224 | 2343 | 2422 | 3.4% | 7.9% | 2.6% | 3.6% |
| lockbox_mbv3_224_t4_2400_r01 | L2 | mbv3 | 224 | 233 | 247.3 | 6.1% | 8.7% | 8.3% | 7.3% |
| lockbox_mbv3_224_t4_2400_r02 | L2 | mbv3 | 224 | 232.7 | 246.7 | 6.0% | 8.5% | 8.2% | 7.2% |
| lockbox_resnet18_224_t4_2400_r00 | L2 | resnet18 | 224 | 929.1 | 917.2 | 1.3% | 7.2% | 4.4% | 2.4% |
| lockbox_resnet18_224_t4_2400_r01 | L2 | resnet18 | 224 | 321.4 | 301.9 | 6.1% | 6.8% | 6.6% | 6.9% |
| lockbox_resnet18_224_t4_2400_r02 | L2 | resnet18 | 224 | 319.8 | 302.2 | 5.5% | 6.2% | 5.7% | 6.3% |
| lockbox_yolo12n_320_t4_2400_r00 | L2 | yolo12n | 320 | 2779 | 2854 | 2.7% | 3.0% | 3.1% | 2.9% |
| lockbox_yolo12n_320_t4_2400_r01 | L2 | yolo12n | 320 | 113.4 | 114.8 | 1.3% | 4.2% | 2.0% | 2.1% |
| lockbox_yolo12n_320_t4_2400_r02 | L2 | yolo12n | 320 | 112.8 | 115 | 1.9% | 4.7% | 1.6% | 2.7% |
| lockbox_yolo12n_640_t4_2400_r00 | L2 | yolo12n | 640 | 2939 | 2997 | 2.0% | 1.1% | 2.8% | 2.5% |
| lockbox_yolo12n_640_t4_2400_r01 | L2 | yolo12n | 640 | 215 | 215.3 | 0.2% | 1.0% | 1.4% | 0.7% |
| lockbox_yolo12n_640_t4_2400_r02 | L2 | yolo12n | 640 | 214 | 215.5 | 0.7% | 1.6% | 1.4% | 1.1% |
| lockbox_yolov8n_480_t4_2400_r00 | L2 | yolov8n | 480 | 2031 | 2066 | 1.7% | 1.4% | 2.9% | 1.2% |
| lockbox_yolov8n_480_t4_2400_r01 | L2 | yolov8n | 480 | 129.4 | 127.8 | 1.3% | 0.6% | 1.3% | 0.8% |
| lockbox_yolov8n_480_t4_2400_r02 | L2 | yolov8n | 480 | 128.8 | 127.1 | 1.3% | 0.8% | 1.6% | 1.0% |
| lockbox_mbv3_224_t4_2400_r00 | L3-ridge | mbv3 | 224 | 2343 | 2491 | 6.3% | 9.4% | 5.0% | 6.4% |
| lockbox_mbv3_224_t4_2400_r01 | L3-ridge | mbv3 | 224 | 233 | 249.2 | 6.9% | 8.9% | 8.6% | 7.4% |
| lockbox_mbv3_224_t4_2400_r02 | L3-ridge | mbv3 | 224 | 232.7 | 248.5 | 6.8% | 8.7% | 8.3% | 7.2% |
| lockbox_resnet18_224_t4_2400_r00 | L3-ridge | resnet18 | 224 | 929.1 | 958.7 | 3.2% | 3.8% | 4.1% | 5.5% |
| lockbox_resnet18_224_t4_2400_r01 | L3-ridge | resnet18 | 224 | 321.4 | 311.4 | 3.1% | 4.2% | 3.8% | 3.9% |
| lockbox_resnet18_224_t4_2400_r02 | L3-ridge | resnet18 | 224 | 319.8 | 310 | 3.1% | 4.2% | 3.8% | 4.0% |
| lockbox_yolo12n_320_t4_2400_r00 | L3-ridge | yolo12n | 320 | 2779 | 2996 | 7.8% | 6.2% | 7.2% | 8.3% |
| lockbox_yolo12n_320_t4_2400_r01 | L3-ridge | yolo12n | 320 | 113.4 | 117 | 3.2% | 5.3% | 4.1% | 4.0% |
| lockbox_yolo12n_320_t4_2400_r02 | L3-ridge | yolo12n | 320 | 112.8 | 115.4 | 2.3% | 4.7% | 3.4% | 2.8% |
| lockbox_yolo12n_640_t4_2400_r00 | L3-ridge | yolo12n | 640 | 2939 | 3097 | 5.4% | 2.1% | 5.6% | 6.0% |
| lockbox_yolo12n_640_t4_2400_r01 | L3-ridge | yolo12n | 640 | 215 | 217.6 | 1.3% | 1.1% | 1.9% | 1.4% |
| lockbox_yolo12n_640_t4_2400_r02 | L3-ridge | yolo12n | 640 | 214 | 218.3 | 2.0% | 1.7% | 1.9% | 2.1% |
| lockbox_yolov8n_480_t4_2400_r00 | L3-ridge | yolov8n | 480 | 2031 | 2142 | 5.4% | 0.6% | 5.3% | 6.3% |
| lockbox_yolov8n_480_t4_2400_r01 | L3-ridge | yolov8n | 480 | 129.4 | 129.2 | 0.2% | 0.6% | 1.1% | 0.5% |
| lockbox_yolov8n_480_t4_2400_r02 | L3-ridge | yolov8n | 480 | 128.8 | 129 | 0.1% | 0.6% | 1.4% | 0.4% |
| lockbox_mbv3_224_t4_2400_r00 | L3-trees | mbv3 | 224 | 2343 | 2351 | 0.4% | 8.4% | 1.2% | 0.8% |
| lockbox_mbv3_224_t4_2400_r01 | L3-trees | mbv3 | 224 | 233 | 249 | 6.8% | 8.8% | 8.7% | 7.7% |
| lockbox_mbv3_224_t4_2400_r02 | L3-trees | mbv3 | 224 | 232.7 | 248.2 | 6.7% | 8.6% | 8.5% | 7.4% |
| lockbox_resnet18_224_t4_2400_r00 | L3-trees | resnet18 | 224 | 929.1 | 914.7 | 1.5% | 5.5% | 1.6% | 1.5% |
| lockbox_resnet18_224_t4_2400_r01 | L3-trees | resnet18 | 224 | 321.4 | 306.1 | 4.8% | 5.9% | 5.3% | 4.8% |
| lockbox_resnet18_224_t4_2400_r02 | L3-trees | resnet18 | 224 | 319.8 | 304 | 4.9% | 6.3% | 5.9% | 4.8% |
| lockbox_yolo12n_320_t4_2400_r00 | L3-trees | yolo12n | 320 | 2779 | 2783 | 0.1% | 2.0% | 1.0% | 0.9% |
| lockbox_yolo12n_320_t4_2400_r01 | L3-trees | yolo12n | 320 | 113.4 | 115 | 1.4% | 2.2% | 1.4% | 1.6% |
| lockbox_yolo12n_320_t4_2400_r02 | L3-trees | yolo12n | 320 | 112.8 | 114.9 | 1.9% | 2.6% | 1.8% | 2.0% |
| lockbox_yolo12n_640_t4_2400_r00 | L3-trees | yolo12n | 640 | 2939 | 2951 | 0.4% | 1.6% | 0.9% | 0.6% |
| lockbox_yolo12n_640_t4_2400_r01 | L3-trees | yolo12n | 640 | 215 | 217.1 | 1.0% | 1.1% | 1.6% | 0.9% |
| lockbox_yolo12n_640_t4_2400_r02 | L3-trees | yolo12n | 640 | 214 | 217.1 | 1.4% | 1.8% | 1.4% | 1.6% |
| lockbox_yolov8n_480_t4_2400_r00 | L3-trees | yolov8n | 480 | 2031 | 2034 | 0.1% | 1.4% | 1.0% | 1.0% |
| lockbox_yolov8n_480_t4_2400_r01 | L3-trees | yolov8n | 480 | 129.4 | 129 | 0.3% | 0.8% | 0.7% | 0.3% |
| lockbox_yolov8n_480_t4_2400_r02 | L3-trees | yolov8n | 480 | 128.8 | 128.5 | 0.3% | 0.9% | 0.7% | 0.5% |
| lockbox_mbv3_224_t4_2400_r00 | L4-ridge | mbv3 | 224 | 2343 | 2488 | 6.2% | 9.2% | 5.0% | 6.1% |
| lockbox_mbv3_224_t4_2400_r01 | L4-ridge | mbv3 | 224 | 233 | 249.1 | 6.9% | 8.7% | 8.2% | 7.2% |
| lockbox_mbv3_224_t4_2400_r02 | L4-ridge | mbv3 | 224 | 232.7 | 248 | 6.6% | 8.6% | 7.8% | 7.0% |
| lockbox_resnet18_224_t4_2400_r00 | L4-ridge | resnet18 | 224 | 929.1 | 956.6 | 3.0% | 3.9% | 3.9% | 5.0% |
| lockbox_resnet18_224_t4_2400_r01 | L4-ridge | resnet18 | 224 | 321.4 | 311.3 | 3.1% | 4.2% | 3.7% | 3.8% |
| lockbox_resnet18_224_t4_2400_r02 | L4-ridge | resnet18 | 224 | 319.8 | 309.2 | 3.3% | 4.5% | 4.0% | 4.2% |
| lockbox_yolo12n_320_t4_2400_r00 | L4-ridge | yolo12n | 320 | 2779 | 2989 | 7.6% | 5.8% | 7.1% | 8.2% |
| lockbox_yolo12n_320_t4_2400_r01 | L4-ridge | yolo12n | 320 | 113.4 | 117 | 3.2% | 5.1% | 4.2% | 3.8% |
| lockbox_yolo12n_320_t4_2400_r02 | L4-ridge | yolo12n | 320 | 112.8 | 115.5 | 2.3% | 4.5% | 3.8% | 2.7% |
| lockbox_yolo12n_640_t4_2400_r00 | L4-ridge | yolo12n | 640 | 2939 | 3092 | 5.2% | 2.0% | 5.3% | 5.6% |
| lockbox_yolo12n_640_t4_2400_r01 | L4-ridge | yolo12n | 640 | 215 | 218.4 | 1.6% | 1.3% | 2.0% | 1.4% |
| lockbox_yolo12n_640_t4_2400_r02 | L4-ridge | yolo12n | 640 | 214 | 218.4 | 2.1% | 1.7% | 2.0% | 2.2% |
| lockbox_yolov8n_480_t4_2400_r00 | L4-ridge | yolov8n | 480 | 2031 | 2138 | 5.2% | 0.8% | 5.2% | 6.0% |
| lockbox_yolov8n_480_t4_2400_r01 | L4-ridge | yolov8n | 480 | 129.4 | 129.1 | 0.2% | 0.8% | 1.3% | 0.6% |
| lockbox_yolov8n_480_t4_2400_r02 | L4-ridge | yolov8n | 480 | 128.8 | 129 | 0.1% | 0.7% | 1.7% | 0.5% |
| lockbox_mbv3_224_t4_2400_r00 | L4-trees | mbv3 | 224 | 2343 | 2353 | 0.4% | 8.3% | 1.2% | 0.7% |
| lockbox_mbv3_224_t4_2400_r01 | L4-trees | mbv3 | 224 | 233 | 248.7 | 6.7% | 8.7% | 8.5% | 7.5% |
| lockbox_mbv3_224_t4_2400_r02 | L4-trees | mbv3 | 224 | 232.7 | 248 | 6.6% | 8.4% | 8.4% | 7.3% |
| lockbox_resnet18_224_t4_2400_r00 | L4-trees | resnet18 | 224 | 929.1 | 914.2 | 1.6% | 5.6% | 1.5% | 1.6% |
| lockbox_resnet18_224_t4_2400_r01 | L4-trees | resnet18 | 224 | 321.4 | 305.9 | 4.8% | 5.9% | 5.5% | 4.9% |
| lockbox_resnet18_224_t4_2400_r02 | L4-trees | resnet18 | 224 | 319.8 | 303.7 | 5.1% | 6.4% | 6.0% | 4.8% |
| lockbox_yolo12n_320_t4_2400_r00 | L4-trees | yolo12n | 320 | 2779 | 2782 | 0.1% | 2.1% | 1.0% | 0.8% |
| lockbox_yolo12n_320_t4_2400_r01 | L4-trees | yolo12n | 320 | 113.4 | 115.2 | 1.6% | 2.3% | 1.4% | 1.7% |
| lockbox_yolo12n_320_t4_2400_r02 | L4-trees | yolo12n | 320 | 112.8 | 114.9 | 1.8% | 2.4% | 1.8% | 1.9% |
| lockbox_yolo12n_640_t4_2400_r00 | L4-trees | yolo12n | 640 | 2939 | 2949 | 0.4% | 1.6% | 0.9% | 0.5% |
| lockbox_yolo12n_640_t4_2400_r01 | L4-trees | yolo12n | 640 | 215 | 218 | 1.4% | 1.8% | 1.7% | 1.4% |
| lockbox_yolo12n_640_t4_2400_r02 | L4-trees | yolo12n | 640 | 214 | 217.1 | 1.5% | 1.8% | 1.4% | 1.6% |
| lockbox_yolov8n_480_t4_2400_r00 | L4-trees | yolov8n | 480 | 2031 | 2034 | 0.1% | 1.3% | 0.9% | 1.0% |
| lockbox_yolov8n_480_t4_2400_r01 | L4-trees | yolov8n | 480 | 129.4 | 129 | 0.3% | 0.8% | 0.6% | 0.4% |
| lockbox_yolov8n_480_t4_2400_r02 | L4-trees | yolov8n | 480 | 128.8 | 128.6 | 0.1% | 0.8% | 0.6% | 0.4% |

## Probe

Mean predicted power over a measured-phase run of cell j times j's latency, against the energy per image of cell k (ordered pairs of one configuration), per rung, against the frozen Mode A primary prediction.

| rung | n_pairs | median_probe_ape | median_mode_a_ape |
|---|---|---|---|
| L0 | 30 | 8.0% | 7.5% |
| L1 | 30 | 21.9% | 7.5% |
| L1-cell | 30 | 22.3% | 7.5% |
| L2 | 30 | 7.3% | 7.5% |
| L3-ridge | 30 | 5.9% | 7.5% |
| L3-trees | 30 | 8.1% | 7.5% |
| L4-ridge | 30 | 6.1% | 7.5% |
| L4-trees | 30 | 8.0% | 7.5% |

