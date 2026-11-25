# Lockbox 3 (R4), scored once

**confirmatory: frozen models, frozen predictions, unchanged code.** Predictions frozen at 2026-11-25 14:58:20 EST (`lockbox3_predictions.csv` sha256 `be2d9805a6562b86`), before any lockbox-3 cell. Configurations: mbv2:224@2400 (similar to mbv3 / mbv3l), shufv2:224@2000 (similar to mbv3), resnet50:224@1500 (similar to resnet18 / resnet34), effb2:224@2400 (similar to effb0 / effb1), vitb32:224@2000 (similar to vitb16), convnexts:224@2400 (similar to convnextt); hooks off, 4 threads, fp32. Interval of record: base-model conformal (`inside_90`); R3-style beside it (`inside_90_r3`). Percent columns are percent.

## Truth

| base_model | imgsz | config_key | n_cells | energy_mJ | latency_ms | mean_P_W |
|---|---|---|---|---|---|---|
| convnexts | 224 | convnexts|convnexts|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 3717 | 314.6 | 11.82 |
| effb2 | 224 | effb2|effb2|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 1237 | 127 | 9.741 |
| mbv2 | 224 | mbv2|mbv2|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 660 | 70.77 | 9.325 |
| resnet50 | 224 | resnet50|resnet50|fp32|224|1|4|1500|nov2026|hooks=off | 3 | 1725 | 281 | 6.138 |
| shufv2 | 224 | shufv2|shufv2|fp32|224|1|4|2000|nov2026|hooks=off | 3 | 285.6 | 36.08 | 7.917 |
| vitb32 | 224 | vitb32|vitb32|fp32|224|1|4|2000|nov2026|hooks=off | 3 | 2351 | 270.8 | 8.682 |

## Mode A, every candidate (AUTO is the primary)

| target | candidate | n | mdape | mape | maxape | inside_90 | width | inside_90_r3 |
|---|---|---|---|---|---|---|---|---|
| energy_mJ | B0Median | 6 | 89.6 | 232.0 | 856.2 | 83.3 | 65.9 | 83.3 |
| energy_mJ | B1MacsProportional | 6 | 62.2 | 52.2 | 71.8 | 100.0 | 12.61 | 100.0 |
| energy_mJ | B2NearestModel | 6 | 28.4 | 37.6 | 78.6 | 83.3 | 3.412 | 66.7 |
| energy_mJ | B3LogLinear | 6 | 24.3 | 21.7 | 31.5 | 100.0 | 3.641 | 100.0 |
| energy_mJ | M1Ridge | 6 | 6.9 | 7.8 | 17.4 | 66.7 | 1.251 | 66.7 |
| energy_mJ | M2Trees | 6 | 3.7 | 14.7 | 49.2 | 83.3 | 3.039 | 83.3 |
| energy_mJ | M3Factored | 6 | 6.9 | 7.8 | 17.4 | 66.7 | 1.251 | 66.7 |
| energy_mJ | M4Physics | 6 | 6.4 | 10.1 | 34.2 | 83.3 | 1.789 | 83.3 |
| energy_mJ | M5Pooled | 6 | 9.9 | 13.2 | 37.3 | 83.3 | 1.78 | 83.3 |
| energy_mJ | AUTO | 6 | 6.9 | 7.8 | 17.4 | 66.7 | 1.251 | 66.7 |
| latency_ms | B0Median | 6 | 82.4 | 216.2 | 774.1 | 100.0 | 86.19 | 83.3 |
| latency_ms | B1MacsProportional | 6 | 65.4 | 56.9 | 94.3 | 100.0 | 14.36 | 100.0 |
| latency_ms | B2NearestModel | 6 | 33.2 | 39.5 | 81.0 | 83.3 | 3.784 | 83.3 |
| latency_ms | B3LogLinear | 6 | 24.5 | 22.5 | 35.5 | 100.0 | 4.111 | 100.0 |
| latency_ms | M1Ridge | 6 | 8.0 | 12.8 | 27.4 | 66.7 | 1.359 | 66.7 |
| latency_ms | M2Trees | 6 | 11.5 | 17.2 | 50.4 | 83.3 | 4.051 | 83.3 |
| latency_ms | M4Physics | 6 | 5.9 | 11.5 | 41.6 | 83.3 | 1.802 | 83.3 |
| latency_ms | M5Pooled | 6 | 8.3 | 14.2 | 44.9 | 83.3 | 1.758 | 83.3 |
| latency_ms | AUTO | 6 | 8.0 | 12.8 | 27.4 | 66.7 | 1.359 | 66.7 |
| mean_P_W | B0Median | 6 | 14.5 | 18.1 | 54.9 | 100.0 | 2.482 | 83.3 |
| mean_P_W | B1MacsProportional | 6 | 6.0 | 6.2 | 14.5 | 100.0 | 1.389 | 83.3 |
| mean_P_W | B2NearestModel | 6 | 1.3 | 5.4 | 16.5 | 83.3 | 1.307 | 83.3 |
| mean_P_W | B3LogLinear | 6 | 4.9 | 4.8 | 6.5 | 100.0 | 1.233 | 100.0 |
| mean_P_W | M1Ridge | 6 | 3.1 | 4.8 | 12.1 | 66.7 | 1.155 | 50.0 |
| mean_P_W | M2Trees | 6 | 3.7 | 4.4 | 12.7 | 83.3 | 1.2 | 83.3 |
| mean_P_W | M4Physics | 6 | 2.0 | 3.5 | 12.6 | 83.3 | 1.144 | 83.3 |
| mean_P_W | M5Pooled | 6 | 1.6 | 3.6 | 13.8 | 83.3 | 1.143 | 83.3 |
| mean_P_W | AUTO | 6 | 2.0 | 3.5 | 12.6 | 83.3 | 1.144 | 83.3 |

## Mode A per configuration (AUTO)

| model | imgsz | target | pred | lo90 | hi90 | truth | ape | inside | novelty | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| mbv2 | 224 | energy_mJ | 648.9 | 580.1 | 725.8 | 660 | 1.7 | True | 2.327 | False |
| shufv2 | 224 | energy_mJ | 319.7 | 285.8 | 357.6 | 285.6 | 11.9 | False | 14.73 | True |
| resnet50 | 224 | energy_mJ | 1576 | 1409 | 1763 | 1725 | 8.6 | True | 0.5 | False |
| effb2 | 224 | energy_mJ | 1262 | 1128 | 1411 | 1237 | 2.0 | True | 0.4147 | False |
| vitb32 | 224 | energy_mJ | 1941 | 1735 | 2171 | 2351 | 17.4 | False | 0.8513 | False |
| convnexts | 224 | energy_mJ | 3907 | 3493 | 4370 | 3717 | 5.1 | True | 0.8531 | False |
| mbv2 | 224 | latency_ms | 66.07 | 56.68 | 77.03 | 70.77 | 6.6 | True | 2.327 | False |
| shufv2 | 224 | latency_ms | 45.96 | 39.42 | 53.58 | 36.08 | 27.4 | False | 14.73 | True |
| resnet50 | 224 | latency_ms | 254.8 | 218.6 | 297.1 | 281 | 9.3 | True | 0.5 | False |
| effb2 | 224 | latency_ms | 130.8 | 112.2 | 152.5 | 127 | 3.0 | True | 0.4147 | False |
| vitb32 | 224 | latency_ms | 204.8 | 175.7 | 238.7 | 270.8 | 24.4 | False | 0.8513 | False |
| convnexts | 224 | latency_ms | 332.6 | 285.3 | 387.8 | 314.6 | 5.8 | True | 0.8531 | False |
| mbv2 | 224 | mean_P_W | 9.399 | 8.788 | 10.05 | 9.325 | 0.8 | True | 2.327 | False |
| shufv2 | 224 | mean_P_W | 8.096 | 7.57 | 8.659 | 7.917 | 2.3 | True | 14.73 | True |
| resnet50 | 224 | mean_P_W | 6.239 | 5.834 | 6.673 | 6.138 | 1.7 | True | 0.5 | False |
| effb2 | 224 | mean_P_W | 9.609 | 8.985 | 10.28 | 9.741 | 1.3 | True | 0.4147 | False |
| vitb32 | 224 | mean_P_W | 9.779 | 9.143 | 10.46 | 8.682 | 12.6 | False | 0.8513 | False |
| convnexts | 224 | mean_P_W | 11.51 | 10.76 | 12.31 | 11.82 | 2.6 | True | 0.8531 | False |

## Deployment calculator (serve phase, predicted duration)

| cell_id | measured_J | predicted_J_oh0 | ape_oh0 | predicted_J_est | ape_est |
|---|---|---|---|---|---|
| lockbox3_convnexts_224_t4_2400_r00 | 47.31 | 48.88 | 3.3 | 48.88 | 3.3 |
| lockbox3_convnexts_224_t4_2400_r01 | 46.8 | 48.88 | 4.4 | 48.88 | 4.4 |
| lockbox3_convnexts_224_t4_2400_r02 | 47.08 | 48.88 | 3.8 | 48.88 | 3.8 |
| lockbox3_effb2_224_t4_2400_r00 | 22.2 | 22.43 | 1.0 | 22.43 | 1.0 |
| lockbox3_effb2_224_t4_2400_r01 | 22.04 | 22.43 | 1.8 | 22.43 | 1.8 |
| lockbox3_effb2_224_t4_2400_r02 | 22 | 22.43 | 1.9 | 22.43 | 2.0 |
| lockbox3_mbv2_224_t4_2400_r00 | 16.28 | 16.3 | 0.1 | 16.3 | 0.1 |
| lockbox3_mbv2_224_t4_2400_r01 | 16.36 | 16.3 | 0.4 | 16.3 | 0.4 |
| lockbox3_mbv2_224_t4_2400_r02 | 16.32 | 16.3 | 0.1 | 16.3 | 0.1 |
| lockbox3_resnet50_224_t4_1500_r00 | 25.16 | 23.65 | 6.0 | 23.65 | 6.0 |
| lockbox3_resnet50_224_t4_1500_r01 | 24.87 | 23.65 | 4.9 | 23.65 | 4.9 |
| lockbox3_resnet50_224_t4_1500_r02 | 25.08 | 23.65 | 5.7 | 23.65 | 5.7 |
| lockbox3_shufv2_224_t4_2000_r00 | 12.19 | 12.23 | 0.3 | 12.23 | 0.3 |
| lockbox3_shufv2_224_t4_2000_r01 | 11.8 | 12.23 | 3.7 | 12.23 | 3.7 |
| lockbox3_shufv2_224_t4_2000_r02 | 11.9 | 12.23 | 2.8 | 12.23 | 2.8 |
| lockbox3_vitb32_224_t4_2000_r00 | 32.49 | 28.44 | 12.5 | 28.44 | 12.5 |
| lockbox3_vitb32_224_t4_2000_r01 | 32.56 | 28.44 | 12.6 | 28.44 | 12.6 |
| lockbox3_vitb32_224_t4_2000_r02 | 32.6 | 28.44 | 12.8 | 28.44 | 12.8 |

## Mode L (frozen rungs, trained on v4, median over cells)

| rung | ape | ape_meas | ape_1s | n_cells |
|---|---|---|---|---|
| L0 | 6.0 | 5.8 | 5.6 | 18 |
| L3-trees | 1.7 | 2.1 | 1.4 | 18 |
| L3k-trees | 2.2 | 2.7 | 1.7 | 18 |
| L5-cmos-trees | 1.8 | 2.5 | 1.5 | 18 |
| L5k-trees | 1.7 | 2.2 | 1.3 | 18 |
