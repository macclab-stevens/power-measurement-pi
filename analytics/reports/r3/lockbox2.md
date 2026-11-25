# Lockbox 2 (R3), scored once

**confirmatory: frozen models, frozen predictions, unchanged code.** Predictions frozen at 2026-11-24 23:43:29 EST (`lockbox2_predictions.csv` sha256 `49d2bdfb5dc27ba7`), before any lockbox-2 cell. Configurations: resnet34:224 (similar to resnet18), mbv3l:224 (similar to mbv3), effb1:224 (similar to effb0), swins:224 (similar to swint), yolo12m:640 (similar to yolo12s); hooks off, 2400 MHz, 4 threads, fp32. Percent columns are percent.

## Truth

| base_model | imgsz | config_key | n_cells | energy_mJ | latency_ms | mean_P_W |
|---|---|---|---|---|---|---|
| effb1 | 224 | effb1|effb1|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 1168 | 119.6 | 9.767 |
| mbv3l | 224 | mbv3l|mbv3l|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 547.3 | 57.09 | 9.586 |
| resnet34 | 224 | resnet34|resnet34|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 1467 | 130 | 11.28 |
| swins | 224 | swins|swins|fp32|224|1|4|2400|nov2026|hooks=off | 3 | 4390 | 387.1 | 11.34 |
| yolo12m | 640 | yolo12m|yolo12m|fp32|640|1|4|2400|nov2026|hooks=off | 3 | 2.012e+04 | 1849 | 10.88 |

## Mode A, every candidate (AUTO is the primary)

| target | candidate | n | mdape | mape | maxape | inside_90 | width |
|---|---|---|---|---|---|---|---|
| energy_mJ | B0Median | 5 | 86.4 | 148.5 | 398.4 | 80.0 | 55.77 |
| energy_mJ | B1MacsProportional | 5 | 59.8 | 52.0 | 97.0 | 80.0 | 11.2 |
| energy_mJ | B2NearestModel | 5 | 15.9 | 31.3 | 97.1 | 100.0 | 3.975 |
| energy_mJ | B3LogLinear | 5 | 24.5 | 36.6 | 92.0 | 100.0 | 5.2 |
| energy_mJ | M1Ridge | 5 | 13.6 | 12.1 | 16.9 | 60.0 | 1.587 |
| energy_mJ | M2Trees | 5 | 18.6 | 24.6 | 51.1 | 100.0 | 6.017 |
| energy_mJ | M3Factored | 5 | 16.1 | 16.0 | 29.2 | 60.0 | 1.63 |
| energy_mJ | M4Physics | 5 | 8.3 | 9.8 | 20.5 | 80.0 | 1.473 |
| energy_mJ | AUTO | 5 | 13.6 | 12.1 | 16.9 | 60.0 | 1.587 |
| latency_ms | B0Median | 5 | 143.5 | 172.7 | 454.4 | 80.0 | 49.77 |
| latency_ms | B1MacsProportional | 5 | 60.8 | 62.6 | 122.2 | 80.0 | 13.8 |
| latency_ms | B2NearestModel | 5 | 22.2 | 37.1 | 110.2 | 100.0 | 4.574 |
| latency_ms | B3LogLinear | 5 | 26.7 | 39.1 | 101.1 | 100.0 | 5.601 |
| latency_ms | M1Ridge | 5 | 16.8 | 15.2 | 22.4 | 40.0 | 1.594 |
| latency_ms | M2Trees | 5 | 25.5 | 22.5 | 44.7 | 100.0 | 6.497 |
| latency_ms | M4Physics | 5 | 8.9 | 9.7 | 19.5 | 100.0 | 1.542 |
| latency_ms | AUTO | 5 | 16.8 | 15.2 | 22.4 | 40.0 | 1.594 |
| mean_P_W | B0Median | 5 | 13.3 | 10.3 | 16.8 | 100.0 | 2.028 |
| mean_P_W | B1MacsProportional | 5 | 8.1 | 7.6 | 11.9 | 100.0 | 1.391 |
| mean_P_W | B2NearestModel | 5 | 3.7 | 3.3 | 5.8 | 100.0 | 1.322 |
| mean_P_W | B3LogLinear | 5 | 3.0 | 3.0 | 4.5 | 100.0 | 1.209 |
| mean_P_W | M1Ridge | 5 | 1.1 | 1.0 | 1.5 | 100.0 | 1.205 |
| mean_P_W | M2Trees | 5 | 1.8 | 3.3 | 6.7 | 100.0 | 1.217 |
| mean_P_W | M4Physics | 5 | 1.4 | 2.5 | 4.9 | 100.0 | 1.136 |
| mean_P_W | AUTO | 5 | 1.4 | 2.5 | 4.9 | 100.0 | 1.136 |

## Mode A per configuration (AUTO)

| model | imgsz | target | pred | lo90 | hi90 | truth | ape | inside | novelty | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet34 | 224 | energy_mJ | 1426 | 1263 | 2005 | 1467 | 2.8 | True | 1.834 | False |
| mbv3l | 224 | energy_mJ | 621.5 | 550.6 | 874 | 547.3 | 13.6 | False | 1.584 | False |
| effb1 | 224 | energy_mJ | 1301 | 1152 | 1829 | 1168 | 11.4 | True | 0.6032 | False |
| swins | 224 | energy_mJ | 5130 | 4545 | 7214 | 4390 | 16.9 | False | 0.7504 | False |
| yolo12m | 640 | energy_mJ | 1.692e+04 | 1.499e+04 | 2.38e+04 | 2.012e+04 | 15.9 | True | 0.7156 | False |
| resnet34 | 224 | latency_ms | 125.6 | 112.5 | 179.4 | 130 | 3.4 | True | 1.834 | False |
| mbv3l | 224 | latency_ms | 66.68 | 59.74 | 95.23 | 57.09 | 16.8 | False | 1.584 | False |
| effb1 | 224 | latency_ms | 138.4 | 124 | 197.6 | 119.6 | 15.7 | False | 0.6032 | False |
| swins | 224 | latency_ms | 474 | 424.6 | 676.8 | 387.1 | 22.4 | False | 0.7504 | False |
| yolo12m | 640 | latency_ms | 1525 | 1366 | 2178 | 1849 | 17.5 | True | 0.7156 | False |
| resnet34 | 224 | mean_P_W | 11.44 | 10.74 | 12.2 | 11.28 | 1.4 | True | 1.834 | False |
| mbv3l | 224 | mean_P_W | 9.182 | 8.62 | 9.791 | 9.586 | 4.2 | True | 1.584 | False |
| effb1 | 224 | mean_P_W | 9.287 | 8.719 | 9.903 | 9.767 | 4.9 | True | 0.6032 | False |
| swins | 224 | mean_P_W | 11.41 | 10.71 | 12.16 | 11.34 | 0.6 | True | 0.7504 | False |
| yolo12m | 640 | mean_P_W | 10.74 | 10.08 | 11.45 | 10.88 | 1.3 | True | 0.7156 | False |

## Deployment calculator (serve phase, predicted duration)

| cell_id | measured_J | predicted_J_oh0 | ape_oh0 | predicted_J_est | ape_est |
|---|---|---|---|---|---|
| lockbox2_effb1_224_t4_hoff_2400_r00 | 21.58 | 22.84 | 5.9 | 22.85 | 5.9 |
| lockbox2_effb1_224_t4_hoff_2400_r01 | 21.39 | 22.84 | 6.8 | 22.85 | 6.8 |
| lockbox2_effb1_224_t4_hoff_2400_r02 | 22.36 | 22.84 | 2.1 | 22.85 | 2.1 |
| lockbox2_mbv3l_224_t4_hoff_2400_r00 | 15.59 | 16.05 | 2.9 | 16.05 | 3.0 |
| lockbox2_mbv3l_224_t4_hoff_2400_r01 | 15.33 | 16.05 | 4.7 | 16.05 | 4.7 |
| lockbox2_mbv3l_224_t4_hoff_2400_r02 | 15.2 | 16.05 | 5.5 | 16.05 | 5.6 |
| lockbox2_resnet34_224_t4_hoff_2400_r00 | 24.55 | 24.09 | 1.9 | 24.09 | 1.9 |
| lockbox2_resnet34_224_t4_hoff_2400_r01 | 24.51 | 24.09 | 1.7 | 24.09 | 1.7 |
| lockbox2_resnet34_224_t4_hoff_2400_r02 | 24.59 | 24.09 | 2.0 | 24.09 | 2.0 |
| lockbox2_swins_224_t4_hoff_2400_r00 | 54.05 | 61.13 | 13.1 | 61.14 | 13.1 |
| lockbox2_swins_224_t4_hoff_2400_r01 | 53.75 | 61.13 | 13.7 | 61.14 | 13.8 |
| lockbox2_swins_224_t4_hoff_2400_r02 | 53.77 | 61.13 | 13.7 | 61.14 | 13.7 |
| lockbox2_yolo12m_640_t4_hoff_2400_r00 | 212.3 | 179.1 | 15.6 | 179.1 | 15.7 |
| lockbox2_yolo12m_640_t4_hoff_2400_r01 | 212.8 | 179.1 | 15.9 | 179.1 | 15.9 |
| lockbox2_yolo12m_640_t4_hoff_2400_r02 | 212 | 179.1 | 15.5 | 179.1 | 15.5 |

## Mode L (frozen rungs, median over cells)

| rung | ape | ape_meas | ape_1s | n_cells |
|---|---|---|---|---|
| L0 | 6.2 | 7.1 | 7.1 | 15 |
| L1 | 8.1 | 20.5 | 22.7 | 15 |
| L1-cell | 7.0 | 17.3 | 18.1 | 15 |
| L2 | 7.3 | 7.3 | 6.8 | 15 |
| L3-ridge | 3.3 | 5.5 | 4.6 | 15 |
| L3-trees | 1.7 | 2.2 | 1.7 | 15 |
| L4-ridge | 3.4 | 5.5 | 4.5 | 15 |
| L4-trees | 1.8 | 2.5 | 1.7 | 15 |
| L5-cmos | 3.1 | 4.9 | 4.4 | 15 |
| L5-cmos-trees | 1.5 | 2.0 | 1.5 | 15 |
