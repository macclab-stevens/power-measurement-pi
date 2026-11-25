# Lockbox, scored again with the R2 models (post hoc (R2): the lockbox was seen; not a confirmatory test)

**post hoc (R2): the lockbox was seen; not a confirmatory test.** The frozen R1 lockbox result (`reports/lockbox.md`, scored once) is unchanged and is only read here, for the R1 columns. Every R2 model below was fitted on the training cells alone; the lockbox configurations enter only as inputs to `predict` (Mode A) or as scored windows (Mode L). The R2 methods were chosen after the lockbox was seen, so these numbers are not a pre-registered test. Percent columns are percent; MdAPE is the median, MAPE the mean and MaxAPE the largest of |pred - truth| / truth over the lockbox configurations (5; 2 of a new model, yolo12n, and 3 of seen models) or cells; width is the median hi90 / lo90.

## Provenance

- run `6fa902daae464555b30ac3a5386c5fab`; seed 20261122; training configurations 58 (base models: effb0, mbv3, resnet18, yolo11m, yolo11n, yolo11s, yolov8m, yolov8n, yolov8s), training cells 403; lockbox cells 15 (15 QC-primary)
- regimes: yolo12n:320 new-model; yolo12n:640 new-model; mbv3:224 seen-model; resnet18:224 seen-model; yolov8n:480 seen-model
- AUTO chose (balanced inner error): energy_mJ/new-model: M4Physics; energy_mJ/seen-model: M1Ridge; latency_ms/new-model: M4Physics; latency_ms/seen-model: M1Ridge; mean_P_W/new-model: M4Physics; mean_P_W/seen-model: M2Trees
- truth equals the frozen truth table (derived/lockbox/<run>_truth.parquet): yes
- lockbox cells with a slip correction in the windows: lockbox_mbv3_224_t4_2400_r00, lockbox_mbv3_224_t4_2400_r01, lockbox_resnet18_224_t4_2400_r00, lockbox_resnet18_224_t4_2400_r01, lockbox_yolo12n_320_t4_2400_r00, lockbox_yolo12n_320_t4_2400_r02, lockbox_yolo12n_640_t4_2400_r00, lockbox_yolo12n_640_t4_2400_r02, lockbox_yolov8n_480_t4_2400_r00, lockbox_yolov8n_480_t4_2400_r01, lockbox_yolov8n_480_t4_2400_r02
- Mode L finals (pickle, sha256): L0 `mode_l_L0_r2.pkl` `3e47b5cfefc4`; L1 `mode_l_L1_r2.pkl` `81eca6889157`; L1-cell `mode_l_L1-cell_r2.pkl` `a00025350139`; L2 `mode_l_L2_r2.pkl` `b99e8cf96ce3`; L3-ridge `mode_l_L3-ridge_r2.pkl` `de35dcf020fa`; L3-trees `mode_l_L3-trees_r2.pkl` `6e09a1e2af8f`; L4-ridge `mode_l_L4-ridge_r2.pkl` `5b69c0fd8c9e`; L4-trees `mode_l_L4-trees_r2.pkl` `8572ea53ffe9`; L5-cmos `mode_l_L5-cmos_r2.pkl` `898b87661078`; L5-cmos-trees `mode_l_L5-cmos-trees_r2.pkl` `9b96aba88f36`
- R1 columns read from the frozen run `bfff786a637b4030ade0f37ee1cf9163`

## Mode A: AUTO and M4Physics, R2 beside R1 (post hoc (R2))

R1 is the frozen primary (AUTO-final: M1Ridge for all three targets) and the frozen secondaries from `lockbox.md`. R2 AUTO is the candidate with the lowest balanced inner error in the regime of the configuration; the regime of each configuration is in the per-configuration table.

| target | group | revision | candidate | n | mdape | mape | maxape | inside_90 | width |
|---|---|---|---|---|---|---|---|---|---|
| energy_mJ | all | R1 | M1Ridge (primary) | 5 | 7.38 | 8.04 | 10.91 | 100.00 | 5.30 |
| energy_mJ | all | R2 | M4Physics | 5 | 6.96 | 12.53 | 23.82 | 80.00 | 1.70 |
| energy_mJ | all | R2 | AUTO | 5 | 3.36 | 10.75 | 23.82 | 80.00 | 1.26 |
| energy_mJ | new model | R1 | M1Ridge (primary) | 2 | 6.58 | 6.58 | 7.38 | 100.00 | 5.30 |
| energy_mJ | new model | R2 | M4Physics | 2 | 22.91 | 22.91 | 23.82 | 50.00 | 1.62 |
| energy_mJ | new model | R2 | AUTO | 2 | 22.91 | 22.91 | 23.82 | 50.00 | 1.62 |
| energy_mJ | seen model | R1 | M1Ridge (primary) | 3 | 8.95 | 9.02 | 10.91 | 100.00 | 5.30 |
| energy_mJ | seen model | R2 | M4Physics | 3 | 5.61 | 5.62 | 6.96 | 100.00 | 1.70 |
| energy_mJ | seen model | R2 | AUTO | 3 | 2.37 | 2.65 | 3.36 | 100.00 | 1.26 |
| latency_ms | all | R1 | M1Ridge (primary) | 5 | 11.78 | 10.16 | 14.36 | 100.00 | 2.88 |
| latency_ms | all | R2 | M4Physics | 5 | 9.41 | 14.26 | 28.04 | 60.00 | 1.65 |
| latency_ms | all | R2 | AUTO | 5 | 3.62 | 11.31 | 28.04 | 60.00 | 1.27 |
| latency_ms | new model | R1 | M1Ridge (primary) | 2 | 5.31 | 5.31 | 7.26 | 100.00 | 2.88 |
| latency_ms | new model | R2 | M4Physics | 2 | 25.31 | 25.31 | 28.04 | 0.00 | 1.61 |
| latency_ms | new model | R2 | AUTO | 2 | 25.31 | 25.31 | 28.04 | 0.00 | 1.61 |
| latency_ms | seen model | R1 | M1Ridge (primary) | 3 | 14.05 | 13.40 | 14.36 | 100.00 | 2.88 |
| latency_ms | seen model | R2 | M4Physics | 3 | 7.60 | 6.89 | 9.41 | 100.00 | 1.65 |
| latency_ms | seen model | R2 | AUTO | 3 | 1.66 | 1.98 | 3.62 | 100.00 | 1.27 |
| mean_P_W | all | R1 | M1Ridge (primary) | 5 | 1.71 | 2.83 | 5.09 | 80.00 | 1.15 |
| mean_P_W | all | R2 | M4Physics | 5 | 2.15 | 2.41 | 5.87 | 100.00 | 1.09 |
| mean_P_W | all | R2 | AUTO | 5 | 0.74 | 1.70 | 5.87 | 100.00 | 1.13 |
| mean_P_W | new model | R1 | M1Ridge (primary) | 2 | 4.82 | 4.82 | 5.09 | 50.00 | 1.15 |
| mean_P_W | new model | R2 | M4Physics | 2 | 3.31 | 3.31 | 5.87 | 100.00 | 1.14 |
| mean_P_W | new model | R2 | AUTO | 2 | 3.31 | 3.31 | 5.87 | 100.00 | 1.14 |
| mean_P_W | seen model | R1 | M1Ridge (primary) | 3 | 1.64 | 1.50 | 1.71 | 100.00 | 1.15 |
| mean_P_W | seen model | R2 | M4Physics | 3 | 2.15 | 1.81 | 2.70 | 100.00 | 1.09 |
| mean_P_W | seen model | R2 | AUTO | 3 | 0.34 | 0.62 | 1.21 | 100.00 | 1.13 |

## Mode A: every R2 candidate, all five configurations

| target | candidate | n | mdape | mape | maxape | inside_90 | width |
|---|---|---|---|---|---|---|---|
| energy_mJ | B0Median | 5 | 135.33 | 264.13 | 866.52 | 80.00 | 41.06 |
| energy_mJ | B1MacsProportional | 5 | 48.66 | 46.91 | 84.75 | 80.00 | 8.95 |
| energy_mJ | B2NearestModel | 5 | 19.94 | 18.03 | 44.56 | 100.00 | 6.50 |
| energy_mJ | B3LogLinear | 5 | 31.46 | 44.20 | 120.52 | 80.00 | 2.85 |
| energy_mJ | M1Ridge | 5 | 3.36 | 6.15 | 11.96 | 100.00 | 1.26 |
| energy_mJ | M2Trees | 5 | 15.78 | 18.41 | 37.97 | 100.00 | 1.56 |
| energy_mJ | M3Factored | 5 | 3.93 | 6.14 | 13.34 | 100.00 | 1.26 |
| energy_mJ | M4Physics | 5 | 6.96 | 12.53 | 23.82 | 80.00 | 1.70 |
| energy_mJ | AUTO | 5 | 3.36 | 10.75 | 23.82 | 80.00 | 1.26 |
| latency_ms | B0Median | 5 | 159.94 | 302.76 | 926.87 | 80.00 | 31.94 |
| latency_ms | B1MacsProportional | 5 | 46.93 | 53.86 | 86.04 | 80.00 | 10.75 |
| latency_ms | B2NearestModel | 5 | 20.35 | 18.72 | 49.88 | 100.00 | 7.48 |
| latency_ms | B3LogLinear | 5 | 32.52 | 48.10 | 133.01 | 80.00 | 3.01 |
| latency_ms | M1Ridge | 5 | 3.62 | 5.87 | 13.05 | 100.00 | 1.27 |
| latency_ms | M2Trees | 5 | 5.42 | 15.43 | 35.27 | 100.00 | 1.62 |
| latency_ms | M4Physics | 5 | 9.41 | 14.26 | 28.04 | 60.00 | 1.65 |
| latency_ms | AUTO | 5 | 3.62 | 11.31 | 28.04 | 60.00 | 1.27 |
| mean_P_W | B0Median | 5 | 9.94 | 9.34 | 17.81 | 100.00 | 1.90 |
| mean_P_W | B1MacsProportional | 5 | 4.80 | 5.12 | 9.00 | 100.00 | 1.45 |
| mean_P_W | B2NearestModel | 5 | 0.77 | 1.29 | 3.55 | 100.00 | 1.37 |
| mean_P_W | B3LogLinear | 5 | 1.58 | 2.57 | 5.36 | 100.00 | 1.13 |
| mean_P_W | M1Ridge | 5 | 1.39 | 1.36 | 1.84 | 100.00 | 1.10 |
| mean_P_W | M2Trees | 5 | 0.33 | 0.47 | 1.21 | 100.00 | 1.13 |
| mean_P_W | M4Physics | 5 | 2.15 | 2.41 | 5.87 | 100.00 | 1.09 |
| mean_P_W | AUTO | 5 | 0.74 | 1.70 | 5.87 | 100.00 | 1.13 |

## Mode A: energy, every R2 candidate by group

| group | candidate | n | mdape | mape | maxape | inside_90 | width |
|---|---|---|---|---|---|---|---|
| new model | B0Median | 2 | 79.86 | 79.86 | 135.33 | 100.00 | 65.27 |
| seen model | B0Median | 3 | 225.37 | 386.98 | 866.52 | 66.67 | 41.06 |
| new model | B1MacsProportional | 2 | 41.34 | 41.34 | 48.66 | 100.00 | 15.36 |
| seen model | B1MacsProportional | 3 | 56.31 | 50.62 | 84.75 | 66.67 | 8.95 |
| new model | B2NearestModel | 2 | 20.78 | 20.78 | 21.63 | 100.00 | 10.75 |
| seen model | B2NearestModel | 3 | 2.08 | 16.20 | 44.56 | 100.00 | 6.50 |
| new model | B3LogLinear | 2 | 11.82 | 11.82 | 20.84 | 100.00 | 6.56 |
| seen model | B3LogLinear | 3 | 45.41 | 65.79 | 120.52 | 66.67 | 2.85 |
| new model | M1Ridge | 2 | 11.39 | 11.39 | 11.96 | 100.00 | 6.53 |
| seen model | M1Ridge | 3 | 2.37 | 2.65 | 3.36 | 100.00 | 1.26 |
| new model | M2Trees | 2 | 37.42 | 37.42 | 37.97 | 100.00 | 5.15 |
| seen model | M2Trees | 3 | 1.04 | 5.74 | 15.78 | 100.00 | 1.56 |
| new model | M3Factored | 2 | 11.79 | 11.79 | 13.34 | 100.00 | 8.09 |
| seen model | M3Factored | 3 | 1.86 | 2.37 | 3.93 | 100.00 | 1.26 |
| new model | M4Physics | 2 | 22.91 | 22.91 | 23.82 | 50.00 | 1.62 |
| seen model | M4Physics | 3 | 5.61 | 5.62 | 6.96 | 100.00 | 1.70 |
| new model | AUTO | 2 | 22.91 | 22.91 | 23.82 | 50.00 | 1.62 |
| seen model | AUTO | 3 | 2.37 | 2.65 | 3.36 | 100.00 | 1.26 |

## Deployment calculator, serve-phase energy of the 15 lockbox cells (APE %)

R2 uses the R2 AUTO energy and latency predictions; R1 the frozen predictions (`reports/r2/lockbox_calculator_gap.md`). `measured duration` leaks the measured serve duration; the two gap scenarios do not use it.

| variant | n | R2 median | R2 mean | R2 max | R1 median | R1 mean | R1 max |
|---|---|---|---|---|---|---|---|
| measured duration (frozen R1 calculator, leaks the duration) | 15 | 1.15 | 3.34 | 9.00 | 1.05 | 1.91 | 5.16 |
| gap scenario, overhead 0 | 15 | 1.89 | 8.51 | 20.25 | 3.63 | 4.89 | 9.10 |
| gap scenario, overhead estimated from training cells | 15 | 1.88 | 8.53 | 20.25 | 3.73 | 4.94 | 9.11 |

## Mode L by family (post hoc (R2))

R2 finals fitted on QC-primary training YOLO counter cells (slip-corrected windows, CMOS features); 15 lockbox cells. YOLO = yolo12n and yolov8n (9 cells), non-YOLO = resnet18 and mbv3 (6 cells, outside the training families). Whole = whole cell, inference = the `forward` phase, 1 s block = median over 10-window blocks, all as median energy APE in percent over cells; forward MAE (W) and R^2 are over the pooled 100 ms forward windows (the frozen R1 run did not record window sums, so those are R2 only).

| family | rung | n_cells | whole | inference | block_1s | fwd_mae_W | fwd_r2 | all_r2 | R1 whole | R1 inference | R1 1 s block |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO | L0 | 9 | 3.738 | 1.838 | 5.738 | 0.351 | 0.896 | 0.910 | 3.738 | 1.839 | 5.748 |
| YOLO | L1 | 9 | 5.420 | 18.399 | 21.783 | 1.818 | -0.984 | -0.157 | 5.419 | 18.400 | 21.785 |
| YOLO | L1-cell | 9 | 5.696 | 28.099 | 26.648 | 2.228 | -2.541 | -0.724 | 5.695 | 28.092 | 26.808 |
| YOLO | L2 | 9 | 1.317 | 1.446 | 1.642 | 0.206 | 0.967 | 0.952 | 1.316 | 1.439 | 1.646 |
| YOLO | L3-ridge | 9 | 2.259 | 1.683 | 3.306 | 0.240 | 0.958 | 0.914 | 2.259 | 1.686 | 3.394 |
| YOLO | L3-trees | 9 | 0.375 | 1.808 | 0.997 | 0.201 | 0.969 | 0.992 | 0.416 | 1.626 | 0.998 |
| YOLO | L4-ridge | 9 | 2.331 | 1.733 | 3.678 | 0.240 | 0.959 | 0.916 | 2.331 | 1.736 | 3.767 |
| YOLO | L4-trees | 9 | 0.368 | 1.655 | 0.996 | 0.190 | 0.972 | 0.992 | 0.363 | 1.763 | 0.982 |
| YOLO | L5-cmos | 9 | 2.288 | 2.160 | 2.735 | 0.258 | 0.952 | 0.935 |  |  |  |
| YOLO | L5-cmos-trees | 9 | 0.544 | 1.070 | 1.288 | 0.188 | 0.973 | 0.987 |  |  |  |
| non-YOLO | L0 | 6 | 4.499 | 8.998 | 8.291 | 0.889 | 0.022 | 0.895 | 4.506 | 8.984 | 8.320 |
| non-YOLO | L1 | 6 | 6.050 | 16.035 | 20.050 | 1.904 | -4.840 | 0.041 | 6.049 | 16.035 | 19.980 |
| non-YOLO | L1-cell | 6 | 3.912 | 11.815 | 17.652 | 1.915 | -6.391 | -0.274 | 3.912 | 11.815 | 17.575 |
| non-YOLO | L2 | 6 | 5.773 | 7.550 | 6.166 | 0.741 | 0.323 | 0.939 | 5.773 | 7.536 | 6.169 |
| non-YOLO | L3-ridge | 6 | 4.756 | 6.469 | 4.580 | 0.608 | 0.500 | 0.911 | 4.756 | 6.466 | 4.575 |
| non-YOLO | L3-trees | 6 | 4.931 | 7.238 | 5.659 | 0.693 | 0.404 | 0.972 | 4.862 | 7.355 | 5.591 |
| non-YOLO | L4-ridge | 6 | 4.771 | 6.539 | 4.460 | 0.610 | 0.503 | 0.912 | 4.768 | 6.536 | 4.484 |
| non-YOLO | L4-trees | 6 | 4.924 | 7.333 | 5.697 | 0.700 | 0.389 | 0.971 | 4.941 | 7.341 | 5.707 |
| non-YOLO | L5-cmos | 6 | 4.362 | 6.707 | 3.857 | 0.615 | 0.442 | 0.920 |  |  |  |
| non-YOLO | L5-cmos-trees | 6 | 4.529 | 7.705 | 4.831 | 0.700 | 0.351 | 0.968 |  |  |  |

## Probe: Mode L power x measured latency of another repeat, against Mode A R2 AUTO

Median APE (percent) of the energy per image of repeat k, from the mean predicted power over the measured phase of repeat j times j's latency (ordered pairs of one configuration), per rung, and the median APE of the R2 AUTO prediction on the same pairs; R1 is the frozen probe against the frozen primary.

| rung | n_pairs | probe APE | Mode A R2 AUTO APE | R1 n_pairs | R1 probe APE | R1 Mode A APE |
|---|---|---|---|---|---|---|
| L0 | 30 | 8.04 | 3.40 | 30.00 | 8.04 | 7.47 |
| L1 | 30 | 21.87 | 3.40 | 30.00 | 21.87 | 7.47 |
| L1-cell | 30 | 22.29 | 3.40 | 30.00 | 22.29 | 7.47 |
| L2 | 30 | 7.31 | 3.40 | 30.00 | 7.32 | 7.47 |
| L3-ridge | 30 | 5.86 | 3.40 | 30.00 | 5.86 | 7.47 |
| L3-trees | 30 | 7.84 | 3.40 | 30.00 | 8.08 | 7.47 |
| L4-ridge | 30 | 6.06 | 3.40 | 30.00 | 6.07 | 7.47 |
| L4-trees | 30 | 7.97 | 3.40 | 30.00 | 7.99 | 7.47 |
| L5-cmos | 30 | 5.49 | 3.40 |  |  |  |
| L5-cmos-trees | 30 | 7.18 | 3.40 |  |  |  |

## Per configuration (Mode A, AUTO, M4Physics and R1 primary)

| target | model | imgsz | regime | candidate | pred | lo90 | hi90 | truth | ape | inside |
|---|---|---|---|---|---|---|---|---|---|---|
| energy_mJ | mbv3 | 224 | seen-model | AUTO=M1Ridge | 299.140 | 275.431 | 345.884 | 289.420 | 3.358 | yes |
| energy_mJ | resnet18 | 224 | seen-model | AUTO=M1Ridge | 840.665 | 774.036 | 972.028 | 859.742 | 2.219 | yes |
| energy_mJ | yolov8n | 480 | seen-model | AUTO=M1Ridge | 1615.413 | 1487.379 | 1867.839 | 1654.652 | 2.371 | yes |
| energy_mJ | yolo12n | 320 | new-model | AUTO=M4Physics | 905.599 | 716.103 | 1161.845 | 1188.688 | 23.815 | no |
| energy_mJ | yolo12n | 640 | new-model | AUTO=M4Physics | 2885.797 | 2281.948 | 3702.356 | 3700.086 | 22.007 | yes |
| energy_mJ | mbv3 | 224 | seen-model | M4Physics | 301.811 | 224.671 | 382.806 | 289.420 | 4.281 | yes |
| energy_mJ | resnet18 | 224 | seen-model | M4Physics | 811.517 | 604.101 | 1029.296 | 859.742 | 5.609 | yes |
| energy_mJ | yolo12n | 320 | new-model | M4Physics | 905.599 | 716.103 | 1161.845 | 1188.688 | 23.815 | no |
| energy_mJ | yolo12n | 640 | new-model | M4Physics | 2885.797 | 2281.948 | 3702.356 | 3700.086 | 22.007 | yes |
| energy_mJ | yolov8n | 480 | seen-model | M4Physics | 1539.498 | 1146.016 | 1952.639 | 1654.652 | 6.959 | yes |
| latency_ms | mbv3 | 224 | seen-model | AUTO=M1Ridge | 32.477 | 29.670 | 37.608 | 31.948 | 1.656 | yes |
| latency_ms | resnet18 | 224 | seen-model | AUTO=M1Ridge | 78.746 | 71.938 | 91.185 | 79.267 | 0.657 | yes |
| latency_ms | yolov8n | 480 | seen-model | AUTO=M1Ridge | 157.406 | 143.798 | 182.271 | 163.327 | 3.625 | yes |
| latency_ms | yolo12n | 320 | new-model | AUTO=M4Physics | 90.819 | 71.955 | 116.007 | 126.210 | 28.041 | no |
| latency_ms | yolo12n | 640 | new-model | AUTO=M4Physics | 289.405 | 229.293 | 369.672 | 373.824 | 22.583 | no |
| latency_ms | mbv3 | 224 | seen-model | M4Physics | 33.122 | 25.657 | 42.431 | 31.948 | 3.675 | yes |
| latency_ms | resnet18 | 224 | seen-model | M4Physics | 73.245 | 56.736 | 93.830 | 79.267 | 7.596 | yes |
| latency_ms | yolo12n | 320 | new-model | M4Physics | 90.819 | 71.955 | 116.007 | 126.210 | 28.041 | no |
| latency_ms | yolo12n | 640 | new-model | M4Physics | 289.405 | 229.293 | 369.672 | 373.824 | 22.583 | no |
| latency_ms | yolov8n | 480 | seen-model | M4Physics | 147.962 | 114.612 | 189.545 | 163.327 | 9.407 | yes |
| mean_P_W | mbv3 | 224 | seen-model | AUTO=M2Trees | 9.028 | 8.392 | 9.459 | 9.059 | 0.343 | yes |
| mean_P_W | resnet18 | 224 | seen-model | AUTO=M2Trees | 10.715 | 9.960 | 11.226 | 10.846 | 1.209 | yes |
| mean_P_W | yolov8n | 480 | seen-model | AUTO=M2Trees | 10.099 | 9.387 | 10.580 | 10.131 | 0.320 | yes |
| mean_P_W | yolo12n | 320 | new-model | AUTO=M4Physics | 9.971 | 9.382 | 10.672 | 9.418 | 5.874 | yes |
| mean_P_W | yolo12n | 640 | new-model | AUTO=M4Physics | 9.971 | 9.382 | 10.672 | 9.898 | 0.743 | yes |
| mean_P_W | mbv3 | 224 | seen-model | M4Physics | 9.112 | 8.715 | 9.507 | 9.059 | 0.584 | yes |
| mean_P_W | resnet18 | 224 | seen-model | M4Physics | 11.079 | 10.597 | 11.559 | 10.846 | 2.150 | yes |
| mean_P_W | yolo12n | 320 | new-model | M4Physics | 9.971 | 9.382 | 10.672 | 9.418 | 5.874 | yes |
| mean_P_W | yolo12n | 640 | new-model | M4Physics | 9.971 | 9.382 | 10.672 | 9.898 | 0.743 | yes |
| mean_P_W | yolov8n | 480 | seen-model | M4Physics | 10.405 | 9.952 | 10.855 | 10.131 | 2.702 | yes |

## Per cell (calculator)

| cell_id | model | imgsz | measured_J | predicted_J_known | ape_known | predicted_J_oh0 | ape_oh0 | predicted_J_est | ape_est |
|---|---|---|---|---|---|---|---|---|---|
| lockbox_mbv3_224_t4_2400_r00 | mbv3 | 224 | 12.785 | 12.839 | 0.423 | 12.866 | 0.632 | 12.883 | 0.766 |
| lockbox_mbv3_224_t4_2400_r01 | mbv3 | 224 | 12.776 | 12.892 | 0.904 | 12.866 | 0.703 | 12.883 | 0.837 |
| lockbox_mbv3_224_t4_2400_r02 | mbv3 | 224 | 12.696 | 12.840 | 1.140 | 12.866 | 1.343 | 12.883 | 1.478 |
| lockbox_resnet18_224_t4_2400_r00 | resnet18 | 224 | 18.648 | 18.289 | 1.925 | 18.281 | 1.968 | 18.298 | 1.876 |
| lockbox_resnet18_224_t4_2400_r01 | resnet18 | 224 | 18.510 | 18.296 | 1.153 | 18.281 | 1.234 | 18.298 | 1.141 |
| lockbox_resnet18_224_t4_2400_r02 | resnet18 | 224 | 18.421 | 18.291 | 0.705 | 18.281 | 0.759 | 18.298 | 0.666 |
| lockbox_yolo12n_320_t4_2400_r00 | yolo12n | 320 | 23.571 | 22.030 | 6.537 | 18.931 | 19.688 | 18.927 | 19.705 |
| lockbox_yolo12n_320_t4_2400_r01 | yolo12n | 320 | 23.238 | 21.987 | 5.383 | 18.931 | 18.535 | 18.927 | 18.553 |
| lockbox_yolo12n_320_t4_2400_r02 | yolo12n | 320 | 23.002 | 21.926 | 4.678 | 18.931 | 17.699 | 18.927 | 17.717 |
| lockbox_yolo12n_640_t4_2400_r00 | yolo12n | 640 | 48.565 | 44.196 | 8.997 | 38.733 | 20.247 | 38.729 | 20.255 |
| lockbox_yolo12n_640_t4_2400_r01 | yolo12n | 640 | 48.483 | 44.227 | 8.780 | 38.733 | 20.111 | 38.729 | 20.120 |
| lockbox_yolo12n_640_t4_2400_r02 | yolo12n | 640 | 48.469 | 44.262 | 8.679 | 38.733 | 20.088 | 38.729 | 20.096 |
| lockbox_yolov8n_480_t4_2400_r00 | yolov8n | 480 | 26.529 | 26.397 | 0.498 | 26.029 | 1.886 | 26.025 | 1.901 |
| lockbox_yolov8n_480_t4_2400_r01 | yolov8n | 480 | 26.406 | 26.359 | 0.175 | 26.029 | 1.427 | 26.025 | 1.442 |
| lockbox_yolov8n_480_t4_2400_r02 | yolov8n | 480 | 26.397 | 26.346 | 0.192 | 26.029 | 1.396 | 26.025 | 1.411 |

## Per cell (Mode L, rungs L0, L3-trees, L5-cmos, L5-cmos-trees; the other rungs are in the CSV)

| cell_id | rung | E_true_J | E_pred_J | ape | ape_meas | ape_1s | mae_fwd_W |
|---|---|---|---|---|---|---|---|
| lockbox_mbv3_224_t4_2400_r00 | L0 | 2343.089 | 2445.976 | 4.391 | 9.974 | 8.724 | 0.920 |
| lockbox_mbv3_224_t4_2400_r01 | L0 | 233.031 | 256.570 | 10.101 | 10.848 | 10.642 | 0.993 |
| lockbox_mbv3_224_t4_2400_r02 | L0 | 232.700 | 255.880 | 9.961 | 10.661 | 10.579 | 0.977 |
| lockbox_resnet18_224_t4_2400_r00 | L0 | 929.095 | 934.659 | 0.599 | 8.021 | 7.588 | 0.880 |
| lockbox_resnet18_224_t4_2400_r01 | L0 | 321.379 | 306.572 | 4.607 | 7.447 | 7.858 | 0.835 |
| lockbox_resnet18_224_t4_2400_r02 | L0 | 319.845 | 306.501 | 4.172 | 7.083 | 7.376 | 0.772 |
| lockbox_yolo12n_320_t4_2400_r00 | L0 | 2779.006 | 2901.541 | 4.409 | 3.374 | 6.474 | 0.569 |
| lockbox_yolo12n_320_t4_2400_r01 | L0 | 113.366 | 120.935 | 6.677 | 3.886 | 7.490 | 0.628 |
| lockbox_yolo12n_320_t4_2400_r02 | L0 | 112.846 | 121.213 | 7.414 | 4.523 | 7.803 | 0.675 |
| lockbox_yolo12n_640_t4_2400_r00 | L0 | 2938.571 | 3053.849 | 3.923 | 0.354 | 5.808 | 0.250 |
| lockbox_yolo12n_640_t4_2400_r01 | L0 | 214.953 | 221.196 | 2.904 | 0.317 | 2.798 | 0.276 |
| lockbox_yolo12n_640_t4_2400_r02 | L0 | 214.013 | 221.318 | 3.413 | 0.976 | 2.353 | 0.256 |
| lockbox_yolov8n_480_t4_2400_r00 | L0 | 2031.288 | 2107.221 | 3.738 | 2.098 | 5.738 | 0.332 |
| lockbox_yolov8n_480_t4_2400_r01 | L0 | 129.402 | 134.108 | 3.636 | 1.091 | 4.990 | 0.264 |
| lockbox_yolov8n_480_t4_2400_r02 | L0 | 128.810 | 133.087 | 3.321 | 1.838 | 4.812 | 0.310 |
| lockbox_mbv3_224_t4_2400_r00 | L3-trees | 2343.089 | 2356.772 | 0.584 | 8.316 | 1.055 | 0.756 |
| lockbox_mbv3_224_t4_2400_r01 | L3-trees | 233.031 | 248.109 | 6.471 | 8.353 | 7.965 | 0.753 |
| lockbox_mbv3_224_t4_2400_r02 | L3-trees | 232.700 | 247.472 | 6.348 | 8.164 | 7.825 | 0.737 |
| lockbox_resnet18_224_t4_2400_r00 | L3-trees | 929.095 | 914.926 | 1.525 | 5.604 | 1.334 | 0.610 |
| lockbox_resnet18_224_t4_2400_r01 | L3-trees | 321.379 | 305.776 | 4.855 | 5.998 | 5.466 | 0.651 |
| lockbox_resnet18_224_t4_2400_r02 | L3-trees | 319.845 | 303.833 | 5.006 | 6.312 | 5.853 | 0.680 |
| lockbox_yolo12n_320_t4_2400_r00 | L3-trees | 2779.006 | 2783.536 | 0.163 | 2.519 | 0.997 | 0.228 |
| lockbox_yolo12n_320_t4_2400_r01 | L3-trees | 113.366 | 115.172 | 1.593 | 2.421 | 1.322 | 0.208 |
| lockbox_yolo12n_320_t4_2400_r02 | L3-trees | 112.846 | 115.019 | 1.925 | 2.733 | 1.880 | 0.232 |
| lockbox_yolo12n_640_t4_2400_r00 | L3-trees | 2938.571 | 2948.594 | 0.341 | 1.808 | 0.848 | 0.210 |
| lockbox_yolo12n_640_t4_2400_r01 | L3-trees | 214.953 | 217.235 | 1.062 | 1.245 | 1.589 | 0.218 |
| lockbox_yolo12n_640_t4_2400_r02 | L3-trees | 214.013 | 217.250 | 1.513 | 1.924 | 1.591 | 0.213 |
| lockbox_yolov8n_480_t4_2400_r00 | L3-trees | 2031.288 | 2032.952 | 0.082 | 1.466 | 0.898 | 0.184 |
| lockbox_yolov8n_480_t4_2400_r01 | L3-trees | 129.402 | 128.917 | 0.375 | 0.819 | 0.470 | 0.133 |
| lockbox_yolov8n_480_t4_2400_r02 | L3-trees | 128.810 | 128.416 | 0.305 | 0.911 | 0.454 | 0.160 |
| lockbox_mbv3_224_t4_2400_r00 | L5-cmos | 2343.089 | 2479.526 | 5.823 | 10.508 | 4.130 | 0.955 |
| lockbox_mbv3_224_t4_2400_r01 | L5-cmos | 233.031 | 251.153 | 7.776 | 10.127 | 9.797 | 0.913 |
| lockbox_mbv3_224_t4_2400_r02 | L5-cmos | 232.700 | 250.416 | 7.613 | 9.944 | 9.556 | 0.898 |
| lockbox_resnet18_224_t4_2400_r00 | L5-cmos | 929.095 | 956.055 | 2.902 | 3.219 | 3.584 | 0.351 |
| lockbox_resnet18_224_t4_2400_r01 | L5-cmos | 321.379 | 312.848 | 2.654 | 3.444 | 3.171 | 0.373 |
| lockbox_resnet18_224_t4_2400_r02 | L5-cmos | 319.845 | 311.489 | 2.613 | 3.471 | 3.071 | 0.374 |
| lockbox_yolo12n_320_t4_2400_r00 | L5-cmos | 2779.006 | 2949.639 | 6.140 | 6.384 | 5.790 | 0.542 |
| lockbox_yolo12n_320_t4_2400_r01 | L5-cmos | 113.366 | 116.926 | 3.140 | 5.692 | 3.577 | 0.480 |
| lockbox_yolo12n_320_t4_2400_r02 | L5-cmos | 112.846 | 115.428 | 2.288 | 5.163 | 2.735 | 0.434 |
| lockbox_yolo12n_640_t4_2400_r00 | L5-cmos | 2938.571 | 3046.931 | 3.687 | 2.467 | 4.469 | 0.236 |
| lockbox_yolo12n_640_t4_2400_r01 | L5-cmos | 214.953 | 217.808 | 1.328 | 1.515 | 1.978 | 0.196 |
| lockbox_yolo12n_640_t4_2400_r02 | L5-cmos | 214.013 | 218.579 | 2.133 | 2.160 | 2.314 | 0.211 |
| lockbox_yolov8n_480_t4_2400_r00 | L5-cmos | 2031.288 | 2108.080 | 3.780 | 0.651 | 4.052 | 0.152 |
| lockbox_yolov8n_480_t4_2400_r01 | L5-cmos | 129.402 | 128.861 | 0.418 | 0.565 | 1.097 | 0.144 |
| lockbox_yolov8n_480_t4_2400_r02 | L5-cmos | 128.810 | 128.578 | 0.180 | 0.560 | 1.073 | 0.156 |
| lockbox_mbv3_224_t4_2400_r00 | L5-cmos-trees | 2343.089 | 2365.982 | 0.977 | 9.831 | 1.618 | 0.893 |
| lockbox_mbv3_224_t4_2400_r01 | L5-cmos-trees | 233.031 | 250.306 | 7.413 | 9.665 | 9.443 | 0.871 |
| lockbox_mbv3_224_t4_2400_r02 | L5-cmos-trees | 232.700 | 249.699 | 7.305 | 9.466 | 9.184 | 0.855 |
| lockbox_resnet18_224_t4_2400_r00 | L5-cmos-trees | 929.095 | 918.655 | 1.124 | 4.327 | 1.404 | 0.474 |
| lockbox_resnet18_224_t4_2400_r01 | L5-cmos-trees | 321.379 | 307.502 | 4.318 | 5.237 | 4.398 | 0.567 |
| lockbox_resnet18_224_t4_2400_r02 | L5-cmos-trees | 319.845 | 304.684 | 4.740 | 5.943 | 5.265 | 0.641 |
| lockbox_yolo12n_320_t4_2400_r00 | L5-cmos-trees | 2779.006 | 2794.113 | 0.544 | 3.353 | 1.337 | 0.292 |
| lockbox_yolo12n_320_t4_2400_r01 | L5-cmos-trees | 113.366 | 115.934 | 2.266 | 3.367 | 1.871 | 0.292 |
| lockbox_yolo12n_320_t4_2400_r02 | L5-cmos-trees | 112.846 | 115.633 | 2.469 | 3.823 | 1.814 | 0.325 |
| lockbox_yolo12n_640_t4_2400_r00 | L5-cmos-trees | 2938.571 | 2957.496 | 0.644 | 0.885 | 1.390 | 0.144 |
| lockbox_yolo12n_640_t4_2400_r01 | L5-cmos-trees | 214.953 | 216.079 | 0.524 | 0.371 | 0.811 | 0.149 |
| lockbox_yolo12n_640_t4_2400_r02 | L5-cmos-trees | 214.013 | 216.209 | 1.026 | 1.070 | 0.891 | 0.159 |
| lockbox_yolov8n_480_t4_2400_r00 | L5-cmos-trees | 2031.288 | 2037.100 | 0.286 | 1.328 | 1.288 | 0.195 |
| lockbox_yolov8n_480_t4_2400_r01 | L5-cmos-trees | 129.402 | 129.163 | 0.185 | 0.606 | 0.589 | 0.137 |
| lockbox_yolov8n_480_t4_2400_r02 | L5-cmos-trees | 128.810 | 128.796 | 0.010 | 0.544 | 0.618 | 0.166 |

