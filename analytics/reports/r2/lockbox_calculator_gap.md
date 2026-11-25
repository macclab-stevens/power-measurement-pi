# Lockbox deployment calculator without the measured duration (post hoc (R2))

The lockbox serve phase is closed loop (request, then a sleep of `serve_gap_ms`), so its wall duration depends on the latency being predicted. The frozen R1 calculator (`reports/lockbox.md`) passed the MEASURED `serve_dur_ms`, which leaks that latency. Here the same frozen Mode A AUTO-final predictions go through the gap scenario: duration = n * (t_pred + gap + overhead). This is **post hoc (R2)**: it was computed after the lockbox was scored; nothing was fitted on a lockbox cell, and the frozen outputs are untouched.

## Provenance

- predictions `analytics/reports/lockbox_predictions.csv` sha256 `2bb7a2606f3b42d9862de0b2e2b16866bad40a3418bcbb735d62812fb1304993`
- training serve cells for the overhead: 250; request overhead (median per family): yolo -0.08 ms, other 0.35 ms, all 0.10 ms
- lockbox cells scored: 15 (the QC-primary ones with a serve phase, as in the frozen calculator); run `566872be58454a509bce537b0f434df1`

## Median and mean APE (post hoc (R2))

| variant | n | median_ape | mean_ape | max_ape |
|---|---|---|---|---|
| measured duration (frozen R1 calculator, leaks the duration) | 15 | 1.1% | 1.9% | 5.2% |
| gap scenario, overhead 0 | 15 | 3.6% | 4.9% | 9.1% |
| gap scenario, overhead estimated from training cells | 15 | 3.7% | 4.9% | 9.1% |

## Per cell

| cell_id | model | imgsz | n_images | gap_s | measured_duration_s | measured_J | predicted_J_known | ape_known | predicted_J_oh0 | ape_oh0 | overhead_est_s | predicted_J_est | ape_est |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lockbox_mbv3_224_t4_2400_r00 | mbv3 | 224 | 10 | 0.2 | 2.319 | 12.79 | 12.84 | 0.4% | 13.03 | 1.9% | 0.0003476 | 13.05 | 2.0% |
| lockbox_mbv3_224_t4_2400_r01 | mbv3 | 224 | 10 | 0.2 | 2.33 | 12.78 | 12.89 | 0.9% | 13.03 | 2.0% | 0.0003476 | 13.05 | 2.1% |
| lockbox_mbv3_224_t4_2400_r02 | mbv3 | 224 | 10 | 0.2 | 2.32 | 12.7 | 12.84 | 1.2% | 13.03 | 2.6% | 0.0003476 | 13.05 | 2.8% |
| lockbox_resnet18_224_t4_2400_r00 | resnet18 | 224 | 10 | 0.2 | 2.789 | 18.65 | 18.52 | 0.7% | 19.09 | 2.4% | 0.0003476 | 19.11 | 2.5% |
| lockbox_resnet18_224_t4_2400_r01 | resnet18 | 224 | 10 | 0.2 | 2.79 | 18.51 | 18.53 | 0.1% | 19.09 | 3.1% | 0.0003476 | 19.11 | 3.2% |
| lockbox_resnet18_224_t4_2400_r02 | resnet18 | 224 | 10 | 0.2 | 2.789 | 18.42 | 18.52 | 0.6% | 19.09 | 3.6% | 0.0003476 | 19.11 | 3.7% |
| lockbox_yolo12n_320_t4_2400_r00 | yolo12n | 320 | 10 | 0.2 | 3.536 | 23.57 | 23.35 | 0.9% | 22.45 | 4.8% | -8.262e-05 | 22.45 | 4.8% |
| lockbox_yolo12n_320_t4_2400_r01 | yolo12n | 320 | 10 | 0.2 | 3.527 | 23.24 | 23.31 | 0.3% | 22.45 | 3.4% | -8.262e-05 | 22.45 | 3.4% |
| lockbox_yolo12n_320_t4_2400_r02 | yolo12n | 320 | 10 | 0.2 | 3.515 | 23 | 23.24 | 1.1% | 22.45 | 2.4% | -8.262e-05 | 22.45 | 2.4% |
| lockbox_yolo12n_640_t4_2400_r00 | yolo12n | 640 | 10 | 0.2 | 6.001 | 48.57 | 46.06 | 5.2% | 44.15 | 9.1% | -8.262e-05 | 44.14 | 9.1% |
| lockbox_yolo12n_640_t4_2400_r01 | yolo12n | 640 | 10 | 0.2 | 6.007 | 48.48 | 46.09 | 4.9% | 44.15 | 8.9% | -8.262e-05 | 44.14 | 9.0% |
| lockbox_yolo12n_640_t4_2400_r02 | yolo12n | 640 | 10 | 0.2 | 6.014 | 48.47 | 46.13 | 4.8% | 44.15 | 8.9% | -8.262e-05 | 44.14 | 8.9% |
| lockbox_yolov8n_480_t4_2400_r00 | yolov8n | 480 | 10 | 0.2 | 3.649 | 26.53 | 27.14 | 2.3% | 28.23 | 6.4% | -8.262e-05 | 28.22 | 6.4% |
| lockbox_yolov8n_480_t4_2400_r01 | yolov8n | 480 | 10 | 0.2 | 3.641 | 26.41 | 27.11 | 2.7% | 28.23 | 6.9% | -8.262e-05 | 28.22 | 6.9% |
| lockbox_yolov8n_480_t4_2400_r02 | yolov8n | 480 | 10 | 0.2 | 3.638 | 26.4 | 27.09 | 2.6% | 28.23 | 6.9% | -8.262e-05 | 28.22 | 6.9% |

