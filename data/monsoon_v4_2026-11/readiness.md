# Predictor readiness: monsoon_v4_2026-11

Generated 2026-11-25T08:52:42 by `host/package_dataset.py`.
7 of 10 items satisfied. Each line is computed from the data in this folder where it can be.

- [x] **All cells ok, or explicitly excluded with a reason.** 128 kept + 0 excluded = 128 of 128 manifest cells
- [ ] **Alignment residual < 1 ms for every kept cell, DVFS included.** worst 0.3466 ms (v4_resnet18_224_t4_1500_r00) over 128 cells; NO DVFS cell in this selection
- [x] **Counter coverage >= 95 % for every v2 cell.** 128 v2 cells, worst coverage 1.0000 (v4_convnextt_224_t4_1500_r00)
- [ ] **Holdout recorded for every cell.** 42 cells with a missing or wrong holdout, e.g. v4_yolov8s_640_t4_1500_r00; cells per tag: none 128
- [x] **Every quantized variant has mAP, energy/frame and latency/frame.** 10 quantized cells, variants ['yolo11n_fp16', 'yolo11n_int8s', 'yolo11s_int8s']
- [ ] **mAP of every YOLO cell measured on the Pi (plan Task 16).** 80 of 94 YOLO cells carry a Pi-measured mAP reference
- [x] **Coverage table published.** coverage.csv written (dimension, value, n_cells)
- [x] **README and manifest included, with the v1-vs-v2 counter split.** README.md with the counter-split section; manifest.jsonl + results.jsonl copied; committing them to the dataset branch is the controller's step
- [x] **Probe outcomes documented.** 8 of 8 probes have an outcome in dataset-v2.md section 6
- [x] **Trace repack verified (parquet identical, events gz identical).** 128 of 128 cells repacked and verified (schema, row count, values; gz bytes)
