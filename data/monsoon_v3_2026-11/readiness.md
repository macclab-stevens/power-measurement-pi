# Predictor readiness: monsoon_v3_2026-11

Generated 2026-11-24T22:55:26 by `host/package_dataset.py`.
6 of 10 items satisfied. Each line is computed from the data in this folder where it can be.

- [x] **All cells ok, or explicitly excluded with a reason.** 63 kept + 0 excluded = 63 of 63 manifest cells
- [ ] **Alignment residual < 1 ms for every kept cell, DVFS included.** worst 0.3884 ms (v3add_vitb16_224_t4_hoff_2400_r01) over 63 cells; NO DVFS cell in this selection
- [x] **Counter coverage >= 95 % for every v2 cell.** 63 v2 cells, worst coverage 1.0000 (v3add_convnextt_224_t4_hoff_2400_r00)
- [ ] **Holdout recorded for every cell.** 12 cells with a missing or wrong holdout, e.g. v3cal_yolo11s_640_t4_hon_2400_r00; cells per tag: none 63
- [ ] **Every quantized variant has mAP, energy/frame and latency/frame.** no quantized cell in this selection
- [ ] **mAP of every YOLO cell measured on the Pi (plan Task 16).** 18 of 30 YOLO cells carry a Pi-measured mAP reference
- [x] **Coverage table published.** coverage.csv written (dimension, value, n_cells)
- [x] **README and manifest included, with the v1-vs-v2 counter split.** README.md with the counter-split section; manifest.jsonl + results.jsonl copied; committing them to the dataset branch is the controller's step
- [x] **Probe outcomes documented.** 8 of 8 probes have an outcome in dataset-v2.md section 6
- [x] **Trace repack verified (parquet identical, events gz identical).** 63 of 63 cells repacked and verified (schema, row count, values; gz bytes)
