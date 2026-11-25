# Predictor readiness: monsoon_lockbox3_2026-11

Generated 2026-11-25T15:18:29 by `host/package_dataset.py`.
6 of 10 items satisfied. Each line is computed from the data in this folder where it can be.

- [x] **All cells ok, or explicitly excluded with a reason.** 18 kept + 0 excluded = 18 of 18 manifest cells
- [ ] **Alignment residual < 1 ms for every kept cell, DVFS included.** worst 0.3543 ms (lockbox3_resnet50_224_t4_1500_r02) over 18 cells; NO DVFS cell in this selection
- [x] **Counter coverage >= 95 % for every v2 cell.** 18 v2 cells, worst coverage 1.0000 (lockbox3_convnexts_224_t4_2400_r00)
- [ ] **Holdout recorded for every cell.** 3 cells with a missing or wrong holdout, e.g. lockbox3_convnexts_224_t4_2400_r00; cells per tag: none 18
- [ ] **Every quantized variant has mAP, energy/frame and latency/frame.** no quantized cell in this selection
- [ ] **mAP of every YOLO cell measured on the Pi (plan Task 16).** 0 of 0 YOLO cells carry a Pi-measured mAP reference
- [x] **Coverage table published.** coverage.csv written (dimension, value, n_cells)
- [x] **README and manifest included, with the v1-vs-v2 counter split.** README.md with the counter-split section; manifest.jsonl + results.jsonl copied; committing them to the dataset branch is the controller's step
- [x] **Probe outcomes documented.** 8 of 8 probes have an outcome in dataset-v2.md section 6
- [x] **Trace repack verified (parquet identical, events gz identical).** 18 of 18 cells repacked and verified (schema, row count, values; gz bytes)
