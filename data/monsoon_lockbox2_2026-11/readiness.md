# Predictor readiness: monsoon_lockbox2_2026-11

Generated 2026-11-24T23:59:42 by `host/package_dataset.py`.
6 of 10 items satisfied. Each line is computed from the data in this folder where it can be.

- [x] **All cells ok, or explicitly excluded with a reason.** 15 kept + 0 excluded = 15 of 15 manifest cells
- [ ] **Alignment residual < 1 ms for every kept cell, DVFS included.** worst 0.1886 ms (lockbox2_swins_224_t4_hoff_2400_r01) over 15 cells; NO DVFS cell in this selection
- [x] **Counter coverage >= 95 % for every v2 cell.** 15 v2 cells, worst coverage 1.0000 (lockbox2_effb1_224_t4_hoff_2400_r00)
- [ ] **Holdout recorded for every cell.** 6 cells with a missing or wrong holdout, e.g. lockbox2_swins_224_t4_hoff_2400_r00; cells per tag: none 15
- [ ] **Every quantized variant has mAP, energy/frame and latency/frame.** no quantized cell in this selection
- [ ] **mAP of every YOLO cell measured on the Pi (plan Task 16).** 0 of 3 YOLO cells carry a Pi-measured mAP reference
- [x] **Coverage table published.** coverage.csv written (dimension, value, n_cells)
- [x] **README and manifest included, with the v1-vs-v2 counter split.** README.md with the counter-split section; manifest.jsonl + results.jsonl copied; committing them to the dataset branch is the controller's step
- [x] **Probe outcomes documented.** 8 of 8 probes have an outcome in dataset-v2.md section 6
- [x] **Trace repack verified (parquet identical, events gz identical).** 15 of 15 cells repacked and verified (schema, row count, values; gz bytes)
