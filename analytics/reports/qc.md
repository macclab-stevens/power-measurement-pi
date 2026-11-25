# QC report

466 cells; included 465, primary 465, excluded 1. Rules are applied by `analytics/qc.py` with the bounds below (`analytics/config.toml`, `[qc]`). Per-cell values: `qc_cells.csv`.

## Bounds

- `q1_ctx_frac_flag` = 0.03
- `q2_max_residual_ms` = 1.0
- `q3_slip_frac_flag` = 0.01
- `q5_pv_rel_flag` = 0.02
- `q6_sample_ratio_lo` = 0.98
- `q6_sample_ratio_hi` = 1.02

## Rule counts (cells triggering each rule)

Q1flag, Q3flag, Q4unparsed, Q5flag only annotate. Q2 and Q6 exclude (`include` False). Q4 keeps the cell but it is not `primary`.

|  | v1 | v2 | v3 | all |
|---|---|---|---|---|
| Q1flag | 0 | 0 | 0 | 0 |
| Q2 | 0 | 0 | 0 | 0 |
| Q3flag | 5 | 24 | 2 | 31 |
| Q4 | 0 | 0 | 0 | 0 |
| Q4unparsed | 0 | 0 | 0 | 0 |
| Q5flag | 0 | 0 | 0 | 0 |
| Q6 | 0 | 0 | 1 | 1 |
| include | 224 | 179 | 62 | 465 |
| primary | 224 | 179 | 62 | 465 |
| cells | 224 | 179 | 63 | 466 |

## Excluded cells (Q2 or Q6)

- `v3cal_yolo12n_640_t4_hon_2400_r02`: Q6: unknown phase labels: done

## Q4 cells (kept, not primary): 0

None.

## Distribution of the QC measures

Finite values only; `n` counts them.

|  | n | min | p50 | p95 | max |
|---|---|---|---|---|---|
| max_residual_ms | 466 | 0.0851437 | 0.170578 | 0.257404 | 0.755539 |
| slip_scored_frac | 466 | 0 | 0 | 0.0338428 | 0.10541 |
| ctx_read_frac_scored | 466 | 0 | 0.010343 | 0.0137521 | 0.0212797 |
| pv_median_rel | 466 | 0 | 0 | 0 | 0 |
| sample_ratio | 466 | 0.998976 | 0.999992 | 1.00061 | 1.00491 |

## Phase labels in the traces

1247 distinct raw labels in 17 families. Outside the known vocabulary: `done`

| label | cells | distinct raw labels |
|---|---|---|
| baseline_A | 466 | 1 |
| baseline_B | 466 | 1 |
| bench:* | 180 | 1228 |
| concurrent_drain | 6 | 1 |
| decode | 48 | 1 |
| done | 1 | 1 |
| first:* | 466 | 4 |
| forward | 322 | 1 |
| gap | 180 | 1 |
| idle | 466 | 1 |
| load_weights | 466 | 1 |
| prefill | 48 | 1 |
| serve | 370 | 1 |
| sync:end | 466 | 1 |
| sync:start | 466 | 1 |
| train | 96 | 1 |
| warmup | 466 | 1 |
