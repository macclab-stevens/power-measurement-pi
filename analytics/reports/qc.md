# QC report

403 cells; included 403, primary 403, excluded 0. Rules are applied by `analytics/qc.py` with the bounds below (`analytics/config.toml`, `[qc]`). Per-cell values: `qc_cells.csv`.

## Bounds

- `q1_ctx_frac_flag` = 0.03
- `q2_max_residual_ms` = 1.0
- `q3_slip_frac_flag` = 0.01
- `q5_pv_rel_flag` = 0.02
- `q6_sample_ratio_lo` = 0.98
- `q6_sample_ratio_hi` = 1.02

## Rule counts (cells triggering each rule)

Q1flag, Q3flag, Q5flag only annotate. Q2 and Q6 exclude (`include` False). Q4 keeps the cell but it is not `primary`.

|  | v1 | v2 | all |
|---|---|---|---|
| Q1flag | 0 | 0 | 0 |
| Q2 | 0 | 0 | 0 |
| Q3flag | 5 | 24 | 29 |
| Q4 | 0 | 0 | 0 |
| Q5flag | 0 | 0 | 0 |
| Q6 | 0 | 0 | 0 |
| include | 224 | 179 | 403 |
| primary | 224 | 179 | 403 |
| cells | 224 | 179 | 403 |

## Excluded cells (Q2 or Q6)

None.

## Q4 cells (kept, not primary): 0

None.

## Distribution of the QC measures

Finite values only; `n` counts them.

|  | n | min | p50 | p95 | max |
|---|---|---|---|---|---|
| max_residual_ms | 403 | 0.0851437 | 0.172239 | 0.258785 | 0.755539 |
| slip_scored_frac | 403 | 0 | 0 | 0.0343398 | 0.10541 |
| ctx_read_frac_scored | 403 | 0 | 0.0103483 | 0.0136418 | 0.0212797 |
| pv_median_rel | 403 | 0 | 0 | 0 | 0 |
| sample_ratio | 403 | 0.998976 | 0.999991 | 1.00065 | 1.00491 |

## Phase labels in the traces

1246 distinct raw labels in 16 families. Outside the known vocabulary: none.

| label | cells | distinct raw labels |
|---|---|---|
| baseline_A | 403 | 1 |
| baseline_B | 403 | 1 |
| bench:* | 180 | 1228 |
| concurrent_drain | 6 | 1 |
| decode | 48 | 1 |
| first:* | 403 | 4 |
| forward | 259 | 1 |
| gap | 180 | 1 |
| idle | 403 | 1 |
| load_weights | 403 | 1 |
| prefill | 48 | 1 |
| serve | 307 | 1 |
| sync:end | 403 | 1 |
| sync:start | 403 | 1 |
| train | 96 | 1 |
| warmup | 403 | 1 |
