# QC report

594 cells; included 593, primary 593, excluded 1. Rules are applied by `analytics/qc.py` with the bounds below (`analytics/config.toml`, `[qc]`). Per-cell values: `qc_cells.csv`.

## Bounds

- `q1_ctx_frac_flag` = 0.03
- `q2_max_residual_ms` = 1.0
- `q3_slip_frac_flag` = 0.01
- `q5_pv_rel_flag` = 0.02
- `q6_sample_ratio_lo` = 0.98
- `q6_sample_ratio_hi` = 1.02

## Rule counts (cells triggering each rule)

Q1flag, Q3flag, Q4unparsed, Q5flag only annotate. Q2 and Q6 exclude (`include` False). Q4 keeps the cell but it is not `primary`.

|  | v1 | v2 | v3 | v4 | all |
|---|---|---|---|---|---|
| Q1flag | 0 | 0 | 0 | 0 | 0 |
| Q2 | 0 | 0 | 0 | 0 | 0 |
| Q3flag | 5 | 24 | 2 | 7 | 38 |
| Q4 | 0 | 0 | 0 | 0 | 0 |
| Q4unparsed | 0 | 0 | 0 | 0 | 0 |
| Q5flag | 0 | 0 | 0 | 0 | 0 |
| Q6 | 0 | 0 | 1 | 0 | 1 |
| include | 224 | 179 | 62 | 128 | 593 |
| primary | 224 | 179 | 62 | 128 | 593 |
| cells | 224 | 179 | 63 | 128 | 594 |

## Excluded cells (Q2 or Q6)

- `v3cal_yolo12n_640_t4_hon_2400_r02`: Q6: unknown phase labels: done

## Q4 cells (kept, not primary): 0

None.

## Distribution of the QC measures

Finite values only; `n` counts them.

|  | n | min | p50 | p95 | max |
|---|---|---|---|---|---|
| max_residual_ms | 594 | 0.0808123 | 0.168936 | 0.254335 | 0.755539 |
| slip_scored_frac | 594 | 0 | 0 | 0.033379 | 0.10541 |
| ctx_read_frac_scored | 594 | 0 | 0.0103757 | 0.0139741 | 0.0212797 |
| pv_median_rel | 594 | 0 | 0 | 0 | 0 |
| sample_ratio | 594 | 0.998401 | 0.999993 | 1.00061 | 1.00491 |

## Phase labels in the traces

1247 distinct raw labels in 17 families. Outside the known vocabulary: `done`

| label | cells | distinct raw labels |
|---|---|---|
| baseline_A | 594 | 1 |
| baseline_B | 594 | 1 |
| bench:* | 180 | 1228 |
| concurrent_drain | 6 | 1 |
| decode | 48 | 1 |
| done | 1 | 1 |
| first:* | 594 | 4 |
| forward | 450 | 1 |
| gap | 180 | 1 |
| idle | 594 | 1 |
| load_weights | 594 | 1 |
| prefill | 48 | 1 |
| serve | 498 | 1 |
| sync:end | 594 | 1 |
| sync:start | 594 | 1 |
| train | 96 | 1 |
| warmup | 594 | 1 |
