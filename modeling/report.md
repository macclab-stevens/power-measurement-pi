# Task 6 report: ablation + bootstrap LOMO eval

VDD_CORE rail only. Schema 3 only (`schema_version==3`, `_TAINTED_V2` skipped).
Energy `E_ABAB = P_delta_ABAB_W * t_per_call_ms` (mJ); `y=log(E_ABAB)`, `X=[logT,logB]`.
Baseline latency-only: `E_pred = mean_P_global * t_per_call_ms` (mean_P from train fold).
Full: per-class `logE = a + b_T*log(T_eff) + c_B*log(bytes)` (`fit_class` lstsq, n>=15 else pooled fallback; unseen test class -> pooled).
LOMO: train on rest, test on held-out family. Bootstrap n=1000 seed=0 resamples test indices w/ replacement; 95% CI (2.5/97.5 pct) for `delta = MAPE_full - MAPE_base`.
Inputs: `runs/v3_effb0 runs/v3_lm128 runs/v3_lm16 runs/v3_lm64 runs/v3_mbv3 runs/v3_yolo11n_640 runs/v3_yolov8n_640` + `modeling/scaling.json`.

## LOMO ablation (bootstrap CI for delta)

| family | n | MAPE_base | MAPE_full | delta | 95% CI | sMAPE_base | sMAPE_full | MAElog_base | MAElog_full |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 75 | 0.1754 | 0.4834 | +0.3080 | [0.2074, 0.4266] | 0.1858 | 0.4260 | 0.1873 | 0.4596 |
| yolo11n | 102 | 0.1862 | 0.8881 | +0.7019 | [0.3662, 1.1548] | 0.1928 | 0.4641 | 0.1944 | 0.5148 |
| effb0 | 109 | 0.4179 | 2.1113 | +1.6935 | [0.5295, 3.2607] | 0.3096 | 0.6297 | 0.3192 | 0.9350 |
| mbv3 | 96 | 0.3842 | 0.5081 | +0.1239 | [0.0266, 0.2254] | 0.2754 | 0.4262 | 0.2846 | 0.4507 |
| lm16 | 10 | 0.3414 | 0.4486 | +0.1072 | [-0.1415, 0.2912] | 0.2351 | 0.3390 | 0.2437 | 0.3499 |
| lm64 | 10 | 0.3632 | 0.4651 | +0.1019 | [-0.1639, 0.3096] | 0.2593 | 0.3559 | 0.2677 | 0.3683 |
| lm128 | 10 | 0.3383 | 0.5272 | +0.1889 | [-0.1025, 0.4648] | 0.2410 | 0.3630 | 0.2489 | 0.3774 |

## Thresholds (FAIL recorded honestly; exit stays 0 unless --strict)

- Ablation `MAPE_full < MAPE_base - 3pp` on >=4/6 families: **FAIL** (0/7 families; brief lists 7 families, threshold says 6 — scored on evaluated families).
- Linear `b_T in [0.8,1.2]`: **FAIL** (b_T=-0.09071662325222359).
- Attention `b_T in [1.7,2.3]`: **FAIL** (no *Attention* class in scaling.json/v3 runs (MiniGPTExplicit uses functional SDPA, no nn.MultiheadAttention leaf)).
- Overall: **FAIL**.

## Scaling coefficients (global fit, from modeling/scaling.json)

| class | n | b_T | c_B | rmse | fallback |
|---|---|---|---|---|---|
| AdaptiveAvgPool2d | 18 | 0.0000 | 0.2384 | 0.3503 |  |
| BatchNorm2d | 67 | -0.0473 | 0.8237 | 0.2991 |  |
| Concat | 8 | 0.2208 | 0.3450 | 0.3198 | pooled |
| Conv2d | 186 | 0.2320 | 0.4977 | 0.5111 |  |
| Dropout | 2 | 0.2208 | 0.3450 | 1.3759 | pooled |
| Embedding | 6 | 0.2208 | 0.3450 | 0.8772 | pooled |
| Hardsigmoid | 7 | 0.2208 | 0.3450 | 0.4420 | pooled |
| Hardswish | 10 | 0.2208 | 0.3450 | 0.6454 | pooled |
| Identity | 2 | 0.2208 | 0.3450 | 2.8827 | pooled |
| LayerNorm | 6 | 0.2208 | 0.3450 | 0.3732 | pooled |
| Linear | 21 | -0.0907 | 0.4572 | 0.2531 |  |
| MaxPool2d | 2 | 0.2208 | 0.3450 | 2.0085 | pooled |
| ReLU | 11 | 0.2208 | 0.3450 | 0.7579 | pooled |
| SiLU | 50 | 0.1360 | 0.3168 | 0.6817 |  |
| Sigmoid | 7 | 0.2208 | 0.3450 | 0.3490 | pooled |
| StochasticDepth | 5 | 0.2208 | 0.3450 | 3.1852 | pooled |
| Upsample | 4 | 0.2208 | 0.3450 | 1.6737 | pooled |

## Notes
- No missing values in scored rows; `t_eff,bytes,E>0` enforced; MAPE denom guarded at 1e-12.
- Baseline sees `t_per_call_ms` directly (E=P*t, P varies little), full sees only `(t_eff,bytes)` — latency-only is expected to be strong; a FAIL here means scaling features do not beat cheat-adjacent latency, not a harness bug.
