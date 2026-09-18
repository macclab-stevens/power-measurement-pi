# LM sub-bin phase-anchoring proof (Task C)

Date UTC 2026-09-18. Branch `sdd/caveats` BASE `2573451`. Repo `/home/macc2026/Desktop/pi_power_measurement` via ssh. Torch pinned `torch==2.14.0`; only `uv run --with pytest --` / `uv run`, never `uv add`; no sudo.

Method: `targets.build_phase_targets` per LM run (E_mJ=P_W*t_ms, dur first-to-last sample). LOMO via Task B `evaluate.lomo` unchanged (per-phase-type ridge + LOTO hill-climb, forward MAPE, bootstrap forward-only n=1000 seed 0, fallbacks, --strict wiring kept). No test-peek (train build+fit before te loop; fit never touches dur/samples power beyond E target; no weights). R4 LM report-only <=15% informational. No model-fit changes (only `targets.py` additive `is_valid_phase_target` + test).

## 1. Evidence table - per LM run x phase (prefill/decode) + counts

| run | phase | dur_s | n_samples | P_mean_W | E_mJ |
|---|---|---|---|---|---|
| v3_lm16 | prefill | 1.280115 | 160 | 2.5608 | 3278.17 |
| v3_lm16 | decode | 0.066313 | 14 | 3.8998 | 258.60 |
| v3_lm64 | prefill | 1.273175 | 156 | 2.5267 | 3216.89 |
| v3_lm64 | decode | 0.039015 | 8 | 4.2562 | 166.05 |
| v3_lm128 | prefill | 1.277591 | 153 | 2.5500 | 3257.89 |
| v3_lm128 | decode | 0.029061 | 6 | 4.6362 | 134.73 |

Counts over 78 LM segments (26/run x 3 runs): n==1 phases = 0; dur_s==0 = 0; E_mJ==0 = 0. Verdict: sane targets - E>0 wherever n>=2 (all prefill/decode n>=6, E 134-3278 mJ). No failure.

## 2. Phase-duration histogram summary (sample dt, sub-bin check, 100ms grid)

- v3_lm16: prefill n=160 dur=1280.1ms dt mean/med/min/max = 8.05/5.93/3.88/48.70ms; decode n=14 dur=66.3ms dt 5.10/4.17/4.03/12.81ms; decode span <100ms bin = True (sub-bin, would be split/merged by grid bins - phase-anchoring required). Prefill ~1.27s spans ~13 grid bins.
- v3_lm64: prefill n=156 dur=1273.2ms dt mean/med/min/max = 8.21/5.90/3.88/42.66ms; decode n=8 dur=39.0ms dt 5.57/5.19/4.10/10.41ms; decode span <100ms bin = True (sub-bin, would be split/merged by grid bins - phase-anchoring required). Prefill ~1.27s spans ~13 grid bins.
- v3_lm128: prefill n=153 dur=1277.6ms dt mean/med/min/max = 8.41/6.22/3.87/43.78ms; decode n=6 dur=29.1ms dt 5.81/5.00/4.21/10.05ms; decode span <100ms bin = True (sub-bin, would be split/merged by grid bins - phase-anchoring required). Prefill ~1.27s spans ~13 grid bins.

Grid 10Hz bins would cut prefill across ~13 bins and swallow each decode whole inside one bin (29/39/66ms); phase-anchored targets keep prefill (1.27s/3.2k mJ) separate from decode (0.03-0.07s/0.13-0.26k mJ). Sound.

## 3. Edge cases (S2): n==1 / dur==0 / single-bin - NOT observed, explicit keep+guard

- Observed in LM data: none (0 n==1, 0 dur==0, 0 E==0 over 78 segments). Per brief 'fix iff observed': NO behavior fix to `build_phase_targets` (E=P_mean*dur_ms, n==1 -> E==0/dur==0 kept).
- Explicit policy (code + tested): new `targets.is_valid_phase_target(row)` = n>=2 and E>0 and dur>0. Train `evaluate._build_train_items` drops E<=0 silently (verified in source); eval `forward_mape` raises ZeroDivisionError on E==0 (verified live). Never hit on LM data (min n=6 decode lm128). Test `test_lm_phase_targets_sane` pins: synthetic 6-sample decode (25ms span, 4.5W) -> E=112.5mJ sane valid; 1-sample -> E==0/dur==0 invalid.
- dur_s==0 bins / single-bin phases: same case as n==1 (only _finalize path to dur 0); none observed; policy covers.

## 4. LOMO LM verdict (Task B machinery, no refit changes)

| family | n | baseMAPE | fullMAPE | base_oracle | full_durnorm | ci95 | fb | level | dur_ratio (t_prior->fwd_actual) |
|---|---|---|---|---|---|---|---|---|---|
| lm16 | 26 | 23.9065 | 0.2207 | 3.2307 | 0.8816 | [0.0199,0.4216] | 0 | 0 | 0.1519 (4.43->0.67s) |
| lm64 | 26 | 36.8442 | 0.0697 | 5.0485 | 0.8418 | [0.0152,0.1241] | 0 | 0 | 0.1479 (4.44->0.66s) |
| lm128 | 26 | 45.1548 | 0.2759 | 6.2576 | 0.8141 | [0.0137,0.5381] | 0 | 0 | 0.1473 (4.44->0.65s) |

Claim rule (brief S3): LM proof requires absolute full MAPE small AND robust to durnorm; else 'phase-anchoring sound, skill unproven at n=10 configs'.
- Absolute full MAPE: lm16 0.2207, lm64 0.0697, lm128 0.2759 - small vs base 23-45 and vs oracle 3.2-6.2 (full wins by 3-6 MAPE points absolute). Skill beyond duration.
- Robust to durnorm (E_pred vs E/dur_ratio): lm16 0.8816, lm64 0.8418, lm128 0.8141 - 3-12x worse than full (full predicts actual-scale, not train-scale; expected since dur_ratio~0.15). NOT robust per literal rule.
- CIs wide (lm128 [0.0137,0.5381], lm16 [0.0199,0.4216], lm64 [0.0152,0.1241]; 2 forward rows/test -> bootstrap variance), fallbacks 0, levels all 0, dur_ratio 0.147-0.152.
- **Verdict: phase-anchoring sound, skill unproven at n=10 configs.** Targets sane (E>0, sub-bin decode captured), full beats base+oracle on absolute scale, but durnorm sensitivity + wide CIs + n=3 LM runs (2 forward rows each) cannot carry a proof claim. Numbers above are the bound.

## 5. Kept / constraints

- No test-peek (sandbox order kept, LOTO inner valid subset of train). R4 LM report-only informational (vision wins 3/4 unchanged). Fallbacks counted (all LM 0). Both-arms x1000 (final CI forward-only n=1000; inner point-MAPE). --strict wiring untouched (verified Task B: STRICT:1 NOSTRICT:0). No fit-code changes. Tests: 11 passed (`pytest trace_forecast/ -q`). Torch pinned; `uv run` only.
