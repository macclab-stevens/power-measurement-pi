# Phase-Conditioned Pi Energy Trace Forecast — Design Spec (v1)

**Status:** draft for user review (do not implement until approved)
**Date:** 2026-09-18 · **Repo:** `/home/macc2026/Desktop/pi_power_measurement` · **Base:** `master@e7b39fd` (v2 envelope merged, 13/13 green)
**Scope:** sampler+model phase trace forecast for **unseen** models. In-model-only and deployment-provisioning explicitly out of v1.

## 1. Problem

Given a PyTorch model + input spec + planned phase schedule + idle env, forecast the Pi 5 VDD_CORE power trace for the whole run **before measuring it**, at **10 Hz (100 ms bins)**, generalizing to unseen model families and unseen seq/imgsz.

- Input (known pre-run): `model_id` (arch fingerprint + dry-trace sums, §3), `input_spec` (`imgsz` or `seq`), `phase_plan` (ordered phases + expected durations from `run.py` logic), `env0` (`temp0_C, freq_MHz, throttle`, threads=4 fixed).
- Output: per-bin `t_s, phase, P_mean_W, P_lo, P_hi, E_cum_mJ` + per-phase `E_mJ` + 95% bands.
- Success (binding): per-phase `E` MAPE **≤5% on ≥4/7 LOMO families**, CIs excluding latency-only baseline, bin coverage **≥90%**. Else ship bound + residual attribution (no test tuning to force GREEN).

## 2. Architecture

New `trace_forecast/` subsystem beside `modeling/` (no measurement changes until proven needed):

1. `dry_trace` — one eager forward with existing `LayerProfiler` (hooks only, sampler off) → `sum_macs/sum_bytes/sum_teff, n_leaves, op_mix, T, d_model proxy, depth`. Random fixtures, never test power.
2. `baseline_model` — `leakage(temp)` fit on idle/gap rows (`P = a·exp(b·T) + c`), plus per-run drift slope.
3. `phase_delta_model` — per-phase-type ridge on `log ΔP` (reuses T5 `fit_class`): `Δ_forward/prefill/decode` from `sum(E_ABAB)/t_fwd`; `Δ_bench:<cid>` = lookup of `perop P_delta_ABAB` + drift (no regression).
4. `stitcher` — phase order `warmup→baseline_A→forward/prefill/decode→baseline_B→bench×N+gap` with durations (test durations are targets, never inputs: `t_fwd` from train prior × planned frames).
5. `hillclimb` — coordinate ascent on validation only (§5).

## 3. Features (hill-climb start point)

Base (locked): `log(sum_macs), log(sum_bytes), log(sum_teff), op_mix (class counts/total)`, env `temp0, freq_MHz (assert 500–3000), threads=4`. Queued rounds: `AI=log(macs/bytes)` + `freq·temp` (R1) → `attn_flops=2·B·T²·d_head` (R2, only if attention residual dominates — T6 gap) → per-family random intercept (R3, board/PSU batch) → per-run temp-drift slope (R4). R5 stops or regularizes harder. Each round must improve locked-validation MAPE ≥0.5pp or revert.

## 4. Data

Train/test exclusively `runs/v3_*` (`schema_version==3`, skip `_TAINTED_V2`, skip SKIPPED): `v3_yolov8n_640, v3_yolo11n_640, v3_effb0, v3_mbv3, v3_lm16/64/128` (n≈412 configs; `samples.csv` 10 Hz-binned for trace targets). Old v2 runs never in train (non-goal 5). `E_ABAB = P_delta_ABAB·t_per_call`; test `samples.csv` unreadable during fit (enforced by sandbox test).

## 5. Eval + error contract

LOMO train-6→forecast-held-out 10 Hz trace. Primary per-phase `E` MAPE/sMAPE; secondary per-bin RMSE + coverage (target ≥90%) + calibration slope. Baselines: (i) latency-only `mean_P_train·t`, (ii) global phase-type mean. Bootstrap n=1000 CIs on ΔMAPE. 5% applies to per-phase E (billable number); per-bin RMSE expected 8–12% from ADC noise alone — not hill-climbed past the sensor. Aborts (named, never silent): freq out of range, `throttle&0x1`, drift>0.5W, `schema!=3`, tainted input, empty phase window (counted exclusions). Unseen bench `config_id` → pooled-class prediction + `fallback:true` (counted).

## 6. Testing (TDD)

`test_trace_schema` (10 Hz, cumulative E monotonic), `test_no_leak` (fit sandbox cannot read test power), `test_stitch_order`, `test_bands_cover` (synthetic drift inside bands). Full suite (`tests/ + modeling/ + trace_forecast/`) green before any claim.

## 7. Non-goals (v1)

Wall power (VDD_CORE only), fused backends (inductor/TFLite/ONNX), NPU/GPU, threads≠4/governor≠ondemand without re-collection, >2×T/HW extrapolation (bound only), native 190 Hz per-sample prediction, tainted v2 in train.

## 8. Risks

7 runs → dynamics unlearnable (hence template, not ARX, as v1); `P≈3–4W` saturation repeats T6 0/7 unless attention/bytes features move residuals (hill-climb proves which); temp↔power simultaneity handled via pre-phase `env0` + train drift prior only.

## 9. Handoff

Bundle: `handoff_phase_trace_v1/` (SPEC.md = this file, MANIFEST.md, FIGS/ v2 PNGs, DOCS/ report+scaling+NON_GOALS, LEDGER.md, NEXT_SESSION.md). Next: user reviews spec → adversarial red-team → writing-plans → SDD implement.
