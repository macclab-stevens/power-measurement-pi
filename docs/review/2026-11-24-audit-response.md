# Response to the external audit of 2026-11-23

The audit (`2026-11-23-external-audit.md`, written by a Cursor agent, `main` at `89edafd`) makes seven headline claims
and ten recommendations. Its scripts ran on another machine and are not available here, so every claim below was
re-checked on the NUC against our own data (`~/pi-power-data`, 418 cells). The levers are in
`analytics/audit/` (CSV outputs alongside). Verdicts:

- **Confirmed**: the mechanism and the size reproduce.
- **Partly**: the mechanism is real but the size or the conclusion does not hold.
- **Not supported**: our data contradict it.

The frozen results (tag `analytics-frozen`, `reports/lockbox.md`) stay as they are: they are the record of what was
predicted before the lockbox was measured. Everything changed here is a **revision (R2)** and exploratory. Any
lockbox number for a revised model is post hoc, because the lockbox has been seen.

## 0. A finding the audit only touched on: the layer profiler slows every measured frame by 10–43 %

The audit lists the `LayerProfiler` hooks as an observer effect (§3.3.1) without measuring it. We measured it on the
Pi (2400 MHz pinned, fan full, 4 threads, latency only). In one process, the hooks with `EventLog` layer events (as
in every measured cell) were switched on and off in alternating blocks, 8 blocks × 15 frames
(`analytics/audit/hookbench_2026-11-24.csv`, script `tools/hookbench.py`).

| model | hooks off ms | hooks on ms | overhead | hooked modules | µs per module per frame |
|---|---:|---:|---:|---:|---:|
| mbv3 (224) | 21.7 | 31.0 | **+43 %** | 208 | 45 |
| effb0 (224) | 84.8 | 102.2 | **+21 %** | 336 | 52 |
| yolov8n (480) | 146.4 | 163.4 | +12 % | 223 | 76 |
| yolo11n (640) | 266.7 | 294.0 | +10 % | 317 | 86 |
| yolo12n (640) | 338.1 | 371.4 | +10 % | 463 | 72 |
| resnet18 (224) | 79.4 | 78.0 | ≈0 (noisy, bimodal 72/86 ms) | 67 | — |

The hooked yolo11n latency (294 ms) matches the dataset (about 291 ms per frame), so **the dataset measures
instrumented inference**. A plain deployment of the same model runs about 10 % faster for YOLO and up to 43 % faster
for small mobile networks. Energy per image falls by somewhat less, since power during the Python hooks is lower.
This is a bias in what we predict, not an error of the predictor, and it is the largest issue found.

Actions:
1. Add a `--layer-hooks off` switch to `src/run.py`. The static features already come from `tools/static_features.py`
   on the NUC, so power cells do not need the hooks.
2. In Mode A, add the hooked-module count as an a priori feature: the overhead scales with it.
3. Propose a small hooks-off Monsoon collection to measure the instrumented-to-plain energy ratio (needs approval).

## 1. Mode A: selection rule and features (audit §2.2, A-1)

| claim | verdict | our check |
|---|---|---|
| Inner selection by pooled row MdAPE lets YOLO rows (49/58) dominate and picks alpha 0.01 on classifier folds | **Confirmed** | `mode_a._Search._best` pools rows; already the cause in `reports/autopsy.md` |
| Rank deficiency (`frac_macs_*` sum to 1; `log_weight_bytes = log_params + ln 4` for FP32) | **Confirmed**, but it is a symptom | Ridge is well defined under collinearity; the failure is the tiny alpha chosen by the selection rule, not the rank |
| `rig_nov` is 100 % confounded with architecture | **Not supported** | yolo11n and yolov8n at 1500/2000/2400 MHz were measured on both rigs (`v1ref_*` cells), which identifies the rig term; the rig gap on those identical configurations is 0.06 W mean absolute (< 1 %) |
| MdAPE hides MAPE 246 % and MaxAPE 4,397 % | **Partly** | `results.md` reports MAPE with a 95 % CI next to MdAPE; the headline and summary quoted only MdAPE, and MaxAPE / P90 are not reported |
| 90 % intervals are about 5.3× wide | **Confirmed** | hi90/lo90 = 5.30 for every lockbox prediction; they are pooled inner-residual quantiles inflated by the classifier folds |
| 3 of 5 lockbox configurations have base models seen in training | **Confirmed** (by design) | the report says so, but not prominently enough |
| `P1_DecomposedCMOS` gives S-LOMO 10.4 % | **To verify** | the audit's code is unavailable; we implement our own physics-factored candidate and score it with nested S-LOMO |

Actions (R2):
- Change inner selection to a **model-balanced** metric: the mean over held-out base models of each model's median
  |log error|. Each base model counts once, whatever its row count.
- Drop `log_weight_bytes` and `frac_macs_other`; keep `rig_nov`.
- Add a priori physics features:
  - `v2f` = V_core(f)² · f, using the measured DVFS table 0.723 / 0.874 / 0.929 V at 1500 / 2000 / 2400 MHz;
  - `n_hooked`, the module count.
- Add a candidate `M4Physics`: latency × power.
  - Latency: ridge on the log graph size, `n_hooked`, threads, clock and image size.
  - Power: P_idle(f) plus a ridge on `v2f` and the operator mix.
  - Selection uses the balanced metric.
- Report MaxAPE, P90 APE, R² on log, and the median interval width (hi/lo) beside every MdAPE.

## 2. Deployment calculator duration leak (audit §2.3, A-4) — Confirmed

The lockbox serve phase is closed loop: a request, then a 200 ms sleep. Its wall duration therefore depends on
the latency being predicted. `lockbox_score.calculator` and `deploy.validate` pass the measured `serve_dur_ms`. With
the a priori duration `n · (t_pred + gap)`, the lockbox calculator error rises **from 1.05 % to 3.63 % MdAPE**
(mean 1.91 → 4.89 %), exactly the audit's figures.

Action: scenarios get two kinds.
- `duration_s`, for open-loop or wall-clock deployments: known a priori, no leak.
- `gap_s`, for closed-loop request-and-sleep: duration = n · (t_pred + gap + per-request overhead). The overhead is
  estimated from training serve cells only.

The calculator headline becomes the honest one.

## 3. Mode L metrics and features (audit §2.4, A-3)

| claim | verdict | note |
|---|---|---|
| Whole-cell APE dilutes errors (idle phases, sign cancellation) | **Confirmed in kind** | we report inference-segment and 1 s block errors beside it, but selection uses whole-cell APE |
| All-phase window R² is inflated by idle/active bimodality | **Confirmed in kind** | report R² on active windows only |
| Window-random CV leaks (lag-1 r ≈ 0.84) | **Agree; not our split** | our folds hold out whole base models; the ladder is useful exposition |
| `ctx_sw_rate` is identically 0 (`:u` counting) | **Confirmed** | 0 in every window; drop the feature (harmless but dead) |
| 1 Hz sysfs clock is stale under ondemand; f_eff = cycles / (4 · busy) is causal at 10 Hz | **Plausible** | implement and score |
| Rate-scaled features (LLC misses per second, V²·cycles) reduce non-YOLO lockbox error 7.4 → 5.3 % | **To verify** | 6 lockbox cells, post hoc; we score it with leave-one-YOLO-model-out first |

Actions (R2):
- Add the window features `f_eff_mhz`, `vcore_eff`, `v2_cycles_rate`, `v2_instr_rate` and `llc_miss_rate`; drop
  `ctx_sw_rate`.
- Add rung `L5-cmos` (ridge) and `L5-cmos-trees`.
- Inner selection on inference-segment APE instead of whole-cell APE.
- Report forward-window MAE and active-only R².

## 4. Sync-pulse clock fit shears timestamps after sample-loss slips (audit §3.1, B-1) — Confirmed; small effect

`host/align.py` fits one line through the start and end pulse trains. A sample-loss step between them becomes a fake
linear drift. Over the 239 slip cells, the step model gives timestamp errors of median 5.2 ms and at most 44.8 ms
(`analytics/audit/slip_shear_2026-11-24.csv`). Effect on what we predict:

- **Energy per image (Mode A):** median 0.001 %, at most 0.11 %. Repeat noise is 0.57 %, so it is negligible.
- **Mode L 100 ms window power:** median 0.001 W, p99 per cell 0.09–1.0 W at phase transitions. A small but real
  label error.

Actions:
- Fix `align.py`: insert the detected slip steps into the sample clock before fitting the line. Tests included.
- Apply the same step correction to the sample time axis when Mode L windows are built (R2), without rewriting the
  published traces.

## 5. Harness I²R loss (audit §3.2, A-2, B-2) — Partly

Regressing Monsoon V − PMIC EXT5V on current during the vcgencmd reads gives the round-trip resistance between the
Monsoon and the PMIC input: **0.18 Ω (v1) and 0.21 Ω (v2, lockbox)**. The audit's figures were 0.16 and 0.18 Ω. That
puts about 0.55–0.68 W, or 6–7 % of measured active power, in the leads, connectors and board input path.

The audit concludes that subtracting it unconfounds the two rigs. **Not supported:** on the identical yolo11n/yolov8n
configurations measured on both rigs, the mean absolute rig gap is 0.060 W raw and 0.073 W after subtracting I²R.

Decisions:
- Keep the Monsoon-measured energy as the target. It is what the meter measures, and the correction depends on a
  PMIC ADC of unknown gain.
- Document the measurement boundary.
- Recommend a short, thick harness for the next collection, and holding Vout fixed across campaigns.

## 6. Core-voltage knee (audit §3.2, A-1) — Confirmed

Median VDD_CORE in `context.csv` is 0.7234 / 0.8745 / 0.9294 V at 1500 / 2000 / 2400 MHz, the same on both rigs
within 1 mV. V²·f ratios are 1 : 1.95 : 2.64. V is a fixed function of the clock, so it is known a priori.

Action: `eda.md` gets a fit P = P_0 + k · V(f)² · f, which has two parameters and three points, so one degree of
freedom. The current free power-law fit has as many parameters as points.

## 7. Observer effects and counters (audit §3.3)

| claim | verdict | our check / action |
|---|---|---|
| `ctx_read` (vcgencmd) samples are not masked | **Confirmed, negligible** | covers 1.05 % of forward time; samples sit 0.15 W lower than their neighbours. Masking would change mean forward power by 0.02 % median (p99 0.12 %). `verify_cell.py` counts them deliberately; document the size rather than mask |
| `layers.py` assumes 4 bytes per parameter for FP16/INT8 | **Confirmed, no effect on the analytics** | `analytics/static.py` builds static features from FP32 layer tables only; fix `src/layers.py` (use `element_size()`) |
| `perf stat -I 100` adds about 1 % latency | **Plausible** | document; the hooks matter 10–40× more (§0) |
| Frame-0 profiler heat pulse | **Minor** | frame 0 is excluded from scoring; document |
| `perf_event_paranoid=2` hides kernel time | **Confirmed** | changing it needs root on the Pi: a recommendation for the next collection, not done here |

## 8. Out of scope for this revision

- **LM and training cells** (audit A-5, 144 cells). The headline target agreed for this programme is per-inference
  vision energy. Per-token LM energy is a new modelling task. Listed as next work.
- **Legacy `modeling/` and `trace_forecast/`.** These are superseded by `analytics/`. The `fam_idx`-as-scalar bug is
  noted in their READMEs and not fixed.
- **Hardware changes** (rewiring, `perf_event_paranoid`, 20 Hz sysfs, ≥ 20 base models). These are next-collection
  recommendations (`docs/review/next-collection.md`). Each needs approval and lab time.

## 9. Outcome of the R2 revision (2026-11-24)

The full numbers are in `analytics/reports/r2/README.md`. Against the audit's predictions:

| audit claim | audit figure | our R2 result |
|---|---:|---:|
| S-LOMO energy MdAPE of a physics-decomposed model | 10.4 % | M4Physics 12.6 %; AUTO 13.5 % (R1 17.5 %) |
| S-LOMO MAPE / MaxAPE | 13.6 % / — | M4Physics 14.3 % / 51 %; AUTO 17.7 % / 61 % (R1 246 % / 4,397 %) |
| lockbox energy MdAPE, all | 5.4 % | 3.4 % (AUTO; post hoc) |
| lockbox energy MdAPE, yolo12n | 2.7 % | **22.9 %** (worse than R1's 6.6 %); every candidate under-predicts yolo12n latency by 10–35 % |
| calculator without the duration leak | 3.6 % | 3.6 % with R1 predictions, 1.9 % median (8.5 % mean) with R2 |
| non-YOLO lockbox, inference-segment error of the CMOS Mode L | 5.3 % | 6.7 % (L5-cmos), against L3-trees 7.1 % and L0 9.0 % |

- The audit's diagnosis of Mode A selection was right, and fixing it removed the catastrophic extrapolations.
- Its yolo12n figure does not reproduce with our physics model. yolo12n is the only attention model, and an additive
  cost without an attention term under-shoots it.
- R2's 90 % intervals are narrower but under-cover (S-LOMO 74 %, against R1's 91 %). This is an open item.
