# Measurement audit (2026-11-24)

These checks of the measurement chain were prompted by the external audit (`docs/review/`). They use 418 cells
(v1 224, v2 179, lockbox 15). The scripts are in `analytics/audit/`, run from `analytics/`.

## 1. The layer profiler slows every measured frame (the largest effect)

`audit/hookbench.py` ran on the Pi at 2400 MHz pinned, fan full, 4 threads, latency only. The hooks with layer events
were on and off in alternating blocks, 8 blocks × 15 frames (`audit/hookbench_2026-11-24.csv`).

| model | off ms | on ms | overhead | hooked modules | µs per module per frame |
|---|---:|---:|---:|---:|---:|
| mbv3 224 | 21.7 | 31.0 | +43 % | 208 | 45 |
| effb0 224 | 84.8 | 102.2 | +21 % | 336 | 52 |
| yolov8n 480 | 146.4 | 163.4 | +12 % | 223 | 76 |
| yolo11n 640 | 266.7 | 294.0 | +10 % | 317 | 86 |
| yolo12n 640 | 338.1 | 371.4 | +10 % | 463 | 72 |
| resnet18 224 | 79.4 | 78.0 | ≈0 (bimodal, noisy) | 67 | — |

The dataset latency of yolo11n at 640 px and 2400 MHz (about 291 ms) matches the hooks-on row. **Every
prediction in this repository is for instrumented inference.** A deployment without hooks is faster. Its energy
per image is lower too, by an amount that still has to be measured on the Monsoon (`docs/review/next-collection.md`
item 1). R2 adds the per-image module-call count as a feature, which models the overhead in the measured target. It
does not remove the overhead from the target.

## 2. Sample-loss slips and the sync-line fit

239 of 418 cells have Monsoon sample-loss slips. The single clock line through the start and end pulse trains turns
each step into a fake drift (`audit/slip_shear.py`, `audit/slip_shear_2026-11-24.csv`).

| quantity | median | p90 | max |
|---|---:|---:|---:|
| timestamp error in the cell, ms | 5.2 | 21.3 | 44.8 |
| energy-per-image change, % | 0.001 | 0.012 | 0.11 |
| 100 ms window power change, W (cell median) | 0.001 | 0.005 | 0.17 |
| 100 ms window power change, W (cell p99) | 0.09 | 0.43 | 1.22 |

Mode A is unaffected: the energy change sits far below the 0.57 % repeat noise. Mode L window labels move at phase
transitions. R2 builds windows on a corrected time axis (`timefix.py`), and `host/align.py` now corrects the slips
before the fit.

## 3. Supply harness

The resistance from the Monsoon to the PMIC input is fitted from V_mon − V_EXT5V against current during the
vcgencmd reads (`audit/cable.py`):

| rig | R (Ω) | active I (A) | I²R (W) | share of P |
|---|---:|---:|---:|---:|
| v1, 5.25 V | 0.179 | 1.76 | 0.55 | 6.0 % |
| v2, 5.40 V | 0.212 | 1.79 | 0.68 | 7.0 % |
| lockbox, 5.40 V | 0.213 | 1.75 | 0.65 | 6.9 % |

Subtracting I²R does not close the rig gap on identical configurations (`audit/rig_test.py`, yolo11n and yolov8n at
640 px, 1500/2000/2400 MHz, on both rigs). The mean absolute gap is 0.060 W raw and 0.073 W corrected. The
Monsoon-measured energy therefore stays the target; it includes about 6–7 % lead and connector loss.

## 4. Other checks

- **vcgencmd context reads (`ctx_read`):** 1.05 % of forward time, at −0.15 W against the neighbouring samples.
  Masking them would change mean forward power by 0.02 % (p99 0.12 %), so they are counted, not masked.
- **Core voltage:** median VDD_CORE is 0.7234 / 0.8745 / 0.9294 V at 1500 / 2000 / 2400 MHz, the same on both rigs
  within 1 mV. V²·f ratios are 1 : 1.95 : 2.64.
- **Calculator duration:** the closed-loop serve duration depends on latency. With the a priori duration the lockbox
  calculator error is 3.63 % median, against 1.05 % with the measured one (`reports/r2/lockbox_calculator_gap.md`).
- **Counters:** `context-switches` is 0 in every window (`perf_event_paranoid=2` forces user-only counting). It is
  dropped from Mode L.
