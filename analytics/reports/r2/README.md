# R2 revision: summary (2026-11-24, post review, exploratory)

R2 is the analysis after the external audit (`docs/review/`). The R1 reports (pre-registered, with a lockbox scored
once) are one level up and in `../r1_frozen/`. Nothing in R2 is a confirmatory test: the lockbox had been seen when
R2 was designed. What changed and why is in `../../PROTOCOL.md` (amendment of 2026-11-24) and
`docs/review/2026-11-24-audit-response.md`.

## What every number here is about

The Pi ran every measured cell with per-layer profiler hooks. They add 10–12 % frame latency for YOLO, 21 % for
EfficientNet-B0 and 43 % for MobileNetV3 (`../measurement_audit.md` §1). The predictions are therefore for that
instrumented inference. A deployment without hooks will use less energy per image; measuring how much is the first
item of `docs/review/next-collection.md`.

## Mode A (a priori energy per image), 58 configurations (`mode_a_r2.md`)

| split (what is new at test time) | R1 AUTO MdAPE / MaxAPE | R2 AUTO MdAPE / MaxAPE | R2 M4Physics MdAPE |
|---|---:|---:|---:|
| S-LOMO (a base model) | 17.5 % / 4,397 % | **13.5 % / 61 %** | 12.6 % |
| S-LSO (a model size class) | 19.2 % / 41 % | 13.2 % / 56 % | 13.1 % |
| S-LVO (a variant: int8, fp16, pruned) | 9.9 % / 28 % | 6.6 % / 25 % | 11.4 % |
| S-LCO (a clock) | 16.1 % / 66 % | 3.8 % / 26 % | 15.1 % |
| RANDOM (nothing structural) | 5.3 % / 46 % | 2.3 % / 24 % | 10.6 % |

- **Most of the gain comes from the selection rule, not the new model.** Balanced inner error means each base model
  counts once. Inner folds follow the regime: leave-one-model-out for a new model, random configurations for a seen
  one. Under R1 the 49 YOLO rows picked a nearly unregularised ridge, which put EfficientNet-B0 at 45 times its
  energy.
- **M4Physics**, an additive latency cost (MACs, activation bytes, module calls, clock, threads) times V²f-based
  power, is the most robust to a new model: MaxAPE 51 %, R² on log 0.97. It is worse than the flexible candidates
  when the model is known.
- **The 90 % intervals are narrower but under-cover:** S-LOMO AUTO 74 % against R1's 91 %. Inner residuals of the
  selected candidate are optimistic. This is an open item; it has not been tuned after seeing it.

## Deployment calculator (`lockbox_calculator_gap.md`, `../deploy_validation.csv`)

A closed-loop serve phase (request, then a 200 ms sleep) lasts as long as the latency makes it. R1 used the measured
duration, which leaked the answer. The honest version predicts the duration as n · (latency + gap + overhead).

| lockbox serve energy, 15 cells | R1 predictions | R2 predictions |
|---|---:|---:|
| measured duration (leaks) | 1.05 % | 1.15 % |
| predicted duration (honest) | 3.63 % | 1.89 % median, 8.5 % mean |

## Mode L (live 100 ms accounting), 5 YOLO models with counters (`mode_l_r2.md`)

Window labels now sit on a slip-corrected time axis. New features: the clock derived from the counters, V², and DRAM
misses per second. The grid is selected on the inference segment.

| rung | whole-cell MdAPE | inference segment | forward R² |
|---|---:|---:|---:|
| L0 lookup | 5.72 % | 5.76 % | 0.50 |
| L3-trees (R1 best) | 1.26 % | 2.00 % | 0.89 |
| **L5-cmos-trees** | **1.11 %** | **1.67 %** | **0.90** |

The gain is small and has no confidence interval. All-window R² (0.81–0.98) mostly measures the idle/active split;
use forward R².

## Power against clock (`../eda.md` §2.1)

The power law `P0 + k f^alpha` had as many parameters as points and gave negative static power. With the measured
core voltage, `P = P0 + k V(f)^2 f` fits every forward-power series with **P0 = 3.29–3.65 W** and residuals of at most
0.056 W. Static and dynamic power are now identified; k ranges from 2.6 (mbv3) to 4.1 (yolov8m).

## Lockbox, scored again with R2 (post hoc, `lockbox_posthoc.md`)

| energy per image | R1 frozen | R2 AUTO |
|---|---:|---:|
| all 5 configurations | 7.4 % | 3.4 % |
| 3 configurations of seen models (yolov8n 480 px, resnet18, mbv3) | 9.0 % | **2.4 %** |
| 2 configurations of the new model yolo12n | **6.6 %** | 22.9 % |

**R2 is worse on the only new architecture.** yolo12n has attention blocks, and no training model does. Every
candidate under-predicts its latency (by 10–35 %). M4Physics's additive cost has no term for attention, so it
under-predicts the most. One model is one data point, but it sits at the edge of the stated goal (predict "a model like one we have
seen"). Without attention models in training, neither revision can be trusted on them.

Mode L on the lockbox:
- YOLO cells: L5-cmos-trees 0.44 % whole-cell, against R1 L3-trees 0.42 %.
- Classifiers, which no counter model saw: L5-cmos 4.4 %, against L0 4.5 % and R1 L3-trees 4.9 %.

Live counter models still do not transfer across families.

## What would settle it

The open questions need new data:
- a hooks-off calibration subset;
- a fresh lockbox, with predictions committed before measurement;
- attention and transformer models in training.

The plan is in `docs/review/next-collection.md`.
