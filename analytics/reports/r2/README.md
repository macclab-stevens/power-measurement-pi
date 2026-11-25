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

- **Where the gains come from.**
  - **The worst-case tail.** It collapses because of the selection rule: balanced inner error, so each base model
    counts once. Under R1 the 49 YOLO rows chose a nearly unregularised ridge, which put EfficientNet-B0 at 45 times
    its energy. M1Ridge alone goes from 4,397 % to 136 % MaxAPE in S-LOMO; part of that is the R2 feature change.
  - **The S-LOMO median.** It improves mostly because of the new candidate: AUTO picks M4Physics in 7 of 9 outer
    folds.
  - **The seen-model splits (variant, clock, random).** These gain from the inner folds following the regime:
    random configurations when the base model is known.
- **M4Physics**, an additive latency cost (MACs, activation bytes, module calls, clock, threads) times V²f-based
  power, is the most robust to a new model: MaxAPE 51 %, R² on log 0.97. It is worse than the flexible candidates
  when the model is known.
- **The 90 % intervals are narrower but under-cover:** S-LOMO AUTO 74 % against R1's 91 %. Inner residuals of the
  selected candidate are optimistic. This is an open item; it has not been tuned after seeing it.

## Deployment calculator (`lockbox_calculator_gap.md`, `../deploy_validation.csv`)

A closed-loop serve phase (request, then a 200 ms sleep) lasts as long as the latency makes it. R1 used the measured
duration, which leaked the answer. The honest version predicts the duration as n · (latency + gap + overhead).

Training serve phases (226 cells, S-LOMO predictions, `../deploy_validation.csv`):

| scenario | median | mean | max |
|---|---:|---:|---:|
| measured duration (leaks) | 3.8 % | 6.0 % | — |
| predicted duration (honest) | **8.6 %** | 12.1 % | 45 % |

Lockbox serve phases (15 cells):

| scenario | R1 predictions median / mean / max | R2 predictions median / mean / max |
|---|---:|---:|
| measured duration (leaks) | 1.05 / 1.91 / 5.2 % | 1.15 / 3.34 / 9.0 % |
| predicted duration (honest) | 3.63 / 4.89 / 9.1 % | **1.89 / 8.51 / 20.2 %** |

R2 has the better median but a worse mean and maximum, both driven by yolo12n (see the lockbox section).

## Mode L (live 100 ms accounting), 5 YOLO models with counters (`mode_l_r2.md`)

Window labels now sit on a slip-corrected time axis. New features: the clock derived from the counters, V², and DRAM
misses per second. The grid is selected on the inference segment.

| rung | whole-cell MdAPE | inference segment | forward R² |
|---|---:|---:|---:|
| L0 lookup | 5.72 % | 5.76 % | 0.50 |
| L3-trees, R1 (frozen run) | 1.16 % | 1.78 % | — |
| L3-trees, R2 re-run | 1.29 % | 1.96 % | 0.90 |
| L5-cmos-trees, R2 | 1.17 % | 1.68 % | 0.90 |

The new features buy nothing on whole-cell energy (1.17 % against R1's 1.16 %) and little on the inference segment
(1.68 % against 1.78 %). The R1 → R2 change of L3-trees itself comes from two things: the slip-corrected labels and
selection on the inference segment. All-window R² (0.81–0.98) mostly measures the idle/active split; use forward R².

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
learned candidate under-predicts its latency (by 10–35 %). M4Physics's additive cost has no term for attention, so it
under-predicts the most. One model is one data point, but it sits at the edge of the stated goal (predict "a model like one we have
seen"). Without attention models in training, neither revision can be trusted on them.

Mode L on the lockbox:
- YOLO cells: L5-cmos-trees 0.5 % whole-cell and 1.1 % on the inference segment, against R1 L3-trees 0.42 % / 1.63 %.
- Classifiers, which no counter model saw: L5-cmos 4.4 %, against L0 4.5 % and R1 L3-trees 4.9 %.

Live counter models still do not transfer across families.

## What would settle it

The open questions need new data:
- a hooks-off calibration subset;
- a fresh lockbox, with predictions committed before measurement;
- attention and transformer models in training.

The plan is in `docs/review/next-collection.md`.
