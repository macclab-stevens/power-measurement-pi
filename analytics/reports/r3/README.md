# R3 and lockbox 2: summary (2026-11-24/25)

R3 is the revision trained on v1 + v2 + v3. The v3 collection was approved on 2026-11-24 and its plan is
`docs/review/next-collection.md`. **Lockbox 2 is R3's confirmatory test:** its predictions were frozen, committed and tagged
(`r3-frozen`, `lockbox2_predictions.csv`) before any of its cells was measured, and it was scored once with unchanged code.
The decisions are in `../../PROTOCOL.md` (amendment "R3 and lockbox 2").

## 1. What the layer hooks cost (v3 calibration, `hooks_calibration.csv`)

Hooks on and hooks off were measured in the same session: 7 models, 2400 MHz, 3 repeats each, nov2026 rig.

| model | energy saved without hooks | latency saved | power change |
|---|---:|---:|---:|
| mbv3 | **24.8 %** | 28.0 % | +4.4 % |
| effb0 | 14.3 % | 16.6 % | +2.8 % |
| yolo12n | 7.7 % | 9.1 % | +1.5 % |
| yolo11n | 7.0 % | 8.8 % | +1.9 % |
| yolov8n | 6.5 % | 7.6 % | +1.2 % |
| resnet18 | 4.2 % | 5.1 % | +0.9 % |
| yolo11s | 3.0 % | 4.2 % | +1.3 % |

- Every earlier measurement over-states deployment energy per image by 3–25 %. The excess grows with the number of
  module calls per frame relative to the compute.
- Without hooks, power is slightly higher: the cores no longer wait on single-threaded Python. So energy falls by less
  than latency.
- R3 models this through `hooks_on` features and an M4 cost term. With those, a model trained mostly on hooks-on data
  predicts hooks-off deployments.

## 2. Attention compute was missing from the MAC count

The layer table counts MACs per module, so attention's matrix products (functional calls) were missing. Swin and ViT were
counted at 0.64–0.66 of their published MACs, and yolo12n at 0.89. `tools/flop_count.py` counts the whole model with
`FlopCounterMode`, with a formula added for the CPU attention kernel. It reproduces ViT-B/16's 17.56 GMACs exactly. R3 gives
the uncounted part (`frac_attn`) its own feature and M4 term.

## 3. Mode A, leave-one-model-out over 15 models (76 configurations)

| | R1 | R2 | R3 |
|---|---:|---:|---:|
| AUTO energy MdAPE | 17.5 % | 13.5 % | **9.9 %** |
| AUTO energy MaxAPE | 4,397 % | 61 % | 40 % |
| 90 % coverage | 91 % | 74 % | 79 % |

- R3's best single candidate is M1Ridge, at 6.6 %.
- yolo12n is now 2.4 %. It was 22.9 % on lockbox 1, when no yolo12 model was in training.
- The hardest held-out models are the small classifiers (mbv3 32 %, effb0 25 %) and the attention models (ViT 22 %,
  Swin-T 21 %; M4Physics gets them to 2.5 % and 10 %).
- Splits where the base model is known: RANDOM 3.4 %, new variant 9.0 %, new clock 13.3 %. New clock is worse than R2 (3.8 %).

## 4. Lockbox 2, confirmatory (`lockbox2.md`)

Five never-collected models, each similar to a training model, with the hooks off:

| configuration | similar to | energy APE (AUTO) | inside 90 % |
|---|---|---:|---|
| resnet34:224 | resnet18 | 2.8 % | yes |
| effb1:224 | effb0 | 11.4 % | yes |
| mbv3l:224 | mbv3 | 13.6 % | no |
| yolo12m:640 | yolo12s | 15.9 % | yes |
| swins:224 | swint | 16.9 % | no |

| headline (stated before measurement) | result |
|---|---:|
| Mode A energy MdAPE, AUTO (M1Ridge), 5 configurations | **13.6 %** (MaxAPE 16.9 %, 3/5 inside the 90 % interval) |
| secondary: M4Physics | 8.3 % (MaxAPE 20.5 %, 4/5 inside) |
| mean power, AUTO | 1.4 % |
| calculator, serve phase, predicted duration (15 cells) | **5.9 %** median, 8.1 % mean |
| Mode L whole-cell, best rung (L5-cmos-trees) / L3-trees / lookup | **1.5 %** / 1.7 % / 6.2 % |
| Mode L non-YOLO models, L5-cmos-trees vs lookup (whole-cell / inference segment) | 1.8 % / 2.4 % vs 5.8 % / 8.3 % |

What it shows:
- **The live counter models now transfer across architectures.** On lockbox 1 they did not: non-YOLO 4.9 % against a
  4.5 % lookup, with training on YOLO only. With classifiers and attention models in the v3 training cells, they beat the
  lookup by about 4× on models they never saw.
- **The a priori predictor lands at 13.6 % on new models.** That is consistent with its leave-one-model-out estimate
  (9.9 % median; the five lockbox models sit in the harder part of that distribution).
- **Its intervals are too narrow.** 3 of 5 truths fell inside, against a nominal 90 %, which matches the 79 %
  leave-one-model-out coverage. This is the main open weakness.
- **Model choice.** The physics-structured M4 was the better choice here (8.3 %), but AUTO picked M1Ridge by inner error,
  as pre-registered. Five configurations cannot separate the two reliably.
- **Errors come from latency.** Power is predicted within about 1–5 %. The largest misses (Swin-S, yolo12m) are latency
  errors of 17–22 %.

## 5. Mode L, leave one model out over all 15 models with counters (`mode_l_r3.md`)

The R3 run holds out every vision model with counter cells: v2 YOLO and all v3 models. It took 3 h 20 min on the NUC.

| rung | whole-cell MdAPE | inference segment | forward R² |
|---|---:|---:|---:|
| L0 lookup | 4.74 % | 4.88 % | 0.59 |
| L3-trees | **1.19 %** | 1.64 % | 0.92 |
| L5-cmos-trees | 1.27 % | 1.79 % | 0.91 |
| L5-cmos (ridge) | 2.09 % | 1.93 % | 0.87 |

- **By family of the held-out model** (best tree rung, whole-cell):
  - YOLO v8/11: about 1.1 %;
  - attention models (including yolo12): about 1.2 %;
  - classifiers and CNNs: about 4 %, against the lookup's 7.2 %.
- **The classifiers are the hardest family for live accounting.** They have few configurations each and short frames.
- **L5's physics features help the ridge models but not the trees**, as in R2.
