> **R1, the pre-registered analysis (frozen).** The post-review revision R2 (2026-11-24, exploratory) is summarised in [`r2/README.md`](r2/README.md); the measurement checks are in [`measurement_audit.md`](measurement_audit.md). This file is unchanged apart from this note; an exact copy is in `r1_frozen/`.

# Worst-10 autopsy: Mode A, S-LOMO, energy per image

Source: `results.md` section 5. Split S-LOMO, target `energy_mJ`, candidate AUTO, master-seed run
`aa634083491146caa1898cb53dd5b2dc`. n = 10 of 58 held-out configurations. Written by the controller on
2026-11-23 from the run outputs (`~/pi-power-data/derived/mode_a/aa634083491146caa1898cb53dd5b2dc.parquet`), the
QC rows and the per-fold candidate table. None of the ten is a data-quality problem: every cell is QC
primary with no current throttle bits and alignment residuals below 0.6 ms.

## Ranks 1-3: EfficientNet-B0 at 1500 / 2000 / 2400 MHz (predicted about 45x the measured energy)

- Measured 0.76-0.97 J per image; AUTO predicted 33.6-43.7 J (APE 4310-4400 %).
- AUTO for this fold chose `M1Ridge` with alpha 0.01, the weakest regularisation in the grid, because
  it won the inner leave-one-model-out on the 8 training models. EfficientNet-B0's descriptor (many
  depthwise convolutions, a large activation footprint per MAC, a high share of SiLU/Sigmoid layers) lies
  outside every training model (novelty 5.01 against a fold threshold of 2.65), and a nearly
  unregularised linear model in log space extrapolates without bound there.
- The flag worked: all three predictions carry `no-similar-model`.
- Other candidates on the same fold: `M3Factored` 5.8 %, `M2Trees` 10.2 %, `B3LogLinear` 37.5 %. So the
  failure is the selection rule picking a fragile model, not a lack of information in the features.
- Not a blocker: the configuration is outside the stated domain ("seen something similar"), and the
  prediction is flagged. It is a negative result to report: the inner leave-one-model-out selection
  does not protect against extrapolation on a new architecture.

## Ranks 4-6: ResNet-18 at 1500 / 2000 / 2400 MHz (predicted about 1/20 of the measured energy)

- Measured 0.64-0.84 J; predicted 33-43 mJ (APE about 95 %, under-prediction).
- Same mechanism as ranks 1-3 in the opposite direction: `M1Ridge` alpha 0.01, novelty 5.55 against a
  threshold of 2.87, all flagged. `B1MacsProportional` (33.5 %) and `M2Trees` (114 %) also miss; ResNet-18 is
  a dense-convolution network much faster per MAC than the training models, so a MAC-based prior
  under- or over-shoots depending on the family it is anchored to.

## Rank 7: yolo11n FP16, 640 px, 2400 MHz (-39 %)

- FP16 is a single configuration. Under S-LOMO it is held out together with all yolo11n
  configurations, and no other FP16 configuration exists, so `is_fp16` is constant (0) in training and
  carries no information. The model predicts an FP32-like energy; FP16 really runs 1.29x slower than FP32
  on the Pi CPU (exploration report section 4.3). Flagged (novelty 2.59 > 2.28).
- Q3flag (3.4 % of measured frame time excluded for slips) does not explain a 39 % error.

## Rank 8: yolo11m, 960 px, 2400 MHz (-37 %)

- The largest configuration in the dataset (42 J per image). With yolo11m held out, training has 960 px
  only for yolo11n and yolo11s, so the model extrapolates in model size at the largest image size;
  yolo11m's latency grows faster than its MACs at 960 px (memory traffic), and the linear-in-log model
  under-predicts it. Not flagged (novelty 1.67 < 4.82): the novelty score is architecture-level and does
  not see image-size extrapolation. A known limitation of the flag.
- Q3flag (10.5 % of measured frame time excluded for slips) was checked: excluding the slip frames
  changes each cell's energy per image by at most 0.03 % (r01: 41,838 against 41,824 mJ raw), and the
  repeat spread is CV 0.54 %, so the error is not the slips.

## Rank 9: yolo11n, 320 px, 2 threads, 2400 MHz (-33 %)

- Holding out yolo11n removes most of the thread and image-size variation from training (the
  threads sweep exists only for yolo11n at 320/640/960 px and yolo11s at 640 px). The thread effect at
  320 px is therefore learned from yolo11s at 640 px only. Flagged (novelty 2.59 > 2.28).

## Rank 10: yolo11s, 320 px, 4 threads, 2400 MHz (+32 %)

- Over-prediction at the smallest image size: with yolo11s held out, the training models at 320 px are
  yolo11n and yolo11m, and the fit between them overshoots yolo11s's small-image efficiency. Not flagged.

## Conclusions

1. All ten are explained; none is a QC or data defect, so none blocks the freeze.
2. Ranks 1-6 (new architectures) are flagged and come from the selection rule choosing a weakly
   regularised ridge model. `M3Factored` is far more robust on those folds (S-LOMO new-architecture MdAPE
   16.9 % against AUTO 94.9 %). The pre-registered AUTO rule is kept; M3 and B3 are reported as declared
   secondary candidates for the lockbox.
3. Ranks 7-10 are inside the YOLO family and come from axes the held-out model alone carries (FP16, the
   threads sweep, the largest and smallest image sizes). The learning curve and the v2-only comparison
   point the same way: more distinct models and more varied configurations per model, not more repeats
   (repeat CV is 0.57 %).
4. The applicability flag catches architecture novelty but not image-size or configuration
   extrapolation (ranks 8 and 10 are unflagged).
