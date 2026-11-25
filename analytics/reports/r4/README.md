# R4: deployment energy on v4 data (2026-11-25)

R4 answers the six open flaws of the external audit (§7). The plan and the reasoning for each decision are in
`docs/review/2026-11-25-r4-plan.md`; the protocol amendment is in `../../PROTOCOL.md`.

**What is new:** the v4 collection on the Pi 5 (128 cells, all ok) re-measured every hooks-on training configuration
with the layer hooks off. It also adds a clock and size sweep, and it ran with kernel-visible counters
(perf_event_paranoid=1).

**The R4 target is deployment energy, measured.** Mode A trains and scores only on hooks-off cells: v3, v4, and
lockbox 2, now training data.
- 83 configurations, 20 base models, 185 cells (`mode_a_configs_r4.csv`).
- The static layer tables of the new configurations were built on the Pi 5. They match the NUC build exactly.

**Status:** R4 is exploratory until lockbox 3 is scored. Lockbox 3 is six never-collected models, pre-registered in
`host/matrix_v4.py` and measured only after the R4 predictions are frozen. Everything below is cross-validated on
training data.

Commands: `uv run python r4.py {table, mode-a, hooks, mode-l, summary, power-table}`.

## The six flaws

| # | flaw | what R4 did | result |
|---|---|---|---|
| 1 | hooks-on targets on pre-v3 data | re-measured every configuration hooks-off; trains and scores on hooks-off only | new-model energy MdAPE **3.9 %** (R3 9.9 %), MaxAPE 27 % (R3 40 %) |
| 2 | 90 % intervals cover ~80 % | conformal interval at the base-model level (`lo90_cf`, `hi90_cf`) | **94 %** inside on new models (R3-style intervals on the same data: 87 %), median width ±13 % |
| 3 | mbv3 / effb0 tail | removing the hooks confound; family pooling (M5Pooled) tested | mbv3 **9.4 %** (R3 32 %), effb0 **3.3 %** (R3 25 %); pooling adds nothing |
| 4 | harness I²R in the label | board label = Monsoon − R·∫I²dt; the physical fix (short thick leads) needs your hands | harness = **7.0 %** of power; board-label new-model energy MdAPE 4.2 % (AUTO) |
| 5 | kernel-blind counters | v4 collected at paranoid=1; Mode L kernel rungs on v4 | counters are live (about 390 context switches/s), but they do not improve the live model: L5 1.17 % without them, 1.56 % with them |
| 6 | AUTO vs M4Physics on lockbox 2 | ENS, STACK, AUTO_R4 compared on deployment data, not on lockbox 2's five points | AUTO keeps the job: M4Physics is the weaker model here (11.1 %) |

## 1. What the profiler cost (`hooks_by_clock.csv`)

There are 47 pairs: a hooks-off v4 or v3 configuration against its hooks-on twin on the same rig.
- The hooks added a median **4.3 %** of energy per image, and up to 25 % (mbv3).
- The overhead follows module calls per millisecond of compute (Spearman 0.88). It is largest for small images, small
  models and batch 1:
  - yolo11n: 320 px 18 %, 640 px 6.6 %, 960 px 4.3 %;
  - batch 8: 2.3 %;
  - yolo11m at 960 px: 0.1 %.
- **It does not depend on the clock** (yolo11n at 640 px: 7.7 / 6.5 / 6.6 % at 1500 / 2000 / 2400 MHz). The hook code
  is CPU work, so it scales with the clock like the inference itself.
- Power rises 0–4 % without the hooks, because the cores no longer wait on single-threaded Python.

## 2. Mode A on deployment energy (`mode_a_summary.csv`, `lomo_by_model_*.csv`)

AUTO is the predictor of record, with its R3 rule unchanged. Energy per image:

| split | AUTO MdAPE | MaxAPE | inside 90 %, conformal (R3-style) |
|---|---:|---:|---:|
| S-LOMO (new model) | **3.9 %** | 27 % | 94 % (87 %) |
| S-LSO (new YOLO size) | 4.6 % | 29 % | 91 % (91 %) |
| S-LCO (new clock) | 4.2 % | 34 % | 100 % (93 %) |
| S-LVO (new int8 / fp16 variant, 3 folds) | 12.0 % | 26 % | 80 % (40 %) |
| RANDOM (seen model) | 2.9 % | 26 % | 98 % (94 %) |

On S-LOMO, power is predicted within **2.4 %** and latency within **5.6 %**, so the errors are mostly latency, as in
R3. With the board label, S-LOMO energy is 4.2 % and power 2.4 % (AUTO). The leads take about the same share in every
configuration, so the label makes no difference to the error.

### Per held-out model (S-LOMO energy APE, median over its configurations)

| model | R3 | R4 |
|---|---:|---:|
| mbv3 | 32 % | 9.4 % |
| effb0 | 25 % | 3.3 % |
| vitb16 | 22 % | 1.9 % |
| swint | 21 % | 8.5 % |
| yolo12n | 2.4 % | 4.3 % |
| new in R4 training (lockbox 2 models) | – | resnet34 1.0, mbv3l 0.9, effb1 4.4, swins 8.5, yolo12m 10.6 |

The largest remaining errors are ConvNeXt-T (9.7 %), yolo12m (10.6 %) and the Swin models (8.5 %).

### Selector and pooling (flaws 3 and 6)

| candidate | S-LOMO | S-LCO | MaxAPE (S-LOMO) |
|---|---:|---:|---:|
| AUTO (= M1Ridge in most folds) | 3.9 % | 4.2 % | 27 % |
| AUTO_R4 (picks STACK in most folds) | 3.8 % | 7.6 % | 32 % |
| ENS (geometric mean M1 × M4) | 5.6 % | 7.9 % | 38 % |
| M4Physics | 11.1 % | 15.7 % | 56 % |
| M5Pooled (M4 + family offsets) | 10.7 % | 14.9 % | 59 % |

- **The R3 small-CNN tail was mostly the profiler's overhead.** It was largest exactly for those models (25 % and 14 % of
  energy). With deployment targets the tail is gone, and partial pooling over families has nothing left to fix.
- **On lockbox 2, M4Physics beat AUTO** (8.3 % against 13.6 %) on five configurations. On 83 deployment configurations it
  is three times worse. Biasing AUTO towards M4 would have fitted the lockbox.
- **AUTO_R4 ties AUTO on new models, but it is worse on new clocks and in the tail.** So the R3 rule stays, and the new
  selectors remain secondary outputs.

### Intervals (flaw 2)

The conformal interval at the base-model level reaches nominal coverage on new models: 94 % inside a nominal 90 %, with a
median hi/lo ratio of 1.30, about ±13 %. In the seen-model regimes it over-covers (S-LCO 100 %, RANDOM 98 %). Those
regimes have only 4–5 inner folds, which makes the finite-sample level (1 − α)(1 + 1/K) reach 1, so the interval
becomes the largest inner error. S-LVO has three folds and five configurations, too few to judge coverage.

## 3. Per-configuration table (`power_table_r4.csv`, `.md`)

One row per configuration. Measured on the Pi 5: power, repeat CV, board power and energy per image. Held-out
predictions: new-model, seen-model and live, with the new-model 90 % interval. Median absolute error of power: new-model
2.44 %, seen-model 1.85 %, live 1.10 % (64 configurations with v4 counter cells). Energy per image: new-model 3.94 %,
with 94 % inside the 90 % interval; seen-model 2.89 %.

## 4. Live model with kernel-visible counters (`mode_l_v4_summary.csv`, `mode_l_v4_by_family.csv`)

Leave one model out over the v4 cells only (15 base models). The kernel rungs add the context-switch and migration
rates (L3k-trees, L5k-trees). The headline is the inference segment.

| rung | inference segment | whole cell |
|---|---:|---:|
| L0 lookup (no counters) | 5.10 % | 5.21 % |
| L3-trees | 1.45 % | 0.98 % |
| L3k-trees (+ kernel rates) | 1.54 % | 1.12 % |
| **L5-cmos-trees** | **1.17 %** | **0.86 %** |
| L5k-trees (+ kernel rates) | 1.56 % | 1.07 % |

- **The kernel-visible counters are live and carry no power information.**
  - In v4 the context-switch rate is about 390 /s during inference and 10 /s at idle; it read 0 in every earlier cell.
  - Its rank correlation with forward-phase power is −0.01 (migrations −0.14).
  - Adding the two rates makes every tree rung slightly worse (L5 1.17 → 1.56 %), the usual cost of an uninformative
    feature in a small training set.
  - For single-process inference on the Pi 5, the kernel does too little to matter. Flaw 5 is fixed at the measurement
    level, with a negative result, and the R3 counter set stays.
- **On deployment cells the live model is at 1.2 % on the inference segment** (L5-cmos-trees), against 5.1 % for a
  lookup. That is about the R3 level (1.64 % inference segment, with hooks).
- **By family of the held-out model** (inference segment, L5-cmos-trees):
  - YOLO 0.8 %, MaxViT 0.3 %, ResNet 1.5 %;
  - ViT 3.7 %, Swin 4.1 %, ConvNeXt 4.3 %;
  - the mobile CNNs 5.5 %, and mbv3 is the hardest (9.3 %, against the lookup's 8.1 %);
  - with 21 ms frames, a 100 ms window spans several frames and the per-window counters average them out.
- **Power per configuration** (`power_table_r4.md`): the live model is within 1.10 % median, the seen-model a priori
  prediction within 1.85 % and the new-model one within 2.44 %. The repeat spread is 0.29 %.

## What is still open

- **Lockbox 3:** freeze the R4 predictions, tag `r4-frozen`, measure 18 cells (about 20 minutes), score once.
- **Harness:** short, thick leads to the 5 V and GND pins. Until then the board label is a correction, not a
  measurement.
- **Quantised variants (S-LVO):** only three int8 / fp16 variants of two models exist, too few to learn or to score
  variant transfer.
