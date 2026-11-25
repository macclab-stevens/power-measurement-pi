# Mode L, R2 against R1 (slip-corrected window labels, CMOS features, selection on the inference segment)

Exploratory R2 revision; the lockbox has been seen, so nothing here is a pre-registered result. Leave one YOLO base model out, QC-primary cells. Percent columns are percent; MdAPE is the median of |E_pred - E_true| / E_true over cells. **Whole** is the whole-cell energy, **inference** the forward (measured) phase only, **1 s block** the median over 10-window blocks. The forward-window MAE (W) and the R^2 are over the 100 ms windows of the held-out cells pooled together (the forward R^2 uses forward windows only; the all-window R^2 is inflated by the idle / active bimodality and is shown for comparison). R1 did not record window sums, so its R^2 and MAE are blank.

Runs: R2 `320dfac37ebb4500811c8cda5e4d4ab7`, R1 `acc4f1bf2b524febb2565e1d8fe3e79b`. R2 changes: window times corrected for sample-loss slips (cells collected before the align.py fix), `ctx_sw_rate` removed from every rung (identically 0), rungs `L5-cmos` / `L5-cmos-trees` (L3 features + llc_miss_rate, f_eff_mhz, vcore_eff, v2_cycles_rate, v2_instr_rate), grid points chosen on the median inference-segment APE instead of the whole-cell APE.

## Ladder: R2 with R1 beside it

| rung | n cells | whole MdAPE | inference MdAPE | 1 s block MdAPE | fwd MAE (W) | fwd R^2 | all-window R^2 | R1 whole | R1 inference | R1 1 s block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 161 | 5.72 | 5.76 | 7.03 | 0.874 | 0.496 | 0.814 | 5.77 | 5.76 | 7.02 |
| L1 | 161 | 6.69 | 13.58 | 17.72 | 1.838 | -0.626 | 0.189 | 6.69 | 13.58 | 17.79 |
| L1-cell | 161 | 5.64 | 12.23 | 16.98 | 1.737 | -0.571 | 0.016 | 5.64 | 12.23 | 16.98 |
| L2 | 161 | 5.30 | 7.39 | 5.98 | 0.893 | 0.495 | 0.832 | 5.07 | 7.04 | 5.38 |
| L3-ridge | 161 | 2.48 | 2.95 | 3.36 | 0.550 | 0.834 | 0.932 | 2.94 | 3.32 | 3.87 |
| L3-trees | 161 | 1.29 | 1.96 | 1.59 | 0.415 | 0.896 | 0.975 | 1.16 | 1.78 | 1.34 |
| L4-ridge | 161 | 2.44 | 3.00 | 3.33 | 0.547 | 0.834 | 0.933 | 3.08 | 3.78 | 3.93 |
| L4-trees | 161 | 1.31 | 2.01 | 1.59 | 0.413 | 0.897 | 0.975 | 1.22 | 1.94 | 1.60 |
| L5-cmos | 161 | 2.18 | 2.61 | 3.38 | 0.524 | 0.863 | 0.940 | - | - | - |
| L5-cmos-trees | 161 | 1.17 | 1.68 | 1.42 | 0.401 | 0.901 | 0.976 | - | - | - |

## Whole-cell error tail (R2)

| rung | MdAPE | P90 | MaxAPE |
|---|---:|---:|---:|
| L0 | 5.72 | 13.63 | 29.9 |
| L1 | 6.69 | 21.47 | 41.4 |
| L1-cell | 5.64 | 12.09 | 24.4 |
| L2 | 5.30 | 16.10 | 41.6 |
| L3-ridge | 2.48 | 6.42 | 16.4 |
| L3-trees | 1.29 | 3.55 | 16.5 |
| L4-ridge | 2.44 | 6.32 | 16.5 |
| L4-trees | 1.31 | 3.48 | 16.6 |
| L5-cmos | 2.18 | 7.10 | 17.0 |
| L5-cmos-trees | 1.17 | 3.52 | 16.1 |

## Grid points chosen per held-out base model (R2)

- `L2`: yolo11m: {"alpha": 10.0}; yolo11n: {"alpha": 0.1}; yolo11s: {"alpha": 100.0}; yolov8m: {"alpha": 10.0}; yolov8s: {"alpha": 10.0}
- `L3-ridge`: yolo11m: {"alpha": 1.0}; yolo11n: {"alpha": 0.01}; yolo11s: {"alpha": 0.1}; yolov8m: {"alpha": 0.01}; yolov8s: {"alpha": 0.01}
- `L3-trees`: yolo11m: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}; yolo11n: {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 50}; yolo11s: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}; yolov8m: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 200}; yolov8s: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}
- `L4-ridge`: yolo11m: {"alpha": 1.0}; yolo11n: {"alpha": 0.01}; yolo11s: {"alpha": 0.1}; yolov8m: {"alpha": 0.01}; yolov8s: {"alpha": 0.01}
- `L4-trees`: yolo11m: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}; yolo11n: {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 200}; yolo11s: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}; yolov8m: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}; yolov8s: {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 50}
- `L5-cmos`: yolo11m: {"alpha": 0.01}; yolo11n: {"alpha": 1.0}; yolo11s: {"alpha": 100.0}; yolov8m: {"alpha": 0.01}; yolov8s: {"alpha": 0.01}
- `L5-cmos-trees`: yolo11m: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 200}; yolo11n: {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 50}; yolo11s: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 200}; yolov8m: {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 50}; yolov8s: {"learning_rate": 0.1, "max_depth": 5, "max_iter": 200, "min_samples_leaf": 50}

## Size of the slip correction on the window labels

- Cells corrected: 90 (slip cells without >= 12 sync pulse events, left as stored: 0). Cells already corrected by the collection code are untouched.
- Largest time shift per cell: median 10.0 ms, maximum 44.8 ms.
- |dP| per 100 ms window (corrected axis against stored axis), over the 168634 windows of the corrected cells: the median over cells of the cell's median is 0.0022 W (largest cell median 0.1601 W), the median over cells of the cell's p99 is 0.182 W (largest 1.151 W), the largest single window 1.789 W.
- 152549 of those windows changed at all, 31765 by more than 0.1 W.
