# Mode L, R3: leave one base model out over every model with counters

Run `a3aaeacbd2484f21abad17ddc3887eed` (15 held-out base models: convnextt, effb0, maxvitt, mbv3, resnet18, swint, vitb16, yolo11m, yolo11n, yolo11s, yolo12n, yolo12s, yolov8m, yolov8n, yolov8s; 223 cells). Whole = whole-cell energy APE, inference = the forward segment, median over held-out cells.

## Per rung

| rung | whole MdAPE % | inference MdAPE % | 1 s block % | forward MAE W | forward R² |
|---|---:|---:|---:|---:|---:|
| L0 | 4.74 | 4.88 | 6.10 | 0.816 | 0.586 |
| L1 | 6.71 | 14.87 | 18.41 | 1.860 | -0.554 |
| L1-cell | 5.85 | 13.24 | 17.91 | 1.728 | -0.475 |
| L2 | 4.64 | 6.61 | 5.20 | 0.832 | 0.584 |
| L3-ridge | 2.40 | 2.63 | 3.23 | 0.571 | 0.839 |
| L3-trees | 1.19 | 1.64 | 1.29 | 0.365 | 0.919 |
| L4-ridge | 2.42 | 2.57 | 3.25 | 0.568 | 0.840 |
| L4-trees | 1.24 | 1.73 | 1.46 | 0.381 | 0.912 |
| L5-cmos | 2.09 | 1.93 | 2.69 | 0.511 | 0.873 |
| L5-cmos-trees | 1.27 | 1.79 | 1.54 | 0.384 | 0.913 |

## Per family of the held-out model (median %, whole / inference)

| rung | YOLO v8/11 whole | YOLO v8/11 inference | attention whole | attention inference | classifier/CNN whole | classifier/CNN inference |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 4.89 | 4.64 | 2.07 | 3.11 | 7.16 | 7.50 |
| L1 | 6.46 | 14.15 | 8.11 | 16.58 | 7.01 | 21.26 |
| L1-cell | 5.68 | 13.01 | 7.10 | 9.58 | 5.85 | 19.41 |
| L2 | 4.42 | 6.54 | 6.07 | 7.15 | 4.91 | 7.14 |
| L3-ridge | 2.08 | 2.34 | 1.68 | 2.01 | 3.35 | 5.82 |
| L3-trees | 1.08 | 1.55 | 1.16 | 1.36 | 3.92 | 5.25 |
| L4-ridge | 2.09 | 2.28 | 1.72 | 2.04 | 3.54 | 5.77 |
| L4-trees | 1.13 | 1.61 | 1.11 | 1.33 | 3.95 | 5.11 |
| L5-cmos | 1.88 | 1.88 | 1.46 | 1.06 | 3.81 | 6.06 |
| L5-cmos-trees | 1.20 | 1.73 | 1.27 | 1.53 | 4.11 | 5.73 |
