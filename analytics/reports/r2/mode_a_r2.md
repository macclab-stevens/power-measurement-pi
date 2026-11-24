# Mode A, R2 against R1 (balanced selection on inner folds that mimic the outer split; R2 features; M4Physics)

Exploratory R2 revision; the lockbox has been seen, so nothing here is a pre-registered result. Percent columns are percent. MdAPE is the median, MAPE the mean, MaxAPE the maximum and P90 the 90th percentile of |pred - y| / y; R2log is 1 - SSE/SST of log y on log pred; width is the median hi90 / lo90 of the 90 % interval. R1 is the frozen protocol (pooled-MdAPE selection, R1 features, no M4).

## S-LOMO, energy per image: every candidate (58 held-out configurations)

| candidate | MdAPE | MAPE | MaxAPE | P90 | R2log | width | cov90 | R1 MdAPE | R1 MAPE | R1 MaxAPE | R1 width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0Median | 74.7 | 150.4 | 1081 | 355.1 | -0.173 | 68.96 | 0.90 | 74.7 | 150.4 | 1081 | 42.67 |
| B1MacsProportional | 33.2 | 39.5 | 105 | 74.7 | 0.612 | 15.12 | 0.91 | 33.2 | 39.5 | 105 | 8.95 |
| B2NearestModel | 28.9 | 51.7 | 431 | 66.7 | 0.760 | 10.86 | 0.95 | 28.9 | 51.7 | 431 | 7.55 |
| B3LogLinear | 15.8 | 28.9 | 172 | 54.5 | 0.866 | 6.93 | 0.93 | 15.8 | 28.9 | 172 | 4.73 |
| M1Ridge | 15.5 | 23.7 | 136 | 50.6 | 0.892 | 2.87 | 0.81 | 18.1 | 246.4 | 4397 | 2.35 |
| M2Trees | 23.3 | 36.0 | 234 | 79.7 | 0.846 | 4.69 | 0.79 | 24.7 | 34.3 | 215 | 4.03 |
| M3Factored | 14.9 | 27.6 | 205 | 60.8 | 0.879 | 2.65 | 0.79 | 15.9 | 19.9 | 96 | 2.38 |
| M4Physics | 12.6 | 14.3 | 51 | 25.7 | 0.971 | 1.59 | 0.84 | - | - | - | - |
| AUTO | 13.5 | 17.7 | 61 | 38.6 | 0.931 | 1.59 | 0.74 | 17.5 | 245.7 | 4397 | 2.38 |

AUTO's choice per outer fold (R2): M4Physics x7, M3Factored x2. R1: M1Ridge x6, M3Factored x3.

## S-LOMO, energy: MdAPE per held-out base model (%)

| base model | AUTO R2 | M4Physics R2 | AUTO R1 |
|---|---:|---:|---:|
| effb0 | 9.2 | 9.2 | 4340.8 |
| mbv3 | 24.7 | 24.7 | 16.9 |
| resnet18 | 60.7 | 6.7 | 94.9 |
| yolo11m | 21.1 | 21.1 | 22.8 |
| yolo11n | 12.2 | 12.2 | 17.5 |
| yolo11s | 15.4 | 15.4 | 13.8 |
| yolov8m | 18.9 | 5.5 | 18.1 |
| yolov8n | 5.1 | 5.1 | 11.5 |
| yolov8s | 10.5 | 10.5 | 2.4 |

## Energy per image: AUTO and M4Physics on every split

| split | candidate | revision | MdAPE | MAPE | MaxAPE | cov90 | width |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-LOMO | AUTO | R2 | 13.5 | 17.7 | 61 | 0.74 | 1.59 |
| S-LOMO | AUTO | R2 before inner-fold fix | 13.5 | 17.7 | 61 | 0.76 | 1.61 |
| S-LOMO | AUTO | R1 | 17.5 | 245.7 | 4397 | 0.91 | 2.38 |
| S-LOMO | M4Physics | R2 | 12.6 | 14.3 | 51 | 0.84 | 1.59 |
| S-LOMO | M4Physics | R2 before inner-fold fix | 12.6 | 14.3 | 51 | 0.86 | 1.68 |
| S-LSO | AUTO | R2 | 16.4 | 17.5 | 42 | 0.14 | 1.17 |
| S-LSO | AUTO | R2 before inner-fold fix | 13.2 | 14.6 | 56 | 0.73 | 1.53 |
| S-LSO | AUTO | R1 | 19.2 | 18.4 | 41 | 0.80 | 1.99 |
| S-LSO | M4Physics | R2 | 13.1 | 14.8 | 56 | 0.86 | 1.47 |
| S-LSO | M4Physics | R2 before inner-fold fix | 13.1 | 14.8 | 56 | 0.82 | 1.53 |
| S-LVO | AUTO | R2 | 6.6 | 10.4 | 25 | 0.80 | 1.31 |
| S-LVO | AUTO | R2 before inner-fold fix | 11.4 | 19.7 | 50 | 0.80 | 1.68 |
| S-LVO | AUTO | R1 | 9.9 | 11.7 | 28 | 1.00 | 1.82 |
| S-LVO | M4Physics | R2 | 11.4 | 19.7 | 50 | 0.80 | 1.69 |
| S-LVO | M4Physics | R2 before inner-fold fix | 11.4 | 19.7 | 50 | 0.80 | 1.68 |
| S-LCO | AUTO | R2 | 3.8 | 7.3 | 26 | 0.77 | 1.26 |
| S-LCO | AUTO | R2 before inner-fold fix | 15.1 | 18.1 | 61 | 0.41 | 1.23 |
| S-LCO | AUTO | R1 | 16.1 | 18.8 | 66 | 0.98 | 6.55 |
| S-LCO | M4Physics | R2 | 15.1 | 18.1 | 61 | 0.48 | 1.30 |
| S-LCO | M4Physics | R2 before inner-fold fix | 15.1 | 18.1 | 61 | 0.41 | 1.23 |
| RANDOM | AUTO | R2 | 2.3 | 4.3 | 24 | 0.88 | 1.25 |
| RANDOM | AUTO | R2 before inner-fold fix | 11.1 | 13.7 | 49 | 0.90 | 1.69 |
| RANDOM | AUTO | R1 | 5.3 | 9.4 | 46 | 0.98 | 2.19 |
| RANDOM | M4Physics | R2 | 10.6 | 12.9 | 49 | 0.88 | 1.68 |
| RANDOM | M4Physics | R2 before inner-fold fix | 11.1 | 13.7 | 49 | 0.90 | 1.69 |

## AUTO MdAPE (%) by split and target, R1 and R2

| split | target | R1 | R2 |
|---|---:|---:|---:|
| RANDOM | energy_mJ | 5.3 | 2.3 |
| RANDOM | latency_ms | 6.9 | 3.9 |
| RANDOM | mean_P_W | 2.5 | 1.6 |
| S-LCO | energy_mJ | 16.1 | 3.8 |
| S-LCO | latency_ms | 11.7 | 6.0 |
| S-LCO | mean_P_W | 7.0 | 5.5 |
| S-LOMO | energy_mJ | 17.5 | 13.5 |
| S-LOMO | latency_ms | 18.3 | 15.4 |
| S-LOMO | mean_P_W | 3.6 | 2.2 |
| S-LSO | energy_mJ | 19.2 | 16.4 |
| S-LSO | latency_ms | 19.7 | 18.8 |
| S-LSO | mean_P_W | 3.1 | 2.0 |
| S-LVO | energy_mJ | 9.9 | 6.6 |
| S-LVO | latency_ms | 11.4 | 4.1 |
| S-LVO | mean_P_W | 4.2 | 5.5 |
