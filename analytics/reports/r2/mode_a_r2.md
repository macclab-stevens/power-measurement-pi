# Mode A, R2 against R1 (selection by the model-balanced inner error; R2 features; M4Physics)

Exploratory R2 revision; the lockbox has been seen, so nothing here is a pre-registered result. Percent columns are percent. MdAPE is the median, MAPE the mean, MaxAPE the maximum and P90 the 90th percentile of |pred - y| / y; R2log is 1 - SSE/SST of log y on log pred; width is the median hi90 / lo90 of the 90 % interval. R1 is the frozen protocol (pooled-MdAPE selection, R1 features, no M4).

## S-LOMO, energy per image: every candidate (58 held-out configurations)

| candidate | MdAPE | MAPE | MaxAPE | P90 | R2log | width | cov90 | R1 MdAPE | R1 MAPE | R1 MaxAPE | R1 width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0Median | 74.7 | 150.4 | 1081 | 355.1 | -0.173 | 42.67 | 0.88 | 74.7 | 150.4 | 1081 | 42.67 |
| B1MacsProportional | 33.2 | 39.5 | 105 | 74.7 | 0.612 | 8.95 | 0.83 | 33.2 | 39.5 | 105 | 8.95 |
| B2NearestModel | 28.9 | 51.7 | 431 | 66.7 | 0.760 | 7.55 | 0.93 | 28.9 | 51.7 | 431 | 7.55 |
| B3LogLinear | 15.8 | 28.9 | 172 | 54.5 | 0.866 | 4.73 | 0.88 | 15.8 | 28.9 | 172 | 4.73 |
| M1Ridge | 15.5 | 23.7 | 136 | 50.6 | 0.892 | 3.16 | 0.84 | 18.1 | 246.4 | 4397 | 2.35 |
| M2Trees | 23.3 | 36.0 | 234 | 79.7 | 0.846 | 3.94 | 0.83 | 24.7 | 34.3 | 215 | 4.03 |
| M3Factored | 14.9 | 27.6 | 205 | 60.8 | 0.879 | 3.04 | 0.81 | 15.9 | 19.9 | 96 | 2.38 |
| M4Physics | 12.6 | 14.3 | 51 | 25.7 | 0.971 | 1.68 | 0.86 | - | - | - | - |
| AUTO | 13.5 | 17.7 | 61 | 38.6 | 0.931 | 1.61 | 0.76 | 17.5 | 245.7 | 4397 | 2.38 |

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

## S-LOMO, energy, AUTO across seeds (R2)

| seed | MdAPE | MAPE | MaxAPE |
|---|---:|---:|---:|
| 20261122 | 13.5 | 17.7 | 61 |
| 20261123 | 13.5 | 17.7 | 61 |
| 20261124 | 13.5 | 17.7 | 61 |

## AUTO MdAPE (%) by split and target, R1 and R2

| split | target | R1 | R2 |
|---|---:|---:|---:|
| RANDOM | energy_mJ | 5.3 | 11.1 |
| RANDOM | latency_ms | 6.9 | 12.1 |
| RANDOM | mean_P_W | 2.5 | 1.7 |
| S-LCO | energy_mJ | 16.1 | 15.1 |
| S-LCO | latency_ms | 11.7 | 14.4 |
| S-LCO | mean_P_W | 7.0 | 2.0 |
| S-LOMO | energy_mJ | 17.5 | 13.5 |
| S-LOMO | latency_ms | 18.3 | 15.4 |
| S-LOMO | mean_P_W | 3.6 | 2.2 |
| S-LSO | energy_mJ | 19.2 | 13.2 |
| S-LSO | latency_ms | 19.7 | 14.8 |
| S-LSO | mean_P_W | 3.1 | 2.5 |
| S-LVO | energy_mJ | 9.9 | 11.4 |
| S-LVO | latency_ms | 11.4 | 17.7 |
| S-LVO | mean_P_W | 4.2 | 3.4 |
