---
cursor:
  subagentId: "bc-1b902105-2d88-5d70-86be-554ed1c4c197"
---

# Scientific Audit of `pi-power-analysis` (`power-measurement-pi`): Dataset, Power Prediction Models, and Hardware Data Collection

**Audit Timestamp**: November 23, 2026, 8:58 PM EST (`America/New_York`, `UTC-5`)
**Audited Repository**: [`macclab-stevens/power-measurement-pi`](https://github.com/macclab-stevens/power-measurement-pi) (`main` at `89edafd`, November 23, 2026, 6:44 PM EST)
**Workspace Path**: `C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi` (kept 100% clean)
**Reproducible Audit Levers & Scratch Outputs**: `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\`

---

## 1. Executive Summary & Guiding Principle (`build-the-lever`)

### 1.1 Application of `/principle-build-the-lever`
- **Located at**: `C:\Users\ishan\.cursor\plugins\cache\cursor-public\pstack\2dd5c0ccc5b207dc23a69759871a35954082cbf7\skills\principle-build-the-lever\SKILL.md` (mirrored under `9717366`).
- **Core Rule**: *The bar for building a tool is triviality, not repetition.* Whenever verification, extraction, or modeling is non-trivial, write a small, deterministic, safe-to-rerun script rather than inspecting ad hoc or building bloated abstractions.
- **How Applied Here**: Rather than accepting `analytics/reports/*.md` at face value or spot-checking cells manually, we built five self-contained, deterministic Python audit levers under `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\` (`materialize_datasets.py`, `audit_collection_and_physics.py`, `audit_pmic_and_cable.py`, `audit_mode_a_experiments.py`, `audit_mode_l_experiments.py`) that re-evaluated all **418 Monsoon cells** (`224 v1` + `179 v2` + `15 lockbox`), **215,399 100 ms counter windows**, and **14 candidate predictors** without touching a single byte of the repository checkout.

### 1.2 Repository & Dataset Topology
- No separate `pi-power-analysis` git repository exists locally under `C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P` or on GitHub (`macclab-stevens` / `IshanAryendu`). `"pi-power-analysis"` is the Cursor Project name; the active modeling package (`pyproject.toml` name `pi-power-analytics`) lives inside [`macclab-stevens/power-measurement-pi`](https://github.com/macclab-stevens/power-measurement-pi) under `analytics/` alongside two earlier modeling generations (`modeling/` and `trace_forecast/`) and the Pi/EliteBook measurement stack (`src/`, `host/`, `tools/`). *(Note: `macclab-stevens/ML2P` is a separate private repository not mounted on this self-hosted worker and was not accessed.)*
- All three Monsoon HVPM hardware-metered datasets live on dedicated branches of `macclab-stevens/power-measurement-pi`:
  1. **`v1` (`origin/dataset-2026-10-matrix`, `data/monsoon_matrix_2026-10`)**: `224` cells (`oct2026` rig, Monsoon `Vout = 5.25 V`, Linux `6.18.50`, no `perf` counters; vision + LM + training).
  2. **`v2` (`origin/dataset-2026-11-v2`, `data/monsoon_v2_2026-11`)**: `179` cells (`nov2026` rig, Monsoon `Vout = 5.40 V`, Linux `6.18.60`, `161` QC-primary YOLO `perf` counter cells + `18` `ref`/exempt cells; `3.21 GB` of 5 kHz Parquet traces).
  3. **`lockbox` (`origin/dataset-2026-11-lockbox`, `data/monsoon_lockbox_2026-11`)**: `15` cells across `5` configurations (`yolo12n:640`, `yolo12n:320`, `yolov8n:480`, `resnet18:224`, `mbv3:224`; `3` repeats each; `198.5 MB` of 5 kHz Parquet traces). Predictions were frozen at `2026-11-23 3:03:51 PM EST` and collection began `22 s` later at `3:04:13 PM EST`, though post-collection scoring required `--allow-code-change` on `analytics/data.py` (`torchvision:resnet18` variant-string normalization), rendering `lockbox.md` officially `EXPLORATORY`.

### 1.3 Headline Scientific Verdict
The repository's software engineering, provenance hashing, and causal windowing hygiene are unusually strong for a systems ML project—specifically, `analytics/` avoids naive row-random splitting in its primary tables and enforces sha256 freeze guards. **However, seven concrete statistical, physical, and instrumentation flaws severely distort both the published error metrics and the underlying ground-truth Watt/Joule measurements:**

1. **Mode A (`M1Ridge` / `AUTO`) Suffers Catastrophic Collinear Regularization Collapse Hidden by `MdAPE`**:
   - `ALL_FEATURES` (`19` features on `58` config rows) is **rank-deficient (`rank = 16 / 19`, condition number `3.68e16`)** with `rig_nov` 100% confounded with architecture family (`0/9` classifiers vs `9/9` `yolo*s/m` in `v2`).
   - Because inner LOMO tunes `alpha` by **pooled row-level Median APE** where **84.5% of rows (`49/58`) are YOLO** (`22/58` `yolo11n` alone), inner CV selects `alpha = 0.01` on classifier-holdout folds, causing `M1Ridge`/`AUTO` to **lose to the 3-feature `B3LogLinear` baseline on honest `S-LOMO` (`17.52%` vs `15.78%` MdAPE)** while hiding **`245.72%` MAPE**, **`4,396.56%` MaxAPE** (`effb0` predicted at `45x` true energy), **`R^2_raw = -0.899`**, and vacuous **5.3x-wide** conformal intervals (`[1,533, 8,128] mJ` on `yolo12n:640`).
   - Removing collinearity and decomposing CMOS physics (`P1_DecomposedCMOS_Ridge`) slashes honest `S-LOMO` MdAPE from **`17.52%` to `10.38%`** (`9.54%` with Huber `P2`), cuts `MAPE` from **`245.72%` to `13.61%`**, raises `S-LOMO` `mean_P_W` `R^2` from **`0.849` to `0.980`** (`MAE` `0.430 W` -> `0.167 W`), lowers overall Lockbox MdAPE from **`7.38%` to `5.38%`**, and cuts unseen-architecture (`yolo12n`) Lockbox MdAPE by **2.4x (`6.58%` -> `2.72%`)**.
2. **Deployment Calculator `1.05%` Lockbox Error Leaks Measured Test Duration**:
   - `lockbox_score.py` feeds the **post-hoc measured `serve_dur_ms`** of each lockbox cell into `deploy.predict()`. Replacing leaked ground-truth duration with honest a priori predicted duration (`N * (t_pred + gap)`) **triples `M1Ridge` median error from `1.05%` to `3.63%` MdAPE (`4.89%` MAPE)**.
3. **Mode L Whole-Cell `APE` Dilution, Bimodal `R^2` Illusion, and GBDT Extrapolation Ceiling**:
   - Scored `forward` inference accounts for a median of **only `25.8%`** (IQR `8.1%–42.7%`) of total cell Joules (`bench:*` microbenchmarks consume `61.2%` when present). Tuning and reporting Mode L on **whole-cell `APE`** cancels opposite-sign phase biases: on `lockbox_mbv3_224_t4_2400_r00`, `L3-trees` has **`8.45%` `forward` error** yet reports **`0.35%` whole-cell error** (`24x` smaller!).
   - Consecutive 100 ms windows have **lag-1 autocorrelation `r = 0.838`** (`0.844` within `forward`). Naive window-random CV gives **`0.158 W` forward MAE (`0.36%` MdAPE)** vs **`0.439 W` (`1.78%` MdAPE)** under honest YOLO LOMO and **`0.703 W` (`7.36%` MdAPE)** on non-YOLO lockbox cells (`resnet18`, `mbv3`). Across non-YOLO lockbox windows, all-phase `R^2` still looks like **`0.970`** due to `4.9 W` idle vs `9.9 W` active bimodality, while **`forward`-only `R^2` collapses to `0.386`**.
   - Replacing stale 1 Hz `sysfs` clocks and per-instruction `llc_mpki` with **10 Hz hardware-counter effective clock `f_eff = cycles_rate / (4e6 * busy)`**, **BCM2712 `V_core(f_eff)^2` dynamic power**, and **per-second DRAM miss rate (`llc_miss_rate_m`)** (`L_CMOS_Ridge`) cuts non-YOLO lockbox `forward` MdAPE from **`7.36%` (`L3-trees`) / `6.47%` (`L3-ridge`) to `5.31%`** (`resnet18` `forward` APE cut **2.2x** from `5.89%` to `2.67%`).
4. **Sync-Pulse Alignment Shears Mid-Run Timestamps Whenever USB Drops Samples (`57.2%` of Cells)**:
   - `239 / 418` cells (`57.2%`) experience Monsoon USB sample-loss slips (up to `157.7 ms` / `788` samples). `host/align.py` fits a single line `offset(t) = c0 + c1*(t - ref)` across endpoint pulse clusters (`sync:start` and `sync:end`), converting discrete mid-run sample drops into **fake linear clock drift up to `633.2 ppm`** (`53` cells `> 50 ppm` vs `0 / 179` zero-slip cells `> 50 ppm`, true crystal drift median `1.13 ppm`) while **every single sheared cell still passes `max_residual_ms < 1.0 ms` (`max = 0.563 ms`)** because two tight endpoint clusters always fit a straight line!
5. **`5.5%–5.9%` (`0.50–0.58 W`) of Reported "Pi Power" Is Cable/Connector `I^2 R` Heating**:
   - Comparing Monsoon output voltage (`5.25 V` in `v1`, `5.40 V` in `v2`/`lockbox`) against the Pi 5 PMIC input rail (`rail_EXT5V_V` in `context.csv`) via both direct `(V_Monsoon - V_EXT5V)/I` and ADC-offset-free differential `dV/dI` proves a **`0.162 Ohm` (`v1`) to `0.179 Ohm` (`v2`)** round-trip harness resistance. At `1.83 A` active load, **`0.581 W` (`5.88%` of reported board power)** is dissipated in the supply wire/connectors before reaching the Pi, and `183 / 418` cells (`43.8%`) sagged below `4.90 V` at the Pi input (minimum `4.706 V`).
6. **Ignored Silicon `VDD_CORE_V` Knee Solves `eda.md`'s "Under-Identified CMOS Exponent"**:
   - `analytics/` never reads `rail_VDD_CORE_V` from `context.csv`, leading `eda.md` §8 to obtain an unphysical `P_static = -8.5 W` when fitting `P_0 + k f^alpha`. Across `408` pinned cells, BCM2712 exhibits a sharp **DVFS core-voltage knee**: **`0.7229 V` (`1500 MHz`) -> `0.8741 V` (`2000 MHz`, `+151 mV`) -> `0.9294 V` (`2400 MHz`, `+55 mV`)**, yielding exact `V_core^2 f` scaling ratios of **`1.000 : 1.950 : 2.645`** that match measured dynamic power (`2.01 W : 3.71 W : 4.94 W` = **`1.00 : 1.84 : 2.45`**).
7. **Unmasked `ctx_read` Perturbations, `:u` Counter Blindness, and `LayerProfiler` Observer Distortion**:
   - Despite `README.md`, `docs/dataset-v2.md`, and `powersampler.py` stating `ctx_read` rows should be dropped, **neither `verify_cell.py` (`phase_energy.csv` -> Mode A) nor `windows.py` (Mode L) drops `ctx_read`**, injecting **`+0.112 W` (`p90 +0.270 W`)** spikes during idle and **`-0.170 W`** OpenMP barrier-stall dips during 4-thread `forward`.
   - `perf_event_paranoid=2` forces `:u` user-only counting, making `ctr_context_switches == 0` in **`215,400 / 215,400` windows (`0 / 176` cells)** and blinding Mode L to kernel/page-cache/driver power.
   - `src/layers.py` registers `pre`+`post` Python hooks on *every* `nn.Module` (container + leaf, up to **`9,486` hook calls/s** on `mbv3` and **`6,718` calls/s** on `yolo12n`), hardcodes `params * 4` bytes even for FP16/INT8 in `LayerProfiler.aggregated()`, and `perf stat -I 100` adds **`+0.9%` to `+1.5%` frame latency** on matched `yolo11n` runs.

---

## 2. Audit of the Power Prediction Models

### 2.1 Inventory of Modeling Generations in the Repository
1. **Legacy Per-Operator Log-Scaling (`C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi\modeling\train_scaling.py`, `evaluate.py`)**:
   - Fits `log(E_ABAB) = a + b_T log(T_eff) + c_B log(bytes_moved)` per `nn.Module` class on isolated `microbench.py` (`perop_power.csv`) replays.
   - **Flaws**: `train_scaling.py` fits on 100% of available `v3_*` files and reports **in-sample OLS RMSE** (`pred = X @ coef`); more fundamentally, isolated cache-hot operator replay (`ABAB`) violates cross-layer L2/L3 cache eviction, OpenMP thread-wakeup, and Python dispatch overhead—summed leaf microbenchmarks mispredict real end-to-end YOLO forward energy by `20%–60%`.
2. **Legacy Phase-Level Trace Forecast (`C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi\trace_forecast\models.py`, `evaluate.py`)**:
   - Ridge (`lambda = 1.0`) over `[1, log(T_eff), log(bytes)]` plus sequential hill-climbing (`+AI`, `+freq*temp`, `+attn_flops`, `+family_intercept`).
   - **Flaws**: `columns()` encodes categorical `fam_idx` as a **scalar integer `0, 1, 2, ...` (`float(feat.get("fam_idx", 0))`)** in a linear dot product (imposing an arbitrary linear ordering over model families!) and reads `temp0` from `context.csv` (a live sensor unavailable a priori).
3. **Active Production Analytics Suite (`C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi\analytics\`)**:
   - **Mode A (`mode_a.py`, `mode_a_models.py`, `tables.py`, `static.py`)**: A priori configuration-level prediction of `energy_mJ` per image, `latency_ms` per image, and `mean_P_W` from static graph features + runtime knobs (`n = 58` configs collapsed from `226` QC-primary vision cells out of `403` `v1+v2` training cells).
   - **Deployment Calculator (`deploy.py`, `lockbox_score.py`)**: Combines Mode A `energy_mJ` and `latency_ms` with an empirical `idle_table` (`p_idle * (duration_s - N * t_pred)`).
   - **Mode L (`windows.py`, `mode_l.py`, `mode_l_models.py`)**: Causal 100 ms live telemetry power prediction (`L0` lookup through `L4-trees` `HistGradientBoostingRegressor`) over `161` QC-primary `v2` YOLO counter cells (`198,619` windows).

---

### 2.2 Mode A Scientific Audit: Design-Matrix Rank Deficiency, Metric Blindness, and Split Leakage

#### Flaw A1: `43.9%` of Collected Training Cells Are Discarded Before Fitting
Out of `403` `v1+v2` training cells, `data.py` (`mode_a_eligible`) discards **`177` cells (`43.9%`)**: all `96` training cells (`train_*`), all `48` LLM cells (`lm16/64/128`, `smollm2`, `qwen25`), and `33` image cells (`ondemand` DVFS, `fan_mode != full`, `serve_fps`, `soak`, `concurrent`, and models outside the 9 hardcoded `VISION_MODELS`). The remaining `226` cells collapse by median into just **`58` configuration rows**—of which **`49` (`84.5%`) are YOLO** (`22` of them `yolo11n` alone = **`37.9%` of all rows**!) and only **`9` (`15.5%`) are non-YOLO classifiers** (`effb0`, `mbv3`, `resnet18` at `1500/2000/2400 MHz` in `v1` only).

#### Flaw A2: Rank-Deficient Design Matrix (`rank = 16 / 19`, Condition Number `3.68e16`) & Total Rig Confounding
SVD of the standardized `58 x 19` `ALL_FEATURES` matrix (`audit_mode_a_experiments.py`) reveals **three exact/near-exact linear dependencies** (singular values `1.42e-13`, `7.61e-16`, `5.18e-16`):
1. **Exact compositional sum**: $\sum_{g \in \{\text{conv, linear, norm, act, pool, other}\}} \text{frac\_macs}_g = 1.0$ to machine precision (`1.11e-16`).
2. **Exact FP32 parameter/weight-byte redundancy**: On `53 / 58` rows (`fp32`), `log_weight_bytes - log_params == ln(4)` (`1.55e-15`), while `is_fp16` is `1` on **only `1` row (`yolo11n_fp16`)** and `is_int8` is `1` on **only `4` rows**.
3. **100% Rig-vs-Architecture Confounding**: `rig_nov` (`nov2026` vs `oct2026`) is **`0` on `9 / 9` classifier rows** and **`1` on `9 / 9` `yolo11s/m` + `yolov8s/m` rows**. Any linear weight on `rig_nov` directly aliases classifier-vs-detector architecture shifts with the `5.25 V -> 5.40 V` Monsoon voltage change.

#### Flaw A3: Inner-CV `MdAPE` Majority-Class Tyranny Causes `4,396%` Out-of-Sample Explosion
In `mode_a.py` (`_Search.select`), hyperparameter `alpha` in `logspace(-2, 4, 7)` is selected by minimizing **pooled row-level `MdAPE`** across inner LOMO predictions. When a classifier (`effb0` or `resnet18`, `3` rows) is held out in outer `S-LOMO`, **49 of the 55 inner validation rows (`89.1%`) are YOLO**. Setting `alpha = 0.01` overfits collinear `frac_macs_*` and `rig_nov` columns to shave `<0.3%` off median YOLO error while blowing up the remaining `6` classifier rows—**which cannot move the 50th percentile because `6 / 55 < 50%`!**
- Consequently, on honest `S-LOMO`, `M1Ridge` and `AUTO` select `alpha = 0.01` when `effb0`, `resnet18`, or `yolo11n` is held out, predicting `effb0` at **45x its true energy (`4,340.8%` MdAPE, `4,396.6%` MaxAPE)** and `resnet18` at **`94.9%` MdAPE**.
- Reporting only `MdAPE` (`17.52%` for `AUTO`, `18.10%` for `M1Ridge`) masked **`MAPE = 245.72%`**, **`R^2_raw = -0.899`**, **`R^2_log = -0.089`**, and the fact that **shipped `AUTO` is worse than the 3-feature `B3LogLinear` baseline (`15.78%` MdAPE)**!
- Furthermore, those extreme `S-LOMO` calibration residuals inflated the 90% conformal multiplier in `lockbox.md` to a **5.3x multiplicative span** (e.g., `yolo12n:640` truth `3,700 mJ`, `90% PI = [1,533, 8,128] mJ`), making `"100% inside_90"` uninformative.

#### Flaw A4: Quantified Split-Leakage Ladder on Mode A
Evaluating the exact same models across splitting granularities proves why random cell or config splits are scientifically invalid:

| Evaluation Split (`energy_mJ`) | `B3LogLinear` MdAPE | `M1Ridge` MdAPE | `AUTO` MdAPE | `AUTO` MAPE | `AUTO` MaxAPE | `AUTO` `R^2_raw` | `P0_CleanRidge9` (Ours) | `P1_DecomposedCMOS` (Ours) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cell-Random 5-Fold** (`n=226`, leaks sibling `r00..r09` repeats) | — | **1.78%** | — | 2.65% (`M1`) | 13.29% | **0.992** | — | — |
| **Config-Random 5-Fold (`RANDOM`)** (`n=58`, leaks seen architectures) | 17.90% | 5.29% | **5.29%** | 7.03% | 31.84% | **0.986** | **5.00%** | 6.31% |
| **Unseen Variant (`S-LVO`)** (INT8 / FP16 / pruned) | 8.61% | 10.61% | **9.91%** | 10.45% | 22.95% | 0.941 | 6.40% | **2.21%** |
| **Unseen Clock (`S-LCO`)** (`1500` / `2000` / `2400 MHz`) | 15.04% | 16.09% | **16.09%** | 15.86% | 35.15% | 0.920 | **6.57%** | 7.64% |
| **Unseen Resolution (`S-LSO`)** (`320` / `480` / `640 px`) | 83.24% | 19.22% | **19.22%** | 19.12% | 40.23% | 0.903 | **6.02%** | 11.17% |
| **Honest Unseen Model (`S-LOMO`)** (`n=58`, 9 base models) | 15.78% | 18.10% | **17.52%** | **245.72%** | **4,396.56%** | **-0.899** | 10.78% | **10.38%** (`P2`: **9.54%**) |
| **`S-LOMO` Classifiers Only** (`effb0`, `mbv3`, `resnet18`, `n=9`) | 58.03% | 94.92% | **94.92%** | **1,487.82%** | **4,396.56%** | -128.4 | 40.97% | **35.48%** |
| **Unseen Family (`LOFO`)** (`yolov8` vs `yolo11` vs `classifiers`) | 23.31% | 22.89% | **22.89%** | 41.25% | 217.50% | 0.684 | **8.84%** | 15.48% (`P2`: **10.61%**) |
| **Lockbox (`n=5` configs, all)** | 31.46% | 7.38% | **7.38%** | 8.04% | 10.91% | 0.985 | 6.00% | **5.38%** |
| **Lockbox Truly Unseen Model (`yolo12n:320, 640`)** | 11.82% | 6.58% | **6.58%** | 6.58% | 7.38% | — | 4.83% | **2.72%** |

#### Why `P1_DecomposedCMOS` Wins Physically and Statistically
Instead of fitting an unregularized black-box log-linear sum over 19 collinear features, `P1_DecomposedCMOS` (`audit_mode_a_experiments.py`) respects silicon physics:
\[
\hat{E}_{\text{frame}} = \hat{t}_{\text{roofline}}(\text{MACs}, \text{Bytes}, f, \text{threads}, \text{prec}) \times \Big( P_{\text{idle}}(f) + \Delta\hat{P}_{\text{dyn}}\big(V_{\text{core}}(f)^2 f,\ \text{AI},\ \text{threads},\ \text{conv\_share}\big) \Big)
\]
with inner hyperparameter selection minimizing **model-balanced mean log-MAE** (`alpha >= 1.0`) rather than YOLO-dominated row `MdAPE`. On `S-LOMO`, `P1_DecomposedCMOS` cuts **latency MdAPE in half (`18.32%` -> `9.10%`, `R^2_log`: `0.462` -> `0.963`)** and cuts **active power `mean_P_W` MdAPE by 2.4x (`3.58%` -> `1.51%`, `MAE`: `0.430 W` -> `0.167 W`, `R^2_raw`: `0.849` -> `0.980`)**. Note also why `lockbox` (`7.38%` `M1Ridge`) looked much easier than `S-LOMO` (`17.52%`): **3 of the 5 lockbox configs (`resnet18:224`, `mbv3:224`, `yolov8n:480`) are base models ALREADY IN THE TRAINING SET** (`novelty = 0`), and when refit on all 58 rows `M1Ridge` happened to jump to `alpha = 10.0`. On the **only genuinely unseen architecture in the lockbox (`yolo12n`)**, `P1_DecomposedCMOS` cuts error from **`6.58%` to `2.72%`** (`0.07%` APE on `yolo12n:320`, `5.38%` on `yolo12n:640`).

---

### 2.3 Deployment Calculator Audit: Post-Hoc Duration Leakage (`1.05%` -> `3.63%` MdAPE)
In `C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi\analytics\lockbox_score.py`, `calculator()` evaluates `deploy.predict()` on the 15 lockbox `serve` phases (`10` requests separated by `200 ms` sleeps) by passing **`duration_s = r["serve_dur_ms"] / 1000.0`**—the **measured wall-clock duration from the test trace**. Because total `serve` energy is $N \cdot E_{\text{img}} + P_{\text{idle}} \cdot (T_{\text{serve}} - N \cdot t_{\text{img}})$, plugging in measured $T_{\text{serve}} \approx N t_{\text{true}} + (N-1)\Delta t_{\text{gap}}$ leaks $N t_{\text{true}}$ and cancels `M1Ridge`'s `+11.8%` latency overestimate (`-P_idle * N * (t_pred - t_true)` partially cancels `+N * (E_pred - E_true)`!).
- **Leaked measured duration (`lockbox.md` headline)**: `MdAPE = 1.05%`, `MAPE = 1.91%`.
- **Honest a priori duration (`N * (t_pred + 200 ms)`)**: `MdAPE = 3.63%` (**`3.45x` higher**), `MAPE = 4.89%`.

---

### 2.4 Mode L (100 ms Live Telemetry) Audit: Temporal Leakage, Bimodal `R^2`, Whole-Cell Dilution, and Non-YOLO Extrapolation

#### Flaw L1: Severe Temporal Autocorrelation & The 5-Level Splitting-Leakage Ladder
Across all `198,619` 100 ms windows (`161` `v2` YOLO cells), board power `P_W` has **median lag-1 (100 ms) autocorrelation `r = 0.838`** (`r = 0.844` strictly inside `forward` windows!), lag-5 (500 ms) `r = 0.539`, and lag-10 (1.0 s) `r = 0.506`. Evaluating `L3-trees` across five progressively stricter splits (`audit_mode_l_experiments.py`) quantifies both **temporal/replicate leakage** and the **bimodal `R^2` illusion**:

| Split Level (`L3-trees`, 100 ms windows) | `win_MAE` All (`W`) | `win_MAE` Forward (`W`) | `win_R^2` All Phases | `win_R^2` Forward Only | Whole-Cell `MdAPE` | **`forward`-Only `MdAPE`** |
|---|---:|---:|---:|---:|---:|---:|
| **1. Window-Random 5-Fold** *(leaks `t-1, t+1` neighbors)* | 0.142 W | **0.158 W** | 0.9919 | 0.9823 | 0.28% | **0.36%** |
| **2. Cell-Random 5-Fold** *(holds out runs, leaks `r00/r01/r02`)* | 0.182 W | **0.254 W** | 0.9867 | 0.9548 | 0.42% | **0.54%** |
| **3. Config-Group 5-Fold** *(holds out configs, leaks architecture)* | 0.197 W | **0.283 W** | 0.9849 | 0.9475 | 0.70% | **1.14%** |
| **4. Honest YOLO LOMO** *(5 unseen YOLO architectures)* | 0.262 W | **0.439 W** | 0.9728 | **0.8789** | 1.16% | **1.78%** |
| **5a. Lockbox YOLO** (`yolo12n` + `yolov8n:480`, `9` cells) | 0.131 W | **0.188 W** | 0.9919 | 0.9728 | 0.42% | **1.63%** |
| **5b. Lockbox Non-YOLO** (`resnet18` + `mbv3`, `6` cells) | 0.265 W | **0.703 W** | **0.9704** *(illusion!)* | **0.3860** *(reality!)* | 4.86% | **7.36%** (`Max 8.82%`) |

Look at row `5b`: **all-phase `R^2` is `0.9704` even when `forward`-only `R^2` is `0.3860`**, because any step function separating `4.88 W` idle windows from `9.8 W` active windows captures >95% of total bimodal variance while missing the inter-model active power spread (`9.06 W` `mbv3` vs `10.85 W` `resnet18`)!

#### Flaw L2: Whole-Cell `APE` Hyperparameter Selection & `bench:*` Phase Cancellation
Across all `418` cells, scored `forward` inference accounts for a **median of only `25.8%`** (IQR `8.1%–42.7%`) of whole-cell energy; when single-layer `bench:*` microbenchmarks run (`r00` cells), `bench:*` consumes a **median of `61.2%`** of the cell's Joules (`r00` cells are `7x–10x` longer than `r01/r02`). Yet `mode_l.py` (`_select`, line 102) tunes hyperparameters on **whole-cell `APE`** and headlines whole-cell `MdAPE` (`1.2%`). On `lockbox_mbv3_224_t4_2400_r00`, `L3-trees` overpredicts `forward` power by **`+8.45%` (`9.86 W` vs `9.06 W`)**, yet whole-cell `APE` is **`0.35%`** because idle/warmup/load phases dominate the duration and cancel the signed error (`r01`, without `bench:*` / long setup dilution, jumps to **`6.83%`** whole-cell APE!).

#### Flaw L3: Missing Rate-Scaled DRAM Features, Stale 1 Hz Clocks, & Tree Extrapolation Ceiling
1. **Ratio vs Rate Physics**: `windows.py` computes `llc_mpki = LLC_load_misses / instructions * 1000` (misses *per instruction*), omitting **`llc_miss_rate = LLC_load_misses / dur_s`** (misses *per second*). LPDDR4X DRAM power (Watts = Joules/second) scales with memory transactions **per second**, not per instruction!
2. **Stale 1 Hz `sysfs` Clock in `ondemand` DVFS**: `sysfs.csv` polls `scaling_cur_freq` every `1.0 s`, whereas Linux `ondemand` switches clocks in `10–20 ms`. Across `ondemand` cells, **`66.9%` of `idle_between_requests` samples** (`4.20 W`, truly at `1500 MHz`) still report stale `2400 MHz`, and **`20.3%` of active 100 ms windows** (`busy >= 0.25`) have `sysfs` `freq_mhz` stale by **`> 200 MHz`**. Yet `windows.py` already has 10 Hz `cycles_rate` and `busy`, from which the **exact 10 Hz hardware-integrated clock** $f_{\text{eff}} = \frac{\text{cycles\_rate}}{4 \times 10^6 \cdot \text{busy}}$ can be computed causally with zero `sysfs` lag!
3. **Zero Non-YOLO Counter Cells in Training + GBDT Leaf Ceiling**: All `161` `v2` counter cells are YOLO (`0` CNN classifiers, `0` LLMs, `0` training runs). On `resnet18` (`10.85 W` active power due to high-IPC dense `3x3` NEON convolutions), `L3-trees` hits a **tree leaf ceiling at `10.19 W` (`5.89%` `forward` MdAPE)** because axis-aligned trees cannot extrapolate beyond training leaf means. Our rate-scaled CMOS model **`L_CMOS_Ridge`** ($V_{\text{core}}(f_{\text{eff}})^2 \cdot \text{cycles\_rate}$, $V_{\text{core}}(f_{\text{eff}})^2 \cdot \text{instr\_rate}$, `llc_miss_rate_m`, thermal leakage; tuned by inner window-MAE) extrapolates linearly to `10.52 W`, cutting **Lockbox Non-YOLO `forward` MdAPE from `7.36%` (`L3-trees`) to `5.31%`** (`resnet18`: **`5.89%` -> `2.67%`**) and beating `L3-trees` on Lockbox YOLO too (`1.55%` vs `1.63%`):

| Mode L Candidate | LOMO Whole `MdAPE` | LOMO `forward` `MdAPE` | LOMO `forward` `R^2` | Lockbox YOLO `forward` `MdAPE` | **Lockbox Non-YOLO `forward` `MdAPE`** | **Lockbox Non-YOLO `forward` `MaxAPE`** |
|---|---:|---:|---:|---:|---:|---:|
| `L0` (lookup) | 5.77% | 5.76% | 0.496 | 1.84% | 8.98% | 10.84% |
| `L1` (5 s PMIC OLS) | 6.69% | 13.58% | -0.627 | 18.40% | 16.03% | 28.91% |
| `L2` (`busy`, `cycles`) | 5.07% | 7.04% | 0.513 | 1.44% | 7.54% | 8.69% |
| `L3-ridge` (official) | 2.94% | 3.32% | 0.768 | 1.69% | 6.47% | 9.41% |
| `L3-trees` (official) | **1.16%** | 1.78% | **0.879** | 1.63% | 7.36% | 8.82% |
| `L4-trees` (+5 s PMIC) | 1.22% | 1.94% | 0.873 | 1.76% | 7.34% | 8.67% |
| **`L_CMOS_Ridge` (Ours)** | 2.10% | 2.00% | 0.856 | **1.55%** | **5.31%** | 8.44% |
| **`L_CMOS_Decomp_Ridge` (Ours)** | 2.36% | 2.91% | 0.818 | 2.02% | 5.65% | **7.52%** |
| **`L_CMOS_Trees` (Ours)** | 1.25% | **1.67%** | 0.837 | 1.73% | 7.93% | 12.08% |

*(Note on `L1` PMIC failure: `L1` fails catastrophically on 100 ms / 1 s windows (`13.58%` LOMO forward MdAPE, `56.3%` idle APE) NOT because the Pi 5 PMIC ADC is inaccurate—on per-cell means, `pmic_W` tracks Monsoon `P_W` at `r > 0.98`—but because `context.csv` only polls every `5.0 s` and `windows.py` holds the reading until the **next** read (`t_{i+1} - 5 s`), making `pmic_W` `2.5–10 s` stale across phase transitions and attenuating `L1`'s OLS slope via errors-in-variables dilution to `b = 0.47` instead of `~1.18`!)*

---

## 3. Audit of the Hardware Data Collection Pipeline (`power-measurement-pi`)

### 3.1 Bug C1 (Critical): Sync-Pulse Linear Clock Fit Converts Mid-Run USB Sample Drops into Timestamp Shear (`host/align.py`)
- **Mechanism** (`C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi\host\align.py`, lines `291–349`):
  Each cell brackets the run with two 3-pulse CPU load trains: `sync:start` (`6` edges inside `0.6 s` at $t \approx 0$) and `sync:end` (`6` edges inside `0.6 s` at $t \approx T$). `align.py` fits a single global line `offset(t) = c0 + c1 * (t - ref)` across those 12 endpoint edges and maps every Monsoon sample via `t_pi = (t_m - c0 + c1 * ref) / (1 + c1)`.
- **Why It Fails Silently**:
  When the Monsoon HVPM drops `K` USB samples mid-run at $t_{\text{slip}} \in (0, T)$, sample indices after $t_{\text{slip}}$ jump ahead in real time by $\Delta t_{\text{slip}} = K / 5000\text{ s}$. Because `sync:start` has offset $c_0$ and `sync:end` has offset $c_0 + \Delta t_{\text{slip}}$, `np.polyfit` draws a diagonal ramp of **fake clock drift** $c_1 \approx \Delta t_{\text{slip}} / T$ across the entire run! Worse, because all 6 start edges share $t \approx 0$ and all 6 end edges share $t \approx T$, **a straight line through two tight endpoint clusters ALWAYS has near-zero residuals at the endpoints**, so `max_residual_ms < 1.0 ms` **never trips** while mid-run timestamps right before/after $t_{\text{slip}}$ are sheared by up to $\pm \Delta t_{\text{slip}} / 2$ (up to **`78.9 ms`** in the 30-min soak, and **`2–16 ms`** in 20–50 s cells—enough to misattribute whole layers and shift 100 ms `perf` window boundaries)!
- **Empirical Proof Across All 418 Cells (`collection_physics_summary.json`)**:
  - **Zero-slip cells (`179 / 418`, `42.8%`)**: True crystal drift `|drift_ppm|` has median **`1.13 ppm`**, p95 **`6.33 ppm`**, max **`20.83 ppm`**; **`0 / 179` cells exceed `50 ppm`**.
  - **Slip cells (`239 / 418`, `57.2%`)**: Apparent `|drift_ppm|` jumps **13.6x** to median **`15.43 ppm`**, p95 **`340.42 ppm`**, max **`633.19 ppm`**, with **`53` cells exceeding `50 ppm`** (including `3 / 15` lockbox cells up to `+183.1` / `-170.1 ppm`). Every single one of those `53` cells passed `max_residual_ms < 1.0 ms` (**worst residual only `0.563 ms`**, e.g. `train_lm64_sgd_1500_r01`: `36` lost samples = `7.13 ms` step -> `224.8 ppm` fake drift, `max_residual_ms = 0.176 ms`!).
  - Ironically, `detect_slips()` on line `328` of `align.py` **already detects the exact `t_ns` and `step_ms` of each slip**, but runs *after* `c0, c1` is fitted and never inserts the step discontinuities into `t_m` before fitting `c0, c1`!

### 3.2 Flaw C2 (Critical): Two-Wire Monsoon Sensing Measures `0.50–0.58 W` (`5.5%–5.9%`) of Cable `I^2 R` Loss & Sags Below `4.90 V`
- **Mechanism & Empirical Proof (`pmic_and_cable_physics.json`)**:
  The Monsoon HVPM regulates and multiplies `V_V * I_A` at its front-panel terminals without remote Kelvin sense leads at the Pi 5 header, while the Pi 5 PMIC (`context.csv` `rail_EXT5V_V`) independently measures voltage at the board input. Computing harness resistance both directly (`(V_Monsoon - V_EXT5V) / I`) and differentially (`(Delta V_Monsoon - Delta V_EXT5V) / Delta I` between idle and active load, canceling any fixed ADC offset):
  - **`v1` (`Vout = 5.25 V`, `224` cells)**: `R_direct = 0.159 Ohm`, `R_diff = 0.162 Ohm`; active `EXT5V = 4.961 V` (median min `4.882 V`); cable Joule heating = **`0.501–0.527 W` (`5.5%` of reported power)**.
  - **`v2` (`Vout = 5.40 V`, `179` cells)**: `R_direct = 0.179 Ohm`, `R_diff = 0.179 Ohm` (harness resistance *rose* by `+17 mOhm` between October and November!); active `EXT5V = 5.080 V`; cable Joule heating = **`0.568–0.581 W` (`5.88%` of reported power)**.
  - **`lockbox` (`Vout = 5.40 V`, `15` cells)**: `R_direct = 0.174 Ohm`, `R_diff = 0.163 Ohm`; cable Joule heating = **`0.541–0.583 W` (`5.7%–5.9%`)**.
  - Across the dataset, **`183 / 418` cells (`43.8%`)** drooped below `4.90 V` on `EXT5V_V` (global minimum **`4.706 V`**), and **all `15 / 15` lockbox cells** started with `throttled_start = 0x50000` (sticky boot under-voltage/throttling history bits). Because DC-DC buck converters on the Pi 5 draw *less* input current at `5.40 V` (`0.910 A` idle in `v2`) than at `5.25 V` (`0.952 A` idle in `v1`) for constant rail load, changing `Vout` + `R_harness` alters both converter efficiency and $I^2 R_{\text{harness}}$ nonlinearly with load!

### 3.3 Flaw C3: Quantified Observer Effects (`LayerProfiler`, `vcgencmd`, `perf`, Frame-0 Profiler)
*(Note on `pstack` / `ptrace`: No `pstack` or `ptrace` stop-the-world stack sampler is used in `power-measurement-pi`—and rightly so, as `ptrace(PTRACE_ATTACH)` halts target threads for millisecond-scale DWARF unwinding, collapsing CPU utilization and injecting artificial idle valleys. However, four real software observer effects ARE present in `src/` and quantified below:)*

1. **In-Process `LayerProfiler` Python Forward Hooks (`src/layers.py`, lines `272–332`)**:
   - Registers both a `forward_pre_hook` and `forward_hook` on **every** `nn.Module` (containers *and* leaves!). Each invocation walks nested Python args/tensors (`_input_record`), hashes JSON signatures, calls `time.perf_counter()`, and emits `EventLog` `begin`/`end` timestamps.
   - Empirical hook rate during `forward`: **`9,486` hook calls/s** on `mbv3` (`146` leaf calls in `30.8 ms`), **`6,718` calls/s** on `yolo12n` (`619` leaf calls in `244.2 ms`), and **`2,905` calls/s** on `yolo11n`. Because container + leaf hooks nest, summed `layer_table.csv` leaf durations + hook overhead exceed or miss wall-clock `forward` duration (`28.9%` unaccounted Python/dispatch time on `mbv3`; `30.7%` unaccounted on `yolo11n_fp16`). Worse, `src/layers.py` line `370` hardcodes `+ self.params(path) * 4` (FP32 bytes) even for FP16/INT8 quantized modules!
2. **Unmasked `vcgencmd` `ctx_read` Subprocess Bursts (`src/powersampler.py`, lines `181–196, 267–290`)**:
   - Every `5.0 s`, `_ctx_loop` spawns **four sequential `/usr/bin/vcgencmd` subprocesses** (`measure_temp`, `measure_clock arm`, `get_throttled`, `pmic_read_adc`), taking `~50 ms` (`~1.04%` of scored time, up to `2.13%`).
   - Direct trace comparison (`ctx_read == True` vs adjacent `ctx_read == False` samples in the same phase) proves **opposite-sign observer distortion**:
     - During `idle` / `baseline`: **`+0.112 W` median spike (`+0.270 W` p90)** from spawning 4 Linux processes + VideoCore mailbox IPC.
     - During 4-thread `forward`: **`-0.170 W` median power dip** because OS scheduling of `vcgencmd` preempts one of the 4 OpenMP NEON worker threads, stalling the other 3 threads at the next barrier and lowering instantaneous core power while stretching frame latency!
   - Even though `align.py` flags `ctx_read` in `power_trace.parquet` specifically so downstream scripts drop it, **neither `host/verify_cell.py` (`phase_energy.csv`) nor `analytics/windows.py` drops `ctx_read`**.
3. **`perf stat -I 100` Counter Collection Overhead (`src/perfcounters.py`)**:
   - Comparing matched `yolo11n` `640 px` `2400 MHz` `v2 ref` (`perf` OFF: `290.99 ms/frame`, `9.941 W` `forward` step) against `v2` `perf` ON (`A-input` / `D`: `293.48–295.47 ms/frame`, `9.908–9.914 W`), `perf` ON adds **`+0.9%` to `+1.5%` (`+2.5` to `+4.5 ms`) frame latency** (and across `eda.md`'s whole-frame comparison, `-0.320 W` / `-3.34%` power).
4. **Dead `:u` `context-switches` Counter & Kernel Blindness (`src/perfcounters.py`, lines `12, 43–45`)**:
   - Under `perf_event_paranoid = 2`, `perf` appends `:u` (user-space only) to all events. Because Linux context switches occur in kernel mode (`:k`), **`ctr_context_switches` is identically `0` across all `215,400` counter intervals in all `176` counter cells (`v2` + `lockbox`)**! Kernel page faults, VFS/page-cache I/O (`A-input` image streaming), and child processes are likewise invisible.
5. **Frame-0 `torch.profiler` Cold Spike & Within-Run Thermal Drift**:
   - Frame `0` runs `torch.profiler` (`1,083.0 ms` median = **`3.56x`** normal frame duration); while correctly excluded from Mode A frame means, its heat pulse (`+14.1 mW/degC` idle leakage slope at `2400 MHz`) contributes to `baseline_B - baseline_A` thermal baseline drift (median `+34` to `+40 mW` with fan `full`, **`+56` to `+186 mW` with fan `auto`/`off`**, max **`+416 mW`**).

---

## 4. Prioritized High-Leverage Recommendations (`build-the-lever`)

### Part A: Highest-Leverage Improvements for Scientific Power Prediction (Immediate, Zero New Hardware Needed)

| Priority | Lever | Concrete Implementation (`analytics/`) | Verified Empirical Payoff |
|---|---|---|---|
| **A-1** | **Replace `M1Ridge`/`AUTO` with Decomposed CMOS Roofline (`P1_DecomposedCMOS`) + Clean Full-Rank Features** | Drop redundant `log_weight_bytes` (use `log_params` + effective bits/weight), drop one `frac_macs_*` simplex column, remove confounded `rig_nov` (or normalize board voltage first), and predict $\hat{E} = \hat{t}_{\text{roofline}} \cdot (P_{\text{idle}}(f) + \Delta\hat{P}_{\text{dyn}}(V_{\text{core}}(f)^2 f, \text{AI}, \text{threads}))$. Tune inner CV on **model-balanced log-MAE** (`alpha >= 1.0`), never row-pooled `MdAPE`. | Cuts `S-LOMO` MdAPE **`17.52% -> 10.38%`** (`P2`: **`9.54%`**), slashes `MAPE` **`245.7% -> 13.6%`**, raises `mean_P_W` `S-LOMO` `R^2` **`0.849 -> 0.980`**, cuts Lockbox MdAPE **`7.38% -> 5.38%`**, and cuts unseen `yolo12n` error **`6.58% -> 2.72%`**. |
| **A-2** | **Subtract `I^2 R_harness` (`P_board = V_EXT5V * I_Monsoon`) Before Modeling** | Build a 30-line `derive_true_board_power.py` lever that interpolates `rail_EXT5V_V` (or uses `V_Monsoon - I_Monsoon * R_harness`, with `R_v1 = 0.162 Ohm`, `R_v2 = 0.179 Ohm`) so targets represent **true Pi 5 board power** (`P_Pi = P_Monsoon - I^2 R_harness`) and eliminate the `5.25 V` vs `5.40 V` rig confound. | Removes **`0.50–0.58 W` (`5.5%–5.9%`)** systematic cable heating bias and unconfounds `oct2026` classifiers from `nov2026` detectors without needing `rig_nov`. |
| **A-3** | **Upgrade Mode L Features to 10 Hz Hardware `f_eff`, `V_core(f_eff)^2`, and Per-Second DRAM Miss Rate (`L_CMOS_Ridge` / Stacked Hybrid)** | In `windows.py`, add causal 10 Hz `f_eff_mhz = cycles_rate / (4e6 * busy)` (fallback `freq_mhz` when `busy < 0.03`), `vcore_eff = interp(f_eff, [1500,2000,2400], [0.723,0.874,0.929])`, `cmos_cyc_dyn = vcore^2 * cycles_rate`, `cmos_ins_dyn = vcore^2 * instr_rate`, and `llc_miss_rate = llc_mpki * instr_rate / 1000`. Select hyperparameters by **window MAE / active-phase APE**, not whole-cell `APE`. | Eliminates **1 Hz `sysfs` DVFS staleness** (`66.9%` idle-gap, `20.3%` active); cuts Non-YOLO lockbox `forward` MdAPE **`7.36% -> 5.31%`** (`resnet18`: **`5.89% -> 2.67%`**) and beats `L3-trees` on Lockbox YOLO (`1.55%` vs `1.63%`). |
| **A-4** | **Fix Deployment Calculator Lookahead & Report Honest Non-Diluted Metrics** | In `lockbox_score.py` and `deploy.py`, compute `duration_s = n_requests * (t_pred_s + gap_s)` when throughput is open-loop; report `MAPE`, `P90 APE`, `MaxAPE`, `forward`-only `R^2` (never bimodal all-phase `R^2` alone), and `ape_meas` (`forward`-only APE, never whole-cell `ape` diluted by `bench:*`/`idle`). | Restores scientific honesty (`3.63%` true a priori `serve` MdAPE vs leaked `1.05%`; exposes `0.386` vs `0.970` `R^2` illusion). |
| **A-5** | **Mask `ctx_read` Rows in `verify_cell.py` & `windows.py` and Unlock `lm` / `train` Modeling** | Filter `~df["ctx_read"]` when computing `phase_energy.csv` and `windows.py` `_label()`, and fit task-parameterized roofline heads on the already-collected `48 lm` (`prefill` + `decode` per-token) and `96 train` (`forward + backward + optimizer` per-step) cells instead of discarding `43.9%` of the dataset. | Eliminates `+0.112 W` idle spikes and `-0.170 W` active preemption dips across all existing traces; unlocks `144` already-paid-for LM & training runs. |

---

### Part B: Highest-Leverage Fixes for Next-Round Hardware Data Collection (`power-measurement-pi`)

| Priority | Lever | Concrete Implementation (`src/`, `host/`) | Why It Matters |
|---|---|---|---|
| **B-1** | **Piecewise-Step Slip Correction in `host/align.py` (`< 25` lines of Python)** | Before fitting `c1, c0 = np.polyfit(pi_t - ref, offs, 1)`, run `slips = detect_slips(...)` and add cumulative `step_ns` at each `slip["t_ns"]` to `t_m`! Fit `c1` with a physical crystal bound (`|drift_ppm| <= 30 ppm`), and add periodic single-core 20 ms sync ticks every `60 s` during long `soak` runs. | Completely eliminates fake `50–633 ppm` linear drift shear in the **`57.2%` of cells** with Monsoon USB sample drops—**can even re-align raw recordings retroactively!** |
| **B-2** | **4-Wire Kelvin Sense / Low-`R` Short Harness + Board-Side `EXT5V` Calibration** | Wire short (`< 20 cm`) `14–16 AWG` soldered/paralleled GPIO `5V`+`GND` leads (targeting `R_harness < 0.025 Ohm` vs current `0.179 Ohm`), clear sticky throttle bits before batch start (or gate strictly on `sysfs` `in0_lcrit_alarm` + `EXT5V_V >= 4.95 V`), and hold Monsoon `Vout` fixed across all campaigns. | Cuts parasitic cable heating by **`>7x`** (`0.58 W -> <0.08 W`), eliminates the `4.71 V` brownout sag (`183` cells `< 4.90 V`), and eliminates cross-campaign voltage confounding. |
| **B-3** | **Decouple Structural Profiling Runs from Clean Power Runs (`--hooks off` by default)** | Run `tools/static_features.py` (or a single un-metered dry pass) once per `(model, imgsz, precision)` to emit `layer_table.csv` (fixing `src/layers.py` line `370` to use actual parameter `element_size()` instead of `* 4`!), and run metered power cells with **zero `nn.Module` hooks**, **no Frame-0 `torch.profiler`**, and **`ctx_loop` paused during `forward`/`serve`** (taking PMIC/temp snapshots only in `baseline_A`, inter-phase gaps, and `baseline_B`). | Eliminates **`9,486` Python hook calls/s**, eliminates `vcgencmd` preemption stalls (`-0.170 W`), eliminates Frame-0 `3.56x` thermal pre-heating, and shrinks cell runtime/storage by **`~3x`** (skipping redundant `bench:*` on `r00`). |
| **B-4** | **Enable Kernel + System-Wide PMU Counters (`perf_event_paranoid = -1`, `-a`) + Fast `sysfs` Polling (`20 Hz`)** | Set `sysctl kernel.perf_event_paranoid=-1` (or `CAP_PERFMON`) on the Pi 5 lab image so `perf stat -I 100 -a` captures both `:u` and `:k` across all 4 Cortex-A76 cores (reviving dead `context-switches`, kernel page faults, and DMA/scheduler overhead), and bump `SysfsSampler` (pure in-process file reads, `< 0.05 ms`) from `1 Hz` to **`20 Hz` (`50 ms`)** while also reading `VDD_CORE_V` via `/sys/class/hwmon` if exposed or bracketing DVFS OPP residencies (`cpufreq/stats/time_in_state`). | Fixes **`100%` dead `context-switches`**, captures kernel/I/O power for multi-process & streaming deployments, and tracks `ondemand` DVFS transitions cleanly even at `busy == 0` idle! |
| **B-5** | **Expand Mode A & Mode L Architectural Diversity (`>= 20` Base Models with Counters ON)** | Collect `3`-repeat clean cells (`perf` ON + `1` `perf`-OFF matched pair per family to measure exact `perf` overhead) across diverse operator regimes: `ConvNeXt-Tiny`, `Swin-T`/`ViT-Tiny` (attention), `MobileNetV3-Large`, `ResNet-18/34/50`, `EfficientNet-B0/B2`, `RT-DETR`, `YOLO11/12`, plus `SmolLM2`/`Qwen2.5` (`prefill` vs `decode`) across `1500/2000/2400 MHz` + `ondemand`. | Breaks the **9-model (`85%` YOLO) Mode A bottleneck** and **5-model (`100%` YOLO) Mode L bottleneck**, cutting classifier `S-LOMO` error from `35%` to `< 10%` and giving conformal intervals sharp, honest coverage. |

---

## 5. Complete Index of Inspected Paths & Generated Audit Artifacts

### 5.1 Skill Definition Inspected
- `C:\Users\ishan\.cursor\plugins\cache\cursor-public\pstack\2dd5c0ccc5b207dc23a69759871a35954082cbf7\skills\principle-build-the-lever\SKILL.md`

### 5.2 Repository Files Inspected (`C:\Users\ishan\OneDrive - stevens.edu\Documents\ML2P\power-measurement-pi`)
- **Analytics & Modeling (`analytics/`)**: `analytics\pyproject.toml`, `analytics\config.toml`, `analytics\PROTOCOL.md`, `analytics\settings.py`, `analytics\data.py`, `analytics\qc.py`, `analytics\derive.py`, `analytics\static.py`, `analytics\tables.py`, `analytics\splits.py`, `analytics\evaluate.py`, `analytics\mode_a.py`, `analytics\mode_a_models.py`, `analytics\mode_a_extras.py`, `analytics\deploy.py`, `analytics\windows.py`, `analytics\mode_l.py`, `analytics\mode_l_models.py`, `analytics\lockbox.py`, `analytics\lockbox_checks.py`, `analytics\lockbox_score.py`, `analytics\reports\eda.md`, `analytics\reports\qc.md`, `analytics\reports\qc_cells.csv`, `analytics\reports\results.md`, `analytics\reports\model_card_mode_a.md`, `analytics\reports\model_card_mode_l.md`, `analytics\reports\mode_a_configs.csv`, `analytics\reports\lockbox.md`, `analytics\reports\lockbox_predictions.csv`, `analytics\reports\lockbox_serve_predictions.csv`, `analytics\reports\lockbox_frozen_models.json`, `analytics\reports\autopsy.md`.
- **Legacy Modeling (`modeling/`, `trace_forecast/`)**: `modeling\train_scaling.py`, `modeling\evaluate.py`, `trace_forecast\models.py`, `trace_forecast\evaluate.py`.
- **Pi & Host Measurement Stack (`src/`, `host/`, `tools/`, `docs/`)**: `src\run.py`, `src\powersampler.py`, `src\perfcounters.py`, `src\sysfs_sampler.py`, `src\layers.py`, `src\syncpulse.py`, `src\microbench.py`, `host\align.py`, `host\clockfit.py`, `host\recorder.py`, `host\verify_cell.py`, `host\package_dataset.py`, `tools\static_features.py`, `docs\dataset-v2.md`, `README.md`, `INSTRUCTIONS.md`.

### 5.3 Reproducible Audit Scripts & Empirical Outputs Created (`internal/power-audit/`)
All scratch scripts and empirical tables are preserved in the shared Project Agent Store:
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\materialize_datasets.py`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\audit_collection_and_physics.py`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\audit_pmic_and_cable.py`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\audit_mode_a_experiments.py`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\audit_mode_l_experiments.py`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\collection_physics_summary.json`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\pmic_and_cable_physics.json`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\cell_audit_metrics.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\cell_audit_metrics_with_pmic.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\dvfs_staleness_audit.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_a_audit_summary.json`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_a_per_model_slomo.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_a_lockbox_bakeoff.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_l_audit_summary.json`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_l_leakage_and_r2_audit.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_l_bakeoff_results.csv`
- `C:\Users\ishan\AppData\Local\Cursor\AgentStores\cursor_agent_stores\bc-4ec81833-c086-4076-9226-1cc12bd1f73b\files\internal\power-audit\mode_l_lockbox_per_cell.csv`
