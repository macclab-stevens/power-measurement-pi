# Review Revision (R2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Act on the verified findings of the 2026-11-23 external audit. Fix the collection code, and add an R2
analysis that is exploratory and post hoc with respect to the lockbox.

**Architecture:**
- R1 stays reproducible from tag `analytics-frozen`.
- `main` moves to R2:
  - new features and candidates are additive;
  - the selection rules change;
  - reports are regenerated, with R1 copies kept under `analytics/reports/r1_frozen/`.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, scipy (`analytics/`, `uv run pytest -q`); harness code in
`src/` and `host/` (repo-root tests: `uv run pytest -q tests`).

**Spec:** `docs/review/2026-11-24-audit-response.md`. Each section's "Actions" are the requirements.

## Global Constraints

- Data are read only from `~/pi-power-data/{v1,v2,lockbox,derived}`. Never modify files under `v1/`, `v2/` or
  `lockbox/`.
- Never re-run or overwrite the frozen lockbox scoring (`cli_lockbox score`), `reports/lockbox*.{md,csv,json}` or
  `derived/lockbox/bfff786a…*`.
- Never use lockbox cells for fitting or selection. R2 lockbox numbers are labelled "post hoc (R2)" everywhere.
- DVFS core-voltage table: 1500 → 0.7234 V, 2000 → 0.8745 V, 2400 → 0.9294 V. Linear interpolation between them;
  clamp outside.
- Master seed `20261122` (config.toml).
- No new dependencies.
- Match the surrounding code style: docstrings stating contracts, `settings`/`runlog` for paths and run logs.

## Review Focus

1. Lockbox isolation: no R2 fit or selection ever sees a lockbox cell.
2. Balanced selection: each held-out inner base model counts once, whatever its row count.
3. NaN and zero handling in the new window features (busy = 0, not counted).
4. Slip correction sign: after a loss of K samples, samples after the slip move later in time.
5. The calculator gap scenario never reads measured `serve_dur_ms`.

---

### Task 1: Collection-side fixes (repo root, worktree)

**Files:** `host/align.py`, `src/layers.py`, `src/run.py`, new `tools/hookbench.py` (copy from
`/home/macc-2026/.claude/jobs/45f7226a/tmp/review/pi/hookbench.py`), tests under `tests/`.

- [ ] **align.py.** Before `np.polyfit`, run `detect_slips` on the block stream. Add the cumulative step to the
  sample-clock times of samples after each slip: `t_m += Σ step_ns·[t_m >= slip_t_ns]`. Then fit `c0`/`c1` as now
  and record `slip_corrected: true` in `alignment.json`.
  - Test with a synthetic recording where K samples are lost mid-run: the post-fix `drift_ppm` stays near the true
    drift, and mid-run edge times are within 0.5 ms.
- [ ] **layers.py:370 and :414.** Weight bytes are `Σ p.numel() * p.element_size()`, not `params * 4`.
  - Test with an fp16 `nn.Linear`: bytes are half the fp32 ones.
- [ ] **run.py `--layer-hooks {on,off}`** (default `on`, so current behaviour is unchanged). With `off`:
  - no `LayerProfiler` hooks;
  - no `layer_table.csv` (log one line saying static features come from `tools/static_features.py`);
  - the mode/iter labels still reach the event log through `set_mode`/`set_frame`.

  Use a `NullProfiler` with the same `set_mode`/`set_frame` interface. Record `layer_hooks` in `run_meta.json`.
  - Test that the parser accepts both values and that `NullProfiler` writes the labels.
- [ ] Commit.

### Task 2: Mode A R2 (analytics)

**Files:** `analytics/static.py`, `analytics/tables.py`, `analytics/mode_a_models.py`, `analytics/mode_a.py`,
`analytics/evaluate.py`, `analytics/config.toml`, tests.

- [ ] **static.py.** Add `n_module_calls`: over ALL layer-table rows (containers and leaves), the sum of per-row
  multiplicity `round(count / min_positive_count)` divided by `batch`. This counts the per-image hook calls that
  the measured cells paid for.
- [ ] **tables.add_derived_columns.** Add:
  - `vcore_V`, from the DVFS table on `cpu_mhz`;
  - `v2f` = vcore_V² · cpu_ghz;
  - `log_n_calls` = log(n_module_calls).
- [ ] **evaluate.metrics.** Add `maxape`, `p90ape` and `r2_log` (R² of log y on log ŷ, 1 − SSE/SST). Add
  `interval_width(lo, hi)`, the median hi/lo.
- [ ] **mode_a_models.**
  - `FEATURES_R2` = FEATURES − {`log_weight_bytes`, `frac_macs_other`} + {`v2f`, `log_n_calls`}. M1Ridge and
    M2Trees use `FEATURES_R2 + [rig_nov]`.
  - New `M4Physics(seed)` for all three targets:
    - **Latency.** Non-negative least squares on additive cost terms, with u = 1/cpu_ghz and
      s = threads^beta: `[macs·u/s·(1−is_q), macs·u/s·is_int8, macs·u/s·is_fp16, act_bytes·u/s, n_module_calls·u, 1]`.
      Rows are scaled by 1/y (relative error) and by 1/sqrt(rows of that base model). Use `scipy.optimize.nnls`.
      Grid: beta ∈ {0.5, 0.75, 1.0}.
    - **Power.** Weighted ridge (relative error, same model balancing) of mean_P_W on
      `[1, v2f, v2f·threads, v2f·frac_macs_conv, v2f·log(act_bytes/macs), rig_nov, is_int8, is_fp16]`. Grid: alpha
      ∈ {0.01, 0.1, 1, 10, 100}.
    - **Energy** = latency × power. Each part is selected on its own target, like M3.
- [ ] **mode_a.** Replace `_mdape` selection with `_balanced`: the mean over inner folds of the fold median
  |log(p/y)| (+inf if not finite). Keep `_mdape` for the applicability threshold.
  - `AUTO_AMONG` energy = M1Ridge, M2Trees, M3Factored, M4Physics; latency and power = M1Ridge, M2Trees,
    M4Physics. `CANDIDATES_BY_TARGET` includes M4Physics.
  - Tests: balanced selection picks a grid point that a pooled-MdAPE rule would not, on a synthetic table with one
    large and one small model; M4 fit/predict shapes; NNLS coefficients ≥ 0.
- [ ] Run `uv run python cli.py mode-a` (or the existing command for the S-LOMO / RANDOM / S-LSO / S-LVO / S-LCO
  battery) and write the runs to `runs.jsonl` as usual. Commit.

### Task 3: Deployment calculator without leaked duration

**Files:** `analytics/deploy.py`, `analytics/lockbox_score.py` (new function only), `analytics/cli*.py`, tests.

- [ ] `deploy.predict` accepts `gap_s` in place of `duration_s`: duration = n · (t_img + gap_s + overhead_s).
  `overhead_s` is a parameter, by default `request_overhead_s(train_cells)`: the median over training serve cells
  of (serve_dur − n·latency − n·gap)/n, computed per family (yolo or other), with a fallback to the overall median.
- [ ] `deploy.validate`: score each serve cell both ways:
  - `duration_known`, which uses the measured duration; this is legitimate only for open-loop or fps cells;
  - `gap`, the honest closed-loop version.
- [ ] `lockbox_score.calculator_posthoc_gap(...)`: the frozen R1 predictions with the gap scenario. This is a new
  function; the frozen `calculator` is unchanged.
  - Write `reports/r2/lockbox_calculator_gap.csv`.
  - Expected median APE about 3.6 % with overhead 0 for comparison; also report the result with the estimated
    overhead.
- [ ] Commit.

### Task 4: Mode L R2

**Files:** `analytics/windows.py`, new `analytics/timefix.py`, `analytics/mode_l_models.py`, `analytics/mode_l.py`,
tests.

- [ ] **timefix.py.** `slip_correction_ns(alignment: dict, pulse_times_ns: np.ndarray, t_ns: np.ndarray)`
  implements the step model:
  - the true offset steps by +step at each slip (slip time on the Pi clock ≈ `t_ns − c0_ns`);
  - the fitted line is exact at the mean times of the first 6 and last 6 pulse events;
  - it returns the per-sample correction to add.

  Reference implementation: `/home/macc-2026/.claude/jobs/45f7226a/tmp/review/levers/slip_shear.py`, `corr_fn`.
  - Test with one slip in the middle: correction = −step/2 at the start side of the slip and +step/2 just after it,
    when the slip is midway.
- [ ] **windows.py.**
  - Sample times are corrected with timefix before window labels are built (cells with slips only).
  - New features:
    - `llc_miss_rate` (LLC_load_misses/s);
    - `f_eff_mhz` = cycles_rate / (N_CORES · busy) / 1e6 when busy ≥ 0.03, else the sysfs clock (`freq_mhz`), and
      clamped to [600, 2400];
    - `vcore_eff` (DVFS table on f_eff);
    - `v2_cycles_rate` and `v2_instr_rate` (vcore_eff² × rate).

    New features are NaN-safe; not-counted windows give 0 rates.
  - Bump the windows cache key.
- [ ] **mode_l_models.**
  - Rungs `L5-cmos` (ridge: L3 ridge features − `ctx_sw_rate` + the five new ones) and `L5-cmos-trees` (same
    features, HistGradientBoosting with the L3-trees grid).
  - Remove `ctx_sw_rate` from every feature list.
- [ ] **mode_l._select.** Select on the median inference-segment APE (`ape_meas` of `_score_cell`), not whole-cell.
  Report adds forward-window MAE (W) and R² on forward windows only.
- [ ] Re-run Mode L (all rungs, leave-one-YOLO-model-out). Commit.

### Task 5: Post-hoc lockbox scoring of R2, reports and protocol

- [ ] Fit R2 finals on training cells only and score the 5 lockbox configurations and 15 cells (Mode A all
  candidates, AUTO; Mode L all rungs). Write to `reports/r2/lockbox_posthoc.md` and CSVs, each table labelled
  "post hoc (R2): the lockbox was seen".
- [ ] Copy the current `reports/*` to `reports/r1_frozen/` (unchanged). Regenerate `results.md`, the model cards and
  `eda.md` under R2.
  - `eda.md` gets the fit P = P0 + k·V(f)²·f.
  - Every MdAPE gets MaxAPE / P90 and interval width beside it.
- [ ] `reports/measurement_audit.md`: the levers in `analytics/audit/` with their numbers (hooks, slips, cable, ctx,
  vcore, calculator).
- [ ] `PROTOCOL.md` amendment "R2 (2026-11-24, post-review, exploratory)". Commit.

### Task 6: Next-collection recommendations

- [ ] `docs/review/next-collection.md` covers:
  - hooks off;
  - the align fix;
  - a short thick harness;
  - Vout fixed;
  - `perf_event_paranoid` (root, needs approval);
  - sysfs at 20 Hz;
  - ≥ 20 base models;
  - a hooks-off calibration subset;
  - a fresh lockbox for R2.
- [ ] Legacy notes in `modeling/` and `trace_forecast/`.
- [ ] Commit.
