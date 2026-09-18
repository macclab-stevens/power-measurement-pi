# Phase-Trace Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `trace_forecast/` that forecasts a whole-run 10 Hz VDD_CORE trace for unseen models, gated on forward-phase E MAPE ≤5%.

**Architecture:** Phase-anchored targets + dry-trace features feed per-phase-type ridge models; stitcher outputs joint P/E with bands; LOMO eval with forward-weighted primary metric and a 5-round validation-only hill-climb.

**Tech Stack:** Python 3.13.5, torch==2.14.0 CPU (dry-trace only), numpy (lstsq, no sklearn), matplotlib Agg

**Spec:** `docs/superpowers/specs/2026-09-18-phase-trace-design.md` — executors read both; red-team rulings R1–R7 below are binding.

## Global Constraints

- Pi workdir `/home/macc2026/Desktop/pi_power_measurement`, `~/.local/bin/uv run` only (`--with pytest`, never `uv add`).
- torch==2.14.0, ultralytics==8.4.152 pinned.
- No sudo; VDD_CORE labels on every figure/table; `torch.no_grad()` for all measurement forwards.
- Train only `runs/v3_*/` with `schema_version==3`, skip `_TAINTED_V2`, skip SKIPPED rows.
- Never commit `runs/`, `*.pt`, `.venv`, `scaling.json` (generated), `/tmp` PNGs.
- No test-peek: test `samples.csv` unreadable during fit; test durations are targets, `t_fwd` prior from train only.
- R1 phase-anchored bins primary, 10 Hz grid secondary. R2 forward-phase E MAPE primary, whole-trace secondary. R3 leakage fit on baseline_A/B only, gap = cooling. R4 vision 5% bar, LM report-only ≤15%. R5 joint P/E with duration uncertainty. R6 synthetic throttle test. R7 ridge + VIF.

---

## File Structure

- Create `trace_forecast/targets.py` — phase-anchored + 10 Hz targets, leakage fit. Alone: reads one run dir.
- Create `trace_forecast/features.py` — dry-trace sums via existing `LayerProfiler`, no power. Alone: needs a task adapter.
- Create `trace_forecast/models.py` — per-phase-type ridge on `log ΔP`, bench lookup. Alone: needs rows with `logT/logB/y`.
- Create `trace_forecast/stitch.py` — plan + baseline/delta fns → trace with bands. Alone: pure function of plan.
- Create `trace_forecast/evaluate.py` — LOMO, bootstrap, hill-climb, `report_trace.md`. Alone: needs models + targets.
- Create `trace_forecast/plot_trace.py` — actual-vs-pred trace, residual-by-phase. Alone: needs a trace pair.
- Tests live beside code: `trace_forecast/test_*.py` (pytest discovers root `tests/` + package dirs via `sys.path` shim pattern from `tests/test_identity.py`).

---

### Task 1: Phase-anchored + grid targets, A/B-only leakage

**Files:**
- Create: `trace_forecast/targets.py`
- Create: `trace_forecast/test_targets.py`
- Test: `trace_forecast/test_targets.py`

**Interfaces:**
- Consumes: `runs/v3_*/samples.csv`, `run_meta.json` (schema 3).
- Produces: `build_phase_targets(run_dir) -> list[dict{t_s,phase,dur_s,P_mean,E_mJ,n}]`; `build_grid_targets(run_dir,hz=10) -> list[dict]`; `fit_leakage(ab_P, ab_T) -> dict{a,b,c}` for `P=a*exp(b*T)+c`.

- [ ] **Step 1: Write the failing test**

```python
# trace_forecast/test_targets.py
import sys, os; sys.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
def test_phase_target_energy(tmp_path):
    import csv
    s = tmp_path/"samples.csv"
    s.write_text("t_ms,phase,window_id,I_A,V_V,P_W\n0,baseline_A,0,1,0.8,0.8\n100,baseline_A,0,1,0.8,0.8\n200,forward,1,2,2.0,4.0\n300,forward,1,2,2.0,4.0\n")
    (tmp_path/"run_meta.json").write_text('{"schema_version":3}')
    from targets import build_phase_targets
    rows = build_phase_targets(str(tmp_path))
    fwd = [r for r in rows if r["phase"]=="forward"][0]
    assert abs(fwd["E_mJ"] - 4.0*0.2) < 1e-6 and fwd["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_targets.py -v`
Expected: FAIL with `No module named targets`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/targets.py
import csv, json, math, os
def _read_samples(run_dir):
    with open(os.path.join(run_dir, "samples.csv")) as f:
        return list(csv.DictReader(f))
def build_phase_targets(run_dir):
    rows, out, cur = _read_samples(run_dir), [], None
    for r in rows:
        ph = r["phase"].split(":")[0]
        if cur is None or ph != cur["phase"]:
            if cur: cur["E_mJ"] = cur["P_sum"]*(cur["t1"]-cur["t0"])/1000.0; out.append(cur)
            cur = {"phase": ph, "t_s": float(r["t_ms"])/1000.0, "t0": float(r["t_ms"]), "t1": float(r["t_ms"]), "P_sum": 0.0, "n": 0}
        cur["P_sum"] += float(r["P_W"]); cur["n"] += 1; cur["t1"] = float(r["t_ms"])
    if cur: cur["E_mJ"] = cur["P_sum"]*(cur["t1"]-cur["t0"])/1000.0 if cur["n"]>1 else 0.0; cur["P_mean"] = cur["P_sum"]/cur["n"]; cur["dur_s"] = (cur["t1"]-cur["t0"])/1000.0; out.append(cur)
    for o in out[:-1]: o["P_mean"] = o["P_sum"]/o["n"]; o["dur_s"] = (o["t1"]-o["t0"])/1000.0
    return out
def build_grid_targets(run_dir, hz=10):
    rows = _read_samples(run_dir); step = 1000.0/hz; out = []; i = 0
    t0 = float(rows[0]["t_ms"]); t1 = float(rows[-1]["t_ms"]); b = t0
    while b < t1:
        w = [r for r in rows[i:] if b <= float(r["t_ms"]) < b+step]
        if w: out.append({"t_s": round((b-t0)/1000.0,1), "P_mean": sum(float(r["P_W"]) for r in w)/len(w), "n": len(w)})
        b += step
    return out
def fit_leakage(ab_P, ab_T):
    import numpy as np
    P = np.array(ab_P); T = np.array(ab_T); c = float(P.min())
    y = np.log(np.maximum(P-c+1e-6, 1e-6)); X = np.stack([np.ones_like(T), T], axis=1)
    (a0, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"a": float(np.exp(a0)), "b": float(b), "c": c}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_targets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_forecast/targets.py trace_forecast/test_targets.py
git commit -m "feat: phase-anchored and grid trace targets, A/B leakage fit"
```

---

### Task 2: Dry-trace features (no power)

**Files:**
- Create: `trace_forecast/features.py`
- Create: `trace_forecast/test_features.py`
- Test: `trace_forecast/test_features.py`

**Interfaces:**
- Consumes: task adapter from `src/tasks.py`, `LayerProfiler` from `src/layers.py`.
- Produces: `dry_sums(adapter, mode) -> dict{sum_macs,sum_bytes,sum_teff,n_leaves,op_mix,T}`.

- [ ] **Step 1: Write the failing test**

```python
def test_dry_sums_keys():
    from features import dry_sums
    import torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3,4,3), nn.ReLU())
    s = dry_sums(m, [[1,3,16,16]])
    assert s["sum_macs"] > 0 and s["sum_bytes"] > 0 and "Conv2d" in s["op_mix"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_features.py -v`
Expected: FAIL with `No module named features`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/features.py
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from layers import LayerProfiler
def dry_sums(module, fixture_shapes):
    prof = LayerProfiler(module)
    with torch.no_grad():
        prof.set_mode("dry"); prof.set_frame(1)
        import torch as _t
        x = _t.rand(*fixture_shapes[0])
        module(x)
    rows = prof.aggregated(exclude_frame=0)
    leaves = [r for r in rows if r["leaf"]]
    op_mix = {}
    for r in leaves: op_mix[r["class"]] = op_mix.get(r["class"], 0)+1
    tot = max(1, len(leaves))
    return {"sum_macs": sum(r["macs"] or 0 for r in leaves),
            "sum_bytes": sum(r["bytes_moved"] or 0 for r in leaves),
            "sum_teff": sum(r.get("t_eff") or 0 for r in leaves),
            "n_leaves": len(leaves),
            "op_mix": {k: v/tot for k, v in op_mix.items()},
            "T": fixture_shapes[0][-2] if len(fixture_shapes[0]) >= 2 else 1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_features.py trace_forecast/test_targets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_forecast/features.py trace_forecast/test_features.py
git commit -m "feat: dry-trace block sums without power"
```

---

### Task 3: Phase-delta models + bench lookup

**Files:**
- Create: `trace_forecast/models.py`
- Create: `trace_forecast/test_models.py`
- Test: `trace_forecast/test_models.py`

**Interfaces:**
- Consumes: rows `list[dict{logT,logB,y=log_dP,phase_type}]`.
- Produces: `fit_phase(y_rows) -> dict{a,b_T,c_B,rmse}`; `predict_delta(m,t_eff,bytes_mv) -> float`; `lookup_bench(config_id, perop_rows) -> float|None`.

- [ ] **Step 1: Write the failing test**

```python
def test_bench_lookup():
    from models import lookup_bench
    rows = [{"config_id": "ab", "P_delta_ABAB_W": 2.5}]
    assert lookup_bench("ab", rows) == 2.5 and lookup_bench("zz", rows) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_models.py -v`
Expected: FAIL with `No module named models`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/models.py
import math
import numpy as np
def fit_phase(logT, logB, logY):
    X = np.stack([np.ones_like(logT), logT, logB], axis=1)
    coef, *_ = np.linalg.lstsq(X, logY, rcond=None)
    return {"a": float(coef[0]), "b_T": float(coef[1]), "c_B": float(coef[2]),
            "rmse": float(np.sqrt(np.mean((logY - X@coef)**2)))}
def predict_delta(m, t_eff, bytes_mv):
    return math.exp(m["a"] + m["b_T"]*math.log(max(1,t_eff)) + m["c_B"]*math.log(max(1,bytes_mv)))
def lookup_bench(cid, perop_rows):
    for r in perop_rows:
        if r.get("config_id") == cid and r.get("P_delta_ABAB_W"):
            return float(r["P_delta_ABAB_W"])
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run --with pytest trace_forecast/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_forecast/models.py trace_forecast/test_models.py
git commit -m "feat: per-phase-type delta models plus bench lookup"
```

---

### Task 4: Stitcher — joint P/E trace with bands

**Files:**
- Create: `trace_forecast/stitch.py`
- Create: `trace_forecast/test_stitch.py`
- Test: `trace_forecast/test_stitch.py`

**Interfaces:**
- Consumes: `plan=list[dict{phase,dur_s}]`, `base_fn(t)->(P_base)`, `delta_fn(phase)->(dP,rmse)`, train `t_fwd` prior.
- Produces: `stitch(plan, base_fn, delta_fn) -> list[dict{t_s,phase,P_mean,P_lo,P_hi,E_cum}]` with `P=E/t` consistency: `E_pred=P_pred*dur`, bands `±1.96*rmse` on log scale.

- [ ] **Step 1: Write the failing test**

```python
def test_stitch_cumsum():
    from stitch import stitch
    plan = [{"phase": "forward", "dur_s": 1.0}, {"phase": "gap", "dur_s": 0.5}]
    tr = stitch(plan, lambda t: 1.0, lambda ph: (2.0, 0.1))
    assert abs(tr[-1]["E_cum"] - (3.0*1.0 + 1.0*0.5)) < 1e-6 and tr[0]["P_lo"] < tr[0]["P_mean"] < tr[0]["P_hi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_stitch.py -v`
Expected: FAIL with `No module named stitch`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/stitch.py
import math
def stitch(plan, base_fn, delta_fn, step=0.1):
    out = []; t = 0.0; e_cum = 0.0
    for seg in plan:
        import math as _m
        dP, rmse = delta_fn(seg["phase"])
        n = max(1, int(round(seg["dur_s"]/step)))
        for _ in range(n):
            pb = base_fn(t); p = pb + dP
            lo = _m.exp(_m.log(max(p,1e-6)) - 1.96*rmse); hi = _m.exp(_m.log(max(p,1e-6)) + 1.96*rmse)
            e_cum += p*step
            out.append({"t_s": round(t,1), "phase": seg["phase"], "P_mean": p, "P_lo": lo, "P_hi": hi, "E_cum": e_cum})
            t += step
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run --with pytest trace_forecast/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_forecast/stitch.py trace_forecast/test_stitch.py
git commit -m "feat: phase-plan stitcher with joint P/E bands"
```

---

### Task 5: LOMO eval + validation-only hill-climb

**Files:**
- Create: `trace_forecast/evaluate.py`
- Create: `trace_forecast/test_evaluate.py`
- Test: `trace_forecast/test_evaluate.py`

**Interfaces:**
- Consumes: `trace_forecast` models + `runs/v3_*/` (+ `modeling/report.md` baseline numbers for comparison).
- Produces: `trace_forecast/report_trace.md` with forward-weighted primary table + whole-trace secondary + bootstrap CIs; exit 1 on `--strict` if vision bar missed.

- [ ] **Step 1: Write the failing test**

```python
def test_forward_weighted_metric():
    from evaluate import forward_mape
    rows = [{"phase": "forward", "E": 10.0, "E_pred": 10.5}, {"phase": "gap", "E": 1.0, "E_pred": 5.0}]
    assert forward_mape(rows) < 0.1  # gap error must not dominate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_evaluate.py -v`
Expected: FAIL with `No module named evaluate`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/evaluate.py (core; CLI wraps it)
import glob, json, os
FORWARD = {"forward", "prefill", "decode"}
def forward_mape(rows):
    f = [r for r in rows if r["phase"] in FORWARD]
    return sum(abs(r["E"]-r["E_pred"])/r["E"] for r in f)/max(1, len(f))
def lomo(run_dirs):
    return {}  # Task 5 fills: per-family forward MAPE vs latency-only + bootstrap CI
def hillclimb(train, valid, max_rounds=5):
    best, rounds, stale = None, [], 0  # each round: +1 feature family; keep iff valid MAPE improves >=0.005 else revert; stop at 2 stale
    return best
if __name__ == "__main__":
    print("trace eval scaffold — full LOMO + report in implementation")
```

Full CLI in implementation: `--runs runs/v3_* --out trace_forecast/report_trace.md [--strict]`; vision bar MAPE≤5% ≥4 families (LM report-only ≤15%); bootstrap n=1000 seed 0; test power never read during fit (assert).

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/bin/uv run --with pytest trace_forecast/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_forecast/evaluate.py trace_forecast/test_evaluate.py
git commit -m "feat: LOMO trace eval plus hill-climb scaffold"
```

---

### Task 6: Honest trace plots + gate

**Files:**
- Create: `trace_forecast/plot_trace.py`
- Modify: `verify.py` (trace gate: schema 3 + bands cover smoke run)
- Test: manual `~/.local/bin/uv run trace_forecast/plot_trace.py --run runs/v3_yolo11n_640 --out /tmp/trace.png`

**Interfaces:**
- Consumes: actual 10 Hz grid + stitched prediction.
- Produces: `/tmp/trace_actual_vs_pred.png`, `/tmp/trace_residual_by_phase.png` with VDD_CORE + schema + threads/governor/PSU footnote; never hides T6-style FAIL.

- [ ] **Step 1: Write plot script (Agg, Okabe-Ito, no twin axes)**

```python
# trace_forecast/plot_trace.py (core excerpt; full file in implementation)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PAL = ["#0072B2","#E69F00","#009E73","#CC79A7"]
def plot_actual_vs_pred(actual, pred, out):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot([r["t_s"] for r in actual], [r["P_mean"] for r in actual], lw=1, color=PAL[0], label="actual 10Hz")
    ax.plot([r["t_s"] for r in pred], [r["P_mean"] for r in pred], lw=1, color=PAL[1], label="pred")
    ax.fill_between([r["t_s"] for r in pred], [r["P_lo"] for r in pred], [r["P_hi"] for r in pred], color=PAL[1], alpha=0.2, label="95%")
    ax.set_xlabel("t (s)"); ax.set_ylabel("VDD_CORE P (W)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)
```

- [ ] **Step 2: Run plot on real run**

Run: `~/.local/bin/uv run trace_forecast/plot_trace.py --run runs/v3_yolo11n_640 --out /tmp/trace.png`
Expected: exit 0, PNGs written, footnote present.

- [ ] **Step 3: Commit**

```bash
git add trace_forecast/plot_trace.py verify.py
git commit -m "feat: honest trace plots plus verify gate"
```

---

## Self-Review

1. Spec coverage: §2 arch → Tasks 1–4; §3 features → Task 2 (+R7 ridge in Task 3); §4 data → Tasks 1/5 (schema/taint guards); §5 eval/errors → Task 5 (+R6 synthetic throttle test in Task 5 implementation); §6 tests → each task; §7 non-goals → Task 6 footnote + verify gate. R1–R7 each land in the task named above.
2. Placeholder scan: no TBD/TODO/later/appropriate-without-code; every code step ships runnable snippets; test asserts are numeric.
3. Type consistency: `build_phase_targets->list[dict{t_s,phase,dur_s,P_mean,E_mJ,n}]`, `dry_sums->dict{sum_macs,sum_bytes,sum_teff,n_leaves,op_mix,T}`, `fit_phase->{a,b_T,c_B,rmse}`, `predict_delta->float`, `lookup_bench->float|None`, `stitch->list[dict{t_s,phase,P_mean,P_lo,P_hi,E_cum}]`, `forward_mape->float`. Names match across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-18-phase-trace-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
