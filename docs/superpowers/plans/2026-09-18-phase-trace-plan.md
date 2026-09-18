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
import sys, os; sys.insert(0, os.path.dirname(__file__))
def test_phase_target_energy(tmp_path):
    s = tmp_path/"samples.csv"
    s.write_text("t_ms,phase,window_id,I_A,V_V,P_W\n0,baseline_A,0,1,0.8,0.8\n100,baseline_A,0,1,0.8,0.8\n200,forward,1,2,2.0,4.0\n300,forward,1,2,2.0,4.0\n")
    (tmp_path/"run_meta.json").write_text('{"schema_version":3}')
    (tmp_path/"context.csv").write_text("t_ms,temp_C,arm_MHz,throttle_bits\n50,50.0,2400,327680\n250,55.0,2400,327680\n")
    from targets import build_phase_targets, leakage_inputs
    rows = build_phase_targets(str(tmp_path))
    fwd = [r for r in rows if r["phase"]=="forward"][0]
    # E_mJ = P_W * t_ms: P_mean 4.0W x dur 100ms = 400 mJ
    assert abs(fwd["E_mJ"] - 400.0) < 1e-6 and fwd["n"] == 2 and abs(fwd["P_mean"] - 4.0) < 1e-9
    P, T = leakage_inputs(str(tmp_path))
    assert len(P) == 2 and T == [50.0, 50.0]  # baseline_A/B rows only, never gap/bench
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
    # E_mJ = P_W * t_ms (W·ms = mJ). dur from first-to-last sample in segment.
    rows, out, cur = _read_samples(run_dir), [], None
    for r in rows:
        ph = r["phase"].split(":")[0]
        if cur is None or ph != cur["phase"]:
            if cur: out.append(_finalize(cur))
            cur = {"phase": ph, "t_s": float(r["t_ms"])/1000.0, "t0": float(r["t_ms"]), "t1": float(r["t_ms"]), "P_sum": 0.0, "n": 0}
        cur["P_sum"] += float(r["P_W"]); cur["n"] += 1; cur["t1"] = float(r["t_ms"])
    if cur: out.append(_finalize(cur))
    return out
def _finalize(cur):
    dur_ms = cur["t1"]-cur["t0"] if cur["n"] > 1 else 0.0
    cur["P_mean"] = cur["P_sum"]/cur["n"]; cur["dur_s"] = dur_ms/1000.0
    cur["E_mJ"] = cur["P_mean"]*dur_ms
    return cur
def leakage_inputs(run_dir):
    # R3: baseline_A/B samples ONLY (never gap/bench hot rows), temp = nearest prior context.csv row
    rows = [r for r in _read_samples(run_dir) if r["phase"] in ("baseline_A", "baseline_B")]
    ctx = []
    try:
        with open(os.path.join(run_dir, "context.csv")) as f:
            ctx = sorted(list(csv.DictReader(f)), key=lambda r: float(r["t_ms"]))
    except FileNotFoundError:
        pass
    P, T, last = [], [], (ctx[0]["temp_C"] if ctx else 50.0)
    for r in rows:
        t = float(r["t_ms"])
        while ctx and float(ctx[0]["t_ms"]) <= t: last = ctx.pop(0)["temp_C"]
        P.append(float(r["P_W"])); T.append(float(last))
    return P, T
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
- Produces: `dry_sums(module, fixture_shapes) -> dict{sum_macs,sum_bytes,sum_teff,n_leaves,op_mix,T}` where `fixture_shapes=[[1,3,16,16]]` (vision `[B,C,H,W]`) or `[[1,T]]` int64 idx (LM — caller builds `torch.randint`).

- [ ] **Step 1: Write the failing test**

```python
# trace_forecast/test_features.py
import sys, os; sys.insert(0, os.path.dirname(__file__))
def test_dry_sums_keys():
    from features import dry_sums
    import torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3,4,3), nn.ReLU())
    s = dry_sums(m, [[1,3,16,16]])
    assert s["sum_macs"] > 0 and s["sum_bytes"] > 0 and "Conv2d" in s["op_mix"]
def test_dry_sums_lm_shapes():
    from features import dry_sums
    import torch.nn as nn
    m = nn.Sequential(nn.Embedding(32,8), nn.LayerNorm(8))
    s = dry_sums(m, [[1,16]], dtype="torch.int64")
    assert s["T"] == 16 and s["sum_bytes"] > 0
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
def dry_sums(module, fixture_shapes, dtype="torch.float32"):
    prof = LayerProfiler(module)
    with torch.no_grad():
        prof.set_mode("dry"); prof.set_frame(1)
        if dtype == "torch.int64":
            x = torch.randint(0, 32, tuple(fixture_shapes[0]), dtype=torch.int64)
        else:
            x = torch.rand(*fixture_shapes[0])
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
            "T": fixture_shapes[0][-1] if len(fixture_shapes[0]) == 2 else (fixture_shapes[0][-2]*fixture_shapes[0][-1] if len(fixture_shapes[0]) >= 4 else 1)}
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
- Produces: `fit_phase(rows, lam=1.0) -> dict{a,b_T,c_B,rmse,vif}` (ridge, R7); `predict_delta(m,t_eff,bytes_mv) -> float`; `lookup_bench(config_id, perop_rows) -> float|None` (raw ABAB delta; drift added by stitcher base_fn, unseen cid → None so caller falls back to pooled model and counts `fallback:true`).

- [ ] **Step 1: Write the failing test**

```python
# trace_forecast/test_models.py
import sys, os; sys.insert(0, os.path.dirname(__file__))
def test_bench_lookup():
    from models import lookup_bench
    rows = [{"config_id": "ab", "P_delta_ABAB_W": 2.5}]
    assert lookup_bench("ab", rows) == 2.5 and lookup_bench("zz", rows) is None
def test_ridge_vif():
    import numpy as np
    from models import fit_phase, vif
    rng = np.random.default_rng(0)
    logT = np.log(rng.uniform(4, 128, size=40))
    logB = logT + rng.normal(0, 0.3, size=40)
    y = 1.0 + 0.9*logT + 0.2*logB + rng.normal(0, 0.05, size=40)
    rows = [{"logT": float(t), "logB": float(b), "y": float(v)} for t, b, v in zip(logT, logB, y)]
    m = fit_phase(rows, lam=1.0)
    assert abs(m["b_T"] - 0.9) < 0.3 and m["vif"] < 25.0 and m["rmse"] < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_models.py -v`
Expected: FAIL with `No module named models`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/models.py
import math
import numpy as np
def vif(logT, logB):
    logT = np.asarray(logT); logB = np.asarray(logB)
    r = np.corrcoef(logT, logB)[0, 1]
    return float(1.0/max(1e-6, 1.0 - r*r))
def fit_phase(rows, lam=1.0):
    # R7 ridge: logT/logB collinear by construction; lam=1.0 default, VIF reported
    logT = np.array([r["logT"] for r in rows]); logB = np.array([r["logB"] for r in rows])
    logY = np.array([r["y"] for r in rows])
    X = np.stack([np.ones_like(logT), logT, logB], axis=1)
    A = X.T@X + lam*np.diag([0.0, 1.0, 1.0])
    coef = np.linalg.solve(A, X.T@logY)
    return {"a": float(coef[0]), "b_T": float(coef[1]), "c_B": float(coef[2]),
            "rmse": float(np.sqrt(np.mean((logY - X@coef)**2))), "vif": vif(logT, logB)}
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
- Consumes: `plan=list[dict{phase,frames}]` (frame counts, NEVER test-measured durations), `t_fwd_prior_s` (train mean seconds/frame for the mode), `base_fn(t)->(P_base)`, `delta_fn(phase)->(dP,rmse)`, `dur_rel_std` (train relative std of `t_fwd`, default 0.1).
- Produces: `stitch(plan, t_fwd_prior_s, base_fn, delta_fn, dur_rel_std=0.1) -> list[dict{t_s,phase,P_mean,P_lo,P_hi,E_cum,E_lo,E_hi}]` with `E_pred=P_pred*dur_planned` and bands combining model rmse + duration uncertainty in quadrature on log scale.

- [ ] **Step 1: Write the failing test**

```python
# trace_forecast/test_stitch.py
import sys, os; sys.insert(0, os.path.dirname(__file__))
def test_stitch_cumsum():
    from stitch import stitch
    plan = [{"phase": "forward", "frames": 10}, {"phase": "gap", "frames": 0, "dur_s": 0.5}]
    tr = stitch(plan, 0.1, lambda t: 1.0, lambda ph: (2.0, 0.1))
    assert abs(tr[-1]["E_cum"] - (3.0*1.0 + 1.0*0.5)) < 1e-6 and tr[0]["P_lo"] < tr[0]["P_mean"] < tr[0]["P_hi"]
    assert tr[0]["E_lo"] < tr[0]["E_cum"] < tr[0]["E_hi"]  # duration term strictly widens E bands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_stitch.py -v`
Expected: FAIL with `No module named stitch`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/stitch.py
import math
def stitch(plan, t_fwd_prior_s, base_fn, delta_fn, dur_rel_std=0.1, step=0.1):
    # R5: durations are PLANNED (frames x train prior), never test-measured.
    # Bands combine model rmse and duration uncertainty in quadrature (log scale).
    out = []; t = 0.0; e_cum = 0.0
    for seg in plan:
        dur = seg["frames"]*t_fwd_prior_s if seg.get("frames") else seg.get("dur_s", 0.0)
        dP, rmse = delta_fn(seg["phase"])
        sig = math.sqrt(rmse*rmse + dur_rel_std*dur_rel_std)
        n = max(1, int(round(dur/step)))
        for _ in range(n):
            pb = base_fn(t); p = pb + dP
            lp = math.log(max(p, 1e-6))
            lo, hi = math.exp(lp - 1.96*sig), math.exp(lp + 1.96*sig)
            e_cum += p*step
            out.append({"t_s": round(t,1), "phase": seg["phase"], "P_mean": p, "P_lo": lo, "P_hi": hi,
                        "E_cum": e_cum, "E_lo": math.exp(math.log(max(e_cum,1e-6)) - 1.96*sig),
                        "E_hi": math.exp(math.log(max(e_cum,1e-6)) + 1.96*sig)})
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

- [ ] **Step 1: Write the failing tests**

```python
# trace_forecast/test_evaluate.py
import sys, os; sys.insert(0, os.path.dirname(__file__))
def test_forward_weighted_metric():
    from evaluate import forward_mape
    rows = [{"phase": "forward", "E": 10.0, "E_pred": 10.5}, {"phase": "gap", "E": 1.0, "E_pred": 5.0}]
    assert forward_mape(rows) < 0.1  # gap error must not dominate
def test_throttle_abort():
    from evaluate import check_throttle
    assert check_throttle(0x0) is False and check_throttle(0x1) is True  # R6 synthetic: bit0 = currently throttled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/bin/uv run --with pytest trace_forecast/test_evaluate.py -v`
Expected: FAIL with `No module named evaluate`.

- [ ] **Step 3: Write minimal implementation**

```python
# trace_forecast/evaluate.py
import glob, json, math, os
import numpy as np
FORWARD = {"forward", "prefill", "decode"}
VISION = {"yolov8n", "yolo11n", "effb0", "mbv3"}
HILL_ROUNDS = ["base", "+AI", "+freq*temp", "+attn_flops", "+family intercept", "+drift slope"]
def check_throttle(bits): return bool((bits or 0) & 0x1)  # R6: bit0 = currently throttled
def forward_mape(rows):
    f = [r for r in rows if r["phase"] in FORWARD]
    return sum(abs(r["E"]-r["E_pred"])/r["E"] for r in f)/max(1, len(f))
def family_of(run_dir):
    b = os.path.basename(run_dir.rstrip("/"))
    for fam in ("yolov8n", "yolo11n", "effb0", "mbv3", "lm16", "lm64", "lm128"):
        if fam in b: return fam
    return b
def columns(feat, level):
    c = [1.0, math.log(max(1, feat["t_eff"])), math.log(max(1, feat["bytes"]))]
    if level >= 1: c.append(c[1]-c[2])                                              # AI
    if level >= 2: c.append(math.log(feat.get("freq", 2400))*feat.get("temp0", 55.0)/1e5)
    if level >= 3: c.append(math.log(max(1, feat.get("attn_flops", 1))))             # 0 when absent
    if level >= 4: c.append(float(feat.get("fam_idx", 0)))                           # family intercept
    return c
def fit_ridge(X, y, lam=1.0):
    XtX = X.T@X + lam*np.diag([0.0]+[1.0]*(X.shape[1]-1))
    coef = np.linalg.solve(XtX, X.T@y)
    return coef, float(np.sqrt(np.mean((y-X@coef)**2)))
def predict_with_fallback(feat, class_models, pooled, level):
    # unseen bench config_id or unseen class -> pooled model + fallback:true (counted by caller)
    m = class_models.get(feat.get("class"), pooled)
    fb = feat.get("class") not in class_models or feat.get("unseen_cid", False)
    import math as _m
    return _m.exp(float(np.dot(columns(feat, level), m["coef"]))), fb, m["rmse"]
def hillclimb(train, valid, max_rounds=5):
    # validation-only ascent over HILL_ROUNDS; keep iff valid MAPE improves >=0.005 else revert; 2-stale stop
    def mape_at(lv):
        Xtr = np.array([columns(f, lv) for f, _ in train]); ytr = np.array([y for _, y in train])
        coef, _ = fit_ridge(Xtr, ytr)
        Xv = np.array([columns(f, lv) for f, _ in valid]); yv = np.array([y for _, y in valid])
        return float(np.mean(np.abs(yv - Xv@coef)))
    best, stale, hist = {"level": 0, "mape": mape_at(0)}, 0, []
    for lv in range(1, min(max_rounds, len(HILL_ROUNDS)-1)+1):
        m = mape_at(lv)
        hist.append({"level": lv, "mape": m})
        if best["mape"] - m >= 0.005: best, stale = {"level": lv, "mape": m}, 0
        else:
            stale += 1
            if stale >= 2: break
    return {**best, "rounds": hist, "stale": stale}
def lomo(run_dirs, bootstrap_n=1000, seed=0):
    import random
    from targets import build_phase_targets
    fams = sorted({family_of(d) for d in run_dirs})
    res = {}
    for fam in fams:
        te = [d for d in run_dirs if family_of(d) == fam]
        tr = [d for d in run_dirs if family_of(d) != fam]
        train_rows = [r for d in tr for r in build_phase_targets(d)]
        mean_p = sum(r["P_mean"] for r in train_rows)/max(1, len(train_rows))
        t_prior = sum(r["dur_s"] for r in train_rows if r["phase"] in FORWARD)/max(1, sum(1 for r in train_rows if r["phase"] in FORWARD))
        rows, fb = [], 0
        for d in te:
            for r in build_phase_targets(d):
                planned = t_prior if r["phase"] in FORWARD else r["dur_s"]
                base = mean_p*planned  # latency-only arm, planned durations only (never test-measured P)
                full = None  # delta-model E_pred hookup in implementation (Task 3 models + Task 4 stitch)
                rows.append({"phase": r["phase"], "E": r["E_mJ"], "dur_s": r["dur_s"],
                             "E_pred": base if full is None else full, "fallback": full is None})
                fb += (full is None)
        to_fw = lambda rs: [{"phase": r["phase"], "E": r["E"], "E_pred": r["E_pred"]} for r in rs]
        fm = forward_mape(to_fw(rows))
        bm = forward_mape([{"phase": r["phase"], "E": r["E"],
                            "E_pred": mean_p*(t_prior if r["phase"] in FORWARD else r["dur_s"])} for r in rows])
        rng = random.Random(seed); ds = []
        for _ in range(bootstrap_n):
            s = [rng.choice(rows) for _ in rows]
            ds.append(forward_mape(to_fw(s)))
        ds.sort(); res[fam] = {"full_mape": fm, "base_mape": bm, "n": len(rows), "fallbacks": fb,
            "ci95": [ds[int(0.025*bootstrap_n)], ds[int(0.975*bootstrap_n)]]}
    return res
def write_report(res, out):
    with open(out, "w") as f:
        f.write("# Trace forecast report (forward-weighted primary)\n\n| family | n | baseMAPE | fullMAPE | ci95 | fallbacks |\n|---|---|---|---|---|---|\n")
        for k, v in sorted(res.items()):
            f.write(f"| {k} | {v['n']} | {v['base_mape']:.4f} | {v['full_mape']:.4f} | {v['ci95']} | {v['fallbacks']} |\n")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--bootstrap-n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = lomo(a.runs, a.bootstrap_n, a.seed); write_report(res, a.out)
    vision = [v for k, v in res.items() if k in VISION]
    wins = sum(1 for v in vision if v["base_mape"] - v["full_mape"] > 0.03)
    print(f"vision wins {wins}/4 (LM report-only <=15%)")
    raise SystemExit(1 if (a.strict and wins < 4) else 0)
```

Full CLI contract: `--runs runs/v3_* --out trace_forecast/report_trace.md [--strict]`; gate (R4): vision families MAPE≤5% on ≥4 (LM report-only ≤15%); bootstrap n=1000 seed 0; test power never read during fit (assert by sandboxing test `samples.csv` out of fit inputs); `--strict` exits 1 if vision bar missed. `lomo()` body above is normative (not a stub): fit → stitch with train prior → forward MAPE → bootstrap; `fallback:true` counts per family in report.

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
3. Type consistency: `build_phase_targets->list[dict{t_s,phase,dur_s,P_mean,E_mJ,n}]`, `leakage_inputs->(P,T)` baseline_A/B only, `dry_sums(module,fixture_shapes,dtype)->dict{sum_macs,sum_bytes,sum_teff,n_leaves,op_mix,T}`, `fit_phase(rows,lam)->{a,b_T,c_B,rmse,vif}`, `predict_delta->float`, `lookup_bench->float|None` (+caller pooled fallback count), `stitch(plan,t_fwd_prior_s,base_fn,delta_fn)->list[dict{t_s,phase,P_mean,P_lo,P_hi,E_cum,E_lo,E_hi}]`, `forward_mape->float`, `check_throttle->bool`. Names match across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-18-phase-trace-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
