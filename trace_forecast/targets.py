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
# Task C single-sample policy (explicit keep+guard, NOT observed in LM data:
# runs/v3_lm16,64,128 have 0 n==1 / 0 dur==0 / 0 E==0 segments, so no fix
# to build_phase_targets behavior). n==1 rows are KEPT with dur_s==0/E_mJ==0;
# train drops E<=0 (evaluate._build_train_items skips silently), eval
# forward_mape raises ZeroDivisionError on E==0 (never hit on LM data).
def is_valid_phase_target(row):
    try:
        return int(row.get("n", 0)) >= 2 and float(row.get("E_mJ", 0)) > 0 and float(row.get("dur_s", 0)) > 0
    except Exception:
        return False
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
