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
        f.write("# Trace forecast report (forward-weighted primary)\n\n| family | n | baseMAPE | fullMAPE | ci95 | fallbacks |\n|---|---|---|---|---|--- |\n")
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
