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
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default
def _read_env(run_dir):
    # Env for feat: temp0/freq. Brief says run_meta.json, but shipping metas carry
    # no temp/freq keys, so temp/freq come from context.csv first row (pre-phase env0);
    # run_meta still read for seq_len/task (no power). Never reads samples.csv.
    meta = {}
    try:
        with open(os.path.join(run_dir, "run_meta.json")) as f:
            meta = json.load(f)
    except FileNotFoundError:
        pass
    temp0, freq = 55.0, 2400.0
    try:
        import csv as _csv
        with open(os.path.join(run_dir, "context.csv")) as f:
            rd = _csv.DictReader(f)
            for row in rd:
                temp0 = _safe_float(row.get("temp_C", temp0), temp0)
                freq = _safe_float(row.get("arm_MHz", freq), freq)
                break
    except FileNotFoundError:
        pass
    return {"temp0": float(temp0), "freq": float(freq), "meta": meta}
def _layer_sums_for_phase(run_dir, phase):
    # Aggregate FORWARD-phase leaves from layer_table.csv only (no weights).
    # Per (run, phase): sums over leaves with mode==phase; fallback to all FORWARD modes.
    import csv as _csv
    sums = {"sum_macs": 0.0, "sum_bytes": 0.0, "sum_teff": 0.0, "n_leaves": 0, "op_mix": {}, "T": 1}
    try:
        with open(os.path.join(run_dir, "layer_table.csv")) as f:
            rows = list(_csv.DictReader(f))
    except FileNotFoundError:
        return sums
    leaves = [r for r in rows if str(r.get("leaf", "")).lower() == "true"]
    ph = [r for r in leaves if r.get("mode", "") == phase]
    if not ph:
        ph = [r for r in leaves if r.get("mode", "") in FORWARD]
    sm = sum(_safe_float(r.get("macs", 0)) for r in ph)
    sb = sum(_safe_float(r.get("bytes_moved", 0)) for r in ph)
    st = sum(_safe_float(r.get("t_eff", 0)) for r in ph)
    from collections import Counter as _C
    ctr = _C(r.get("class", "?") for r in ph)
    tot = max(1, len(ph))
    op_mix = {k: v / tot for k, v in ctr.items()}
    T = 1
    try:
        cands = [int(float(r.get("seq_len", 0) or 0)) for r in ph if str(r.get("seq_len", "")).strip() not in ("", "None")]
        cands = [v for v in cands if v > 0]
        if cands:
            T = max(cands)
    except Exception:
        pass
    return {"sum_macs": sm, "sum_bytes": sb, "sum_teff": st, "n_leaves": len(ph), "op_mix": op_mix, "T": T}
def _feat_for_run(run_dir, phase, fam_idx):
    s = _layer_sums_for_phase(run_dir, phase)
    e = _read_env(run_dir)
    meta = e.get("meta", {}) or {}
    T = s.get("T", 1)
    try:
        mseq = int(meta.get("seq_len", 0) or 0)
        if T <= 1 and mseq > 0:
            T = mseq
    except Exception:
        pass
    return {"t_eff": max(1.0, float(s["sum_teff"])), "bytes": max(1.0, float(s["sum_bytes"])),
            "macs": float(s["sum_macs"]), "op_mix": s["op_mix"], "T": T,
            "temp0": float(e["temp0"]), "freq": float(e["freq"]),
            "attn_flops": 0.0, "fam_idx": float(fam_idx),
            "class": phase, "phase": phase, "unseen_cid": False}
def _build_train_items(train_dirs, fam_index):
    # TRAIN ONLY: phase E via build_phase_targets + layer/env feats. No test file read.
    try:
        from targets import build_phase_targets as _bpt
    except ImportError:
        from trace_forecast.targets import build_phase_targets as _bpt  # type: ignore
    items = []
    for d in train_dirs:
        for r in _bpt(d):
            if r.get("phase") not in FORWARD:
                continue
            E = _safe_float(r.get("E_mJ", 0), 0.0)
            if E <= 0:
                continue
            fam = family_of(d)
            feat = _feat_for_run(d, r["phase"], fam_index.get(fam, 0))
            items.append((feat, math.log(E), E, r["phase"]))
    return items
def _fit_at_level(pairs, level):
    # pairs: list of (feat, logE). Level 0 via models.fit_phase, >=1 via fit_ridge.
    if not pairs:
        return None
    if level == 0:
        try:
            from models import fit_phase as _fp
        except Exception:
            from trace_forecast.models import fit_phase as _fp  # type: ignore
        rows = [{"logT": math.log(max(1.0, float(f.get("t_eff", 1)))), "logB": math.log(max(1.0, float(f.get("bytes", 1)))), "y": float(y)} for f, y in pairs]
        m = _fp(rows, lam=1.0)
        import numpy as _np
        return {"coef": _np.array([m["a"], m["b_T"], m["c_B"]], dtype=float), "rmse": float(m["rmse"])}
    X = np.array([columns(f, level) for f, _ in pairs])
    y = np.array([float(v) for _, v in pairs])
    coef, rmse = fit_ridge(X, y)
    return {"coef": coef, "rmse": rmse}
def _fit_per_phase(items, level):
    # Per-phase-type ridge + pooled fallback. items: (feat, logE, E, phase).
    by_phase = {}
    for feat, logE, _E, ph in items:
        by_phase.setdefault(ph, []).append((feat, logE))
    class_models = {}
    for ph, pairs in by_phase.items():
        m = _fit_at_level(pairs, level)
        if m is not None:
            class_models[ph] = m
    pooled = _fit_at_level([(f, y) for f, y, _e, _p in items], level)
    return class_models, pooled
def _choose_level(items):
    # Validation-only hill-climb on TRAIN items only (no test peek). Deterministic
    # split: every 3rd item -> valid, rest -> train. <4 items -> level 0.
    if len(items) < 4:
        return 0
    pairs = [(f, y) for f, y, _e, _p in items]
    valid = [pairs[i] for i in range(len(pairs)) if i % 3 == 2]
    train = [pairs[i] for i in range(len(pairs)) if i % 3 != 2]
    if len(train) < 2 or len(valid) < 1:
        return 0
    try:
        best = hillclimb(train, valid, max_rounds=5)
        return int(best.get("level", 0))
    except Exception:
        return 0
def lomo(run_dirs, bootstrap_n=1000, seed=0):
    import random
    try:
        from targets import build_phase_targets
    except ImportError:
        from trace_forecast.targets import build_phase_targets  # type: ignore
    fams = sorted({family_of(d) for d in run_dirs})
    fam_index = {f: i for i, f in enumerate(fams)}
    res = {}
    for fam in fams:
        te = [d for d in run_dirs if family_of(d) == fam]
        tr = [d for d in run_dirs if family_of(d) != fam]
        # ---- TRAIN ONLY (sandbox): build all train structures before touching te dirs.
        train_pt = [r for d in tr for r in build_phase_targets(d)]
        mean_p = sum(r["P_mean"] for r in train_pt)/max(1, len(train_pt))
        fwd_tr = [r for r in train_pt if r["phase"] in FORWARD]
        t_prior = sum(r["dur_s"] for r in fwd_tr)/max(1, len(fwd_tr))
        train_items = _build_train_items(tr, fam_index)
        level = _choose_level(train_items)
        if train_items:
            class_models, pooled = _fit_per_phase(train_items, level)
        else:
            class_models, pooled = {}, None
        # ---- EVAL ONLY: now touch te dirs for targets + test feats (never used in fit).
        rows, fb = [], 0
        for d in te:
            for r in build_phase_targets(d):
                planned = t_prior if r["phase"] in FORWARD else r["dur_s"]
                base_mJ = mean_p*planned*1000.0  # mJ (P_W * s * 1000); scaffold omitted *1000 (J-scale)
                if r["phase"] in FORWARD and pooled is not None:
                    feat_te = _feat_for_run(d, r["phase"], fam_index.get(fam, 0))
                    try:
                        full, is_fb, _rmse = predict_with_fallback(feat_te, class_models, pooled, level)
                    except Exception:
                        full, is_fb = base_mJ, True
                    rows.append({"phase": r["phase"], "E": r["E_mJ"], "dur_s": r["dur_s"],
                                 "E_pred": full, "fallback": is_fb})
                    fb += (1 if is_fb else 0)
                else:
                    is_fb = True if (r["phase"] in FORWARD and pooled is None) else False
                    rows.append({"phase": r["phase"], "E": r["E_mJ"], "dur_s": r["dur_s"],
                                 "E_pred": base_mJ, "fallback": is_fb})
                    if r["phase"] in FORWARD:
                        fb += (1 if is_fb else 0)
        to_fw = lambda rs: [{"phase": r["phase"], "E": r["E"], "E_pred": r["E_pred"]} for r in rs]
        fm = forward_mape(to_fw(rows))
        bm = forward_mape([{"phase": r["phase"], "E": r["E"],
                            "E_pred": mean_p*(t_prior if r["phase"] in FORWARD else r["dur_s"])*1000.0} for r in rows])
        rng = random.Random(seed); ds = []
        for _ in range(bootstrap_n):
            s = [rng.choice(rows) for _ in rows]
            ds.append(forward_mape(to_fw(s)))
        ds.sort(); res[fam] = {"full_mape": fm, "base_mape": bm, "n": len(rows), "fallbacks": fb,
            "ci95": [ds[int(0.025*bootstrap_n)], ds[int(0.975*bootstrap_n)]], "level": level}
    return res
def write_report(res, out):
    with open(out, "w") as f:
        f.write("# Trace forecast report (forward-weighted primary)\n\n| family | n | baseMAPE | fullMAPE | ci95 | fallbacks | level |\n|---|---|---|---|---|---|---|\n")
        for k, v in sorted(res.items()):
            f.write(f"| {k} | {v['n']} | {v['base_mape']:.4f} | {v['full_mape']:.4f} | {v['ci95']} | {v['fallbacks']} | {v.get('level', 0)} |\n")
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
