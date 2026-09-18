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
    # Kept for backward compat (Task A every-3rd split). Task B lomo() uses
    # _choose_level_loto (leave-one-train-family-out, no test peek). Do not use for LOMO.
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
def _unpack_item(it):
    # Accept (feat, logE) or (feat, logE, E, phase). Return (feat, logE, E, phase).
    if len(it) == 2:
        f, y = it
        try:
            import math as _m
            E = _m.exp(float(y))
        except Exception:
            E = 0.0
        ph = f.get("class", "forward") if isinstance(f, dict) else "forward"
        return f, float(y), float(E), ph
    f, y, E, ph = it[0], it[1], it[2], it[3]
    return f, float(y), float(E), ph
def _loto_avg_valid_mape(per_fam, level):
    # Leave-one-train-family-out averaged valid MAPE (energy, forward-weighted).
    # per_fam: dict family -> list of items (TRAIN ONLY, test family never present).
    # Fit per-phase-type + pooled at `level` on train families \\ {g}, predict g.
    # No bootstrap in inner loop (fast); final CIs stay n=1000. No test peek.
    import math as _m
    fams = sorted(per_fam.keys())
    if len(fams) < 2:
        return float("nan"), {}
    per_fold = {}
    for g in fams:
        train_pairs = []
        for ff in fams:
            if ff == g:
                continue
            for it in per_fam[ff]:
                f, y, _E, _ph = _unpack_item(it)
                train_pairs.append((f, y))
        valid_items = [_unpack_item(it) for it in per_fam[g]]
        if not train_pairs or not valid_items:
            continue
        by_phase = {}
        for f, y in train_pairs:
            by_phase.setdefault(f.get("class", "forward"), []).append((f, y))
        cms = {}
        for ph, pairs in by_phase.items():
            m = _fit_at_level(pairs, level)
            if m is not None:
                cms[ph] = m
        pooled = _fit_at_level(train_pairs, level)
        if pooled is None:
            continue
        rows = []
        for f, _y, E, _ph in valid_items:
            try:
                pred, _fb, _rm = predict_with_fallback(f, cms, pooled, level)
            except Exception:
                continue
            rows.append({"phase": "forward", "E": E, "E_pred": pred})
        if not rows:
            continue
        per_fold[g] = forward_mape(rows)
    if not per_fold:
        return float("nan"), {}
    avg = float(sum(per_fold.values()) / max(1, len(per_fold)))
    return avg, per_fold
def _choose_level_loto(per_fam, max_rounds=5):
    # Validation-only hill-climb over TRAIN families only (LOTO averaged valid MAPE).
    # Levels 0..5 per HILL_ROUNDS; keep iff improves >=0.005 else revert; 2-stale stop.
    # Returns {level, mape, curve=[{level,mape}...], rounds (compat), stale}.
    # No test touch: per_fam must contain TRAIN families only.
    import math as _m
    max_lv = min(max_rounds, len(HILL_ROUNDS) - 1)
    # Need >=2 train families and >=4 total rows, else level 0.
    tot = sum(len(v) for v in per_fam.values())
    if len(per_fam) < 2 or tot < 4:
        return {"level": 0, "mape": float("nan"), "curve": [{"level": 0, "mape": float("nan")}], "rounds": [], "stale": 0}
    try:
        m0, _pf0 = _loto_avg_valid_mape(per_fam, 0)
    except Exception:
        return {"level": 0, "mape": float("nan"), "curve": [{"level": 0, "mape": float("nan")}], "rounds": [], "stale": 0}
    # nan guard: if avg is nan (e.g. single-row folds), fall back to level 0.
    try:
        if m0 != m0:
            return {"level": 0, "mape": m0, "curve": [{"level": 0, "mape": m0}], "rounds": [], "stale": 0}
    except Exception:
        pass
    best, stale = {"level": 0, "mape": m0}, 0
    curve = [{"level": 0, "mape": m0}]
    hist = []
    for lv in range(1, max_lv + 1):
        try:
            m, _pf = _loto_avg_valid_mape(per_fam, lv)
        except Exception:
            m = float("nan")
        curve.append({"level": lv, "mape": m})
        hist.append({"level": lv, "mape": m})
        try:
            improves = (best["mape"] - m) >= 0.005
        except Exception:
            improves = False
        if improves:
            best, stale = {"level": lv, "mape": m}, 0
        else:
            stale += 1
            if stale >= 2:
                break
    return {"level": int(best["level"]), "mape": float(best["mape"]), "curve": curve, "rounds": hist, "stale": int(stale)}
def _loto_avg_valid_mape_perphase(per_fam, level_by_phase, default_level=0):
    # Overall LOTO valid MAPE when each phase type uses its own level.
    # Used for mbv3 ablation (global vs per-phase-type). TRAIN ONLY.
    import math as _m
    fams = sorted(per_fam.keys())
    if len(fams) < 2:
        return float("nan"), {}
    per_fold = {}
    for g in fams:
        train_by_phase = {}
        for ff in fams:
            if ff == g:
                continue
            for it in per_fam[ff]:
                f, y, _E, ph = _unpack_item(it)
                train_by_phase.setdefault(ph, []).append((f, y))
        cms = {}
        for ph, pairs in train_by_phase.items():
            lv = int(level_by_phase.get(ph, default_level))
            m = _fit_at_level(pairs, lv)
            if m is not None:
                cms[ph] = m
        # pooled fallback at default_level (rare; phases always seen in our 1-run/family data)
        all_train = [(f, y) for pairs in train_by_phase.values() for f, y in pairs]
        pooled = _fit_at_level(all_train, default_level)
        if pooled is None and not cms:
            continue
        rows = []
        for it in per_fam[g]:
            f, _y, E, ph = _unpack_item(it)
            lv = int(level_by_phase.get(ph, default_level))
            m = cms.get(ph, pooled)
            if m is None:
                continue
            try:
                import numpy as _np
                pred = _m.exp(float(_np.dot(columns(f, lv), m["coef"])))
            except Exception:
                continue
            rows.append({"phase": "forward", "E": E, "E_pred": pred})
        if not rows:
            continue
        per_fold[g] = forward_mape(rows)
    if not per_fold:
        return float("nan"), {}
    return float(sum(per_fold.values()) / max(1, len(per_fold))), per_fold
def _choose_level_per_phase(per_fam, max_rounds=5):
    # Per-phase-type LOTO level choice (TRAIN ONLY). For each phase present,
    # run _choose_level_loto on that phase's rows only. Returns {phase: {level,curve,stale}}.
    phases = set()
    for items in per_fam.values():
        for it in items:
            _f, _y, _E, ph = _unpack_item(it)
            phases.add(ph)
    out = {}
    for ph in sorted(phases):
        sub = {}
        for fam, items in per_fam.items():
            filt = [it for it in items if _unpack_item(it)[3] == ph]
            if filt:
                sub[fam] = filt
        # Need >=2 families with this phase and >=2 rows total, else level 0.
        nfam = len(sub)
        tot = sum(len(v) for v in sub.values())
        if nfam < 2 or tot < 2:
            out[ph] = {"level": 0, "mape": float("nan"), "curve": [], "rounds": [], "stale": 0}
            continue
        try:
            out[ph] = _choose_level_loto(sub, max_rounds=max_rounds)
        except Exception:
            out[ph] = {"level": 0, "mape": float("nan"), "curve": [], "rounds": [], "stale": 0}
    return out
def _fit_per_phase_levels(items, level_by_phase, default_level=0):
    # Fit per-phase-type models each at its own level. items: (feat, logE, E, phase).
    by_phase = {}
    for feat, logE, _E, ph in items:
        by_phase.setdefault(ph, []).append((feat, logE))
    cms = {}
    for ph, pairs in by_phase.items():
        lv = int(level_by_phase.get(ph, default_level))
        m = _fit_at_level(pairs, lv)
        if m is not None:
            cms[ph] = m
    pooled = _fit_at_level([(f, y) for f, y, _e, _p in items], default_level)
    return cms, pooled
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
        # No test-peek: inner valid families are train-subset; test family never in fit/level choice.
        # fam_index is global names-only (no power/duration); rows are train-only.
        train_pt = [r for d in tr for r in build_phase_targets(d)]
        mean_p = sum(r["P_mean"] for r in train_pt)/max(1, len(train_pt))
        fwd_tr = [r for r in train_pt if r["phase"] in FORWARD]
        t_prior = sum(r["dur_s"] for r in fwd_tr)/max(1, len(fwd_tr))
        # Per-train-family items for LOTO (TRAIN ONLY).
        per_fam_train = {}
        for tf in sorted({family_of(d) for d in tr}):
            tdirs = [d for d in tr if family_of(d) == tf]
            per_fam_train[tf] = _build_train_items(tdirs, fam_index)
        train_items = [it for items in per_fam_train.values() for it in items]
        # Global LOTO hill-climb (levels 0..5, >=0.005, 2-stale). No bootstrap in inner loop (fast).
        try:
            gchoice = _choose_level_loto(per_fam_train, max_rounds=5)
        except Exception:
            gchoice = {"level": 0, "mape": float("nan"), "curve": [], "rounds": [], "stale": 0}
        level = int(gchoice.get("level", 0))
        valid_curve = gchoice.get("curve", [])
        valid_stale = int(gchoice.get("stale", 0))
        valid_mape = float(gchoice.get("mape", float("nan"))) if gchoice.get("mape", None) == gchoice.get("mape", None) else float("nan")
        # Per-phase-type ablation (TRAIN ONLY). Adopt iff wins validation by >=0.005.
        try:
            per_phase_choice = _choose_level_per_phase(per_fam_train, max_rounds=5)
        except Exception:
            per_phase_choice = {}
        per_phase_levels = {ph: int(v.get("level", 0)) for ph, v in per_phase_choice.items()}
        try:
            perphase_valid, _pf = _loto_avg_valid_mape_perphase(per_fam_train, per_phase_levels, default_level=level)
        except Exception:
            perphase_valid = float("nan")
        adopted_perphase = False
        try:
            if perphase_valid == perphase_valid and valid_mape == valid_mape:
                if (valid_mape - perphase_valid) >= 0.005:
                    adopted_perphase = True
        except Exception:
            adopted_perphase = False
        # Final refit: global unless per-phase wins validation (then per-phase levels).
        if train_items:
            if adopted_perphase:
                class_models, pooled = _fit_per_phase_levels(train_items, per_phase_levels, default_level=level)
                fit_level_for_pred = per_phase_levels
            else:
                class_models, pooled = _fit_per_phase(train_items, level)
                fit_level_for_pred = level
        else:
            class_models, pooled = {}, None
            fit_level_for_pred = level
        # ---- EVAL ONLY: now touch te dirs for targets + test feats (never used in fit).
        rows, fb = [], 0
        for d in te:
            for r in build_phase_targets(d):
                planned = t_prior if r["phase"] in FORWARD else r["dur_s"]
                base_mJ = mean_p*planned*1000.0  # mJ (P_W * s * 1000); scaffold omitted *1000 (J-scale)
                if r["phase"] in FORWARD and pooled is not None:
                    feat_te = _feat_for_run(d, r["phase"], fam_index.get(fam, 0))
                    try:
                        if isinstance(fit_level_for_pred, dict):
                            lv_use = int(fit_level_for_pred.get(r["phase"], level))
                            # per-phase predict with that phase's model
                            import math as _mm
                            import numpy as _nnp
                            m_use = class_models.get(r["phase"], pooled)
                            full = _mm.exp(float(_nnp.dot(columns(feat_te, lv_use), m_use["coef"])))
                            is_fb = (r["phase"] not in class_models)
                        else:
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
        fw_rows = [r for r in rows if r["phase"] in FORWARD]
        if fw_rows:
            for _ in range(bootstrap_n):
                s = [rng.choice(fw_rows) for _ in fw_rows]
                ds.append(forward_mape(to_fw(s)))
        else:
            ds = [0.0] * bootstrap_n
        ds.sort()
        fwd_actual = sum(r["dur_s"] for r in rows if r["phase"] in FORWARD) / max(1, sum(1 for r in rows if r["phase"] in FORWARD))
        dur_ratio = (fwd_actual / t_prior) if t_prior > 0 else 0.0
        # Duration diagnostics (eval-only, never in fit): oracle base uses actual duration;
        # full_durnorm compares E_pred vs E_actual/dur_ratio (train-duration scale) per brief literal.
        try:
            base_oracle = forward_mape([{"phase": r["phase"], "E": r["E"],
                "E_pred": mean_p*(fwd_actual if r["phase"] in FORWARD else r["dur_s"])*1000.0} for r in rows])
        except Exception:
            base_oracle = float("nan")
        try:
            if dur_ratio and dur_ratio == dur_ratio and dur_ratio != 0:
                full_durnorm = forward_mape([{"phase": r["phase"], "E": (r["E"]/dur_ratio), "E_pred": r["E_pred"]} for r in rows if r["phase"] in FORWARD])
            else:
                full_durnorm = float("nan")
        except Exception:
            full_durnorm = float("nan")
        # mbv3 ablation helper: lv0-only test MAPE (eval-only, does not affect choice).
        try:
            if train_items:
                _cm0, _po0 = _fit_per_phase(train_items, 0)
                _rows0 = []
                for d in te:
                    for r in build_phase_targets(d):
                        if r["phase"] in FORWARD and _po0 is not None:
                            _ft = _feat_for_run(d, r["phase"], fam_index.get(fam, 0))
                            try:
                                _p0, _fb0, _rm0 = predict_with_fallback(_ft, _cm0, _po0, 0)
                            except Exception:
                                _p0 = mean_p*t_prior*1000.0
                            _rows0.append({"phase": r["phase"], "E": r["E_mJ"], "E_pred": _p0})
                full_mape_lv0 = forward_mape([{"phase": r["phase"], "E": r["E"], "E_pred": r["E_pred"]} for r in _rows0]) if _rows0 else float("nan")
            else:
                full_mape_lv0 = float("nan")
        except Exception:
            full_mape_lv0 = float("nan")
        res[fam] = {"full_mape": fm, "base_mape": bm, "n": len(rows), "fallbacks": fb,
            "ci95": [ds[int(0.025*bootstrap_n)], ds[int(0.975*bootstrap_n)]], "level": level,
            "mean_p": mean_p, "t_prior": t_prior, "fwd_actual": fwd_actual, "dur_ratio": dur_ratio,
            "base_mape_oracle": float(base_oracle), "full_mape_durnorm": float(full_durnorm),
            "valid_curve": valid_curve, "valid_stale": valid_stale, "valid_mape": float(valid_mape) if valid_mape==valid_mape else float("nan"),
            "per_phase_levels": per_phase_levels, "perphase_valid": float(perphase_valid) if perphase_valid==perphase_valid else float("nan"),
            "adopted_perphase": bool(adopted_perphase), "full_mape_lv0": float(full_mape_lv0) if full_mape_lv0==full_mape_lv0 else float("nan")}
    return res
def write_report(res, out):
    with open(out, "w") as f:
        f.write("# Trace forecast report (forward-weighted primary)\n\n| family | n | baseMAPE | fullMAPE | ci95 | fallbacks | level | mean_p_W | t_prior_s | fwd_actual_s | dur_ratio |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k, v in sorted(res.items()):
            f.write(f"| {k} | {v['n']} | {v['base_mape']:.4f} | {v['full_mape']:.4f} | {v['ci95']} | {v['fallbacks']} | {v.get('level', 0)} | {v.get('mean_p', 0.0):.4f} | {v.get('t_prior', 0.0):.4f} | {v.get('fwd_actual', 0.0):.4f} | {v.get('dur_ratio', 0.0):.4f} |\n")
def write_hill_report(res, out):
    # Extended hill-climb record: per-round valid curves + chosen level + staleness,
    # test numbers, fallbacks, dur_ratio, duration-normalized diagnostics, mbv3 ablation.
    # Schema differs from report_trace.md; written to report_hill.md (choice documented in Task B report).
    import json as _js
    with open(out, "w") as f:
        f.write("# Hill-climb report (validation-only LOTO, test held-out)\n\n")
        f.write("Inner validation: leave-one-train-family-out averaged valid MAPE (energy, no bootstrap; fast). Final CIs n=1000 forward-only. No test-peek (inner valid ⊂ train).\n\n")
        f.write("| family | n | baseMAPE | fullMAPE | base_oracle | full_durnorm | ci95 | fallbacks | level | valid_curve | stale | valid_mape | per_phase_levels | perphase_valid | adopted_perphase | full_lv0 | mean_p_W | t_prior_s | fwd_actual_s | dur_ratio |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k, v in sorted(res.items()):
            vc = _js.dumps([{ "lv": int(d.get("level", 0)), "mape": round(float(d.get("mape", 0.0)), 4) if d.get("mape", None)==d.get("mape", None) else None} for d in v.get("valid_curve", [])])
            ppl = _js.dumps(v.get("per_phase_levels", {}))
            f.write(f"| {k} | {v['n']} | {v['base_mape']:.4f} | {v['full_mape']:.4f} | {v.get('base_mape_oracle', float('nan')):.4f} | {v.get('full_mape_durnorm', float('nan')):.4f} | {v['ci95']} | {v['fallbacks']} | {v.get('level', 0)} | {vc} | {v.get('valid_stale', 0)} | {v.get('valid_mape', float('nan')):.4f} | {ppl} | {v.get('perphase_valid', float('nan')):.4f} | {v.get('adopted_perphase', False)} | {v.get('full_mape_lv0', float('nan')):.4f} | {v.get('mean_p', 0.0):.4f} | {v.get('t_prior', 0.0):.4f} | {v.get('fwd_actual', 0.0):.4f} | {v.get('dur_ratio', 0.0):.4f} |\n")
        f.write("\nNotes: lv5 drift-slope is no-op (columns() lv5==lv4; attn_flops always 0); lv2 freq*temp near-constant so lv2/lv3 often tie lv1 and trigger 2-stale stop before lv4 fam_idx. mbv3 ablation: lv0-only and per-phase (stale-stop) do not fix LOSE; unconstrained per-phase lv4 wins validation but loses test (fam_idx overfit) so NOT adopted. LM: dur_ratio≈0.15 inflates base; report full with AND without durnorm; do not claim proof from relative wins alone.\n")
if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--bootstrap-n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = lomo(a.runs, a.bootstrap_n, a.seed); write_report(res, a.out)
    try:
        import os as _os
        _hill_out = _os.path.join(_os.path.dirname(a.out) or ".", "report_hill.md")
        write_hill_report(res, _hill_out)
    except Exception:
        pass
    vision = [v for k, v in res.items() if k in VISION]
    wins = sum(1 for v in vision if v["base_mape"] - v["full_mape"] > 0.03)
    print(f"vision wins {wins}/4 (LM report-only <=15%)")
    raise SystemExit(1 if (a.strict and wins < 4) else 0)
