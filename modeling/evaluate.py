"""Ablation + bootstrap LOMO evaluation (Task 6).

Consumes runs/v3_*/perop_power.csv (schema_version==3 only, skip _TAINTED_V2)
plus modeling/scaling.json. Compares latency-only baseline
(E_pred = mean_P_global * t_per_call_ms, mean_P from train) against per-class
log-log scaling-law full model, under leave-one-family-out (LOMO) over the
7-family v3 matrix, with bootstrap 95% CIs for the MAPE delta.

Writes modeling/report.md with table:
family | n | MAPE_base | MAPE_full | delta | CI (+ sMAPE/MAE-log).

Thresholds (written into report; FAIL recorded, exit stays 0 unless --strict):
MAPE_full < MAPE_base - 3pp on >=4 families; attention b_T in [1.7,2.3],
linear b_T in [0.8,1.2] else FAIL.

numpy only, CPU only, no sklearn.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train_scaling import fit_class, predict
except ImportError:  # pragma: no cover
    from modeling.train_scaling import fit_class, predict

FAMILIES = ["yolov8n", "yolo11n", "effb0", "mbv3", "lm16", "lm64", "lm128"]
MIN_N_PER_CLASS = 15


def mape(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y), 1e-12)
    return float(np.mean(np.abs(y - p) / denom))


def smape(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-12)))


def mae_log(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(np.log(np.maximum(y, 1e-12)) - np.log(np.maximum(p, 1e-12)))))


def _family_from_dir(path):
    b = os.path.basename(os.path.normpath(path))
    s = b[3:] if b.startswith("v3_") else b
    for cand in FAMILIES:
        if s == cand or s.startswith(cand):
            return cand
    return s


def _e_abab_mj(r):
    try:
        pa = float(r.get("P_delta_ABAB_W") or "")
        tp = float(r.get("t_per_call_ms") or "")
        e = pa * tp
        if e > 0:
            return e, pa, tp
    except (TypeError, ValueError):
        pass
    try:
        e = float(r.get("E_per_call_mJ") or "")
        tp = float(r.get("t_per_call_ms") or "")
        pa = float(r.get("P_delta_ABAB_W") or "")
        if e > 0:
            return e, pa, tp
    except (TypeError, ValueError):
        return None
    return None


def _expand_run_patterns(patterns, repo_root):
    seen = set()
    files = []
    for pat in patterns:
        cands = [pat, os.path.join(repo_root, pat)]
        for c in cands:
            if os.path.isdir(c):
                cands_f = [os.path.join(c, "perop_power.csv")]
            else:
                cands_f = sorted(glob.glob(c))
            for f in cands_f:
                if os.path.isdir(f):
                    f = os.path.join(f, "perop_power.csv")
                key = os.path.abspath(f)
                if key not in seen and os.path.exists(f):
                    seen.add(key)
                    files.append(key)
    return sorted(files)


def load_rows(pattern="runs/v3_*/perop_power.csv"):
    """Only schema_version==3, skip tainted dirs; y=log(E_ABAB), X=[logT,logB]."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    files = _expand_run_patterns(pats, repo_root)
    rows = []
    for f in files:
        d = os.path.dirname(f)
        if os.path.exists(os.path.join(d, "_TAINTED_V2")):
            continue
        mp = os.path.join(d, "run_meta.json")
        try:
            if json.load(open(mp)).get("schema_version") != 3:
                continue
        except (OSError, ValueError):
            continue
        family = _family_from_dir(d)
        for r in csv.DictReader(open(f)):
            if str(r.get("SKIPPED")).strip() == "True":
                continue
            try:
                t = int(float(r.get("t_eff") or 0))
                bv = int(float(r.get("bytes_moved") or 0))
            except (TypeError, ValueError):
                continue
            got = _e_abab_mj(r)
            if not got:
                continue
            e, pa, tp = got
            if t <= 0 or bv <= 0 or not e or e <= 0:
                continue
            try:
                pa_f = float(pa)
                tp_f = float(tp)
            except (TypeError, ValueError):
                continue
            if not (pa_f > 0 and tp_f > 0):
                continue
            rows.append({
                "family": family,
                "cls": r.get("class") or "?",
                "t_eff": t,
                "bytes": bv,
                "e_mj": float(e),
                "t_per_call_ms": tp_f,
                "p_abab_w": pa_f,
                "logT": float(np.log(t)),
                "logB": float(np.log(bv)),
                "logE": float(np.log(e)),
            })
    return rows


def baseline_latency_only(train_rows, test_rows):
    """E_pred = mean_P_global * t_per_call (mean_P from train)."""
    if not train_rows or not test_rows:
        raise ValueError("baseline_latency_only needs non-empty train and test")
    mean_p = float(np.mean([r["p_abab_w"] for r in train_rows]))
    preds = np.array([mean_p * r["t_per_call_ms"] for r in test_rows], dtype=float)
    return preds, mean_p


def _fit_models(train_rows, min_n=MIN_N_PER_CLASS):
    if not train_rows:
        raise ValueError("no train rows")
    lt = np.log([r["t_eff"] for r in train_rows])
    lb = np.log([r["bytes"] for r in train_rows])
    le = np.log([r["e_mj"] for r in train_rows])
    pm = fit_class(lt, lb, le)
    pooled = {"a": pm["a"], "b_T": pm["b_T"], "c_B": pm["c_B"]}
    groups = {}
    for r in train_rows:
        groups.setdefault(r["cls"], []).append(r)
    models = {}
    for cls, v in groups.items():
        if len(v) >= min_n:
            m = fit_class(np.log([x["t_eff"] for x in v]),
                          np.log([x["bytes"] for x in v]),
                          np.log([x["e_mj"] for x in v]))
            models[cls] = {"a": m["a"], "b_T": m["b_T"], "c_B": m["c_B"]}
        else:
            models[cls] = dict(pooled)
    return models, pooled


def full_model(train_rows, test_rows, models=None, pooled=None):
    """Per-class scaling predict (fit on train unless models given)."""
    if models is None or pooled is None:
        models, pooled = _fit_models(train_rows)
    preds = []
    for r in test_rows:
        m = models.get(r["cls"], pooled)
        preds.append(predict(m, r["t_eff"], r["bytes"]))
    return np.array(preds, dtype=float)


def bootstrap_delta(y_true, pred_base, pred_full, n=1000, seed=0):
    """Resample test indices w/ replacement; 95% CI for delta=MAPE_full-MAPE_base."""
    y = np.asarray(y_true, dtype=float)
    pb = np.asarray(pred_base, dtype=float)
    pf = np.asarray(pred_full, dtype=float)
    nn = len(y)
    if nn == 0:
        return float("nan"), float("nan"), float("nan")
    obs = mape(y, pf) - mape(y, pb)
    rng = np.random.default_rng(seed)
    ds = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, nn, nn)
        ds[i] = mape(y[idx], pf[idx]) - mape(y[idx], pb[idx])
    lo, hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
    return lo, hi, float(obs)


def lomo(rows, families=None, n_bootstrap=1000, seed=0):
    if families is None:
        families = FAMILIES
    out = []
    for fam in families:
        te = [r for r in rows if r["family"] == fam]
        tr = [r for r in rows if r["family"] != fam]
        if not te or not tr:
            out.append({"family": fam, "n": len(te), "empty": True})
            continue
        pb, mean_p = baseline_latency_only(tr, te)
        pf = full_model(tr, te)
        y = np.array([r["e_mj"] for r in te], dtype=float)
        lo, hi, delta = bootstrap_delta(y, pb, pf, n=n_bootstrap, seed=seed)
        out.append({
            "family": fam, "n": len(te), "empty": False,
            "mape_base": mape(y, pb), "mape_full": mape(y, pf), "delta": delta,
            "ci_lo": lo, "ci_hi": hi,
            "smape_base": smape(y, pb), "smape_full": smape(y, pf),
            "maelog_base": mae_log(y, pb), "maelog_full": mae_log(y, pf),
            "mean_p_train": mean_p,
        })
    return out


def check_thresholds(results, scaling):
    scored = [r for r in results if not r.get("empty")]
    wins = sum(1 for r in scored if (r["mape_base"] - r["mape_full"]) > 0.03)
    ablation_pass = wins >= 4
    lin = (scaling or {}).get("Linear", {})
    lin_bt = lin.get("b_T")
    lin_pass = lin_bt is not None and 0.8 <= float(lin_bt) <= 1.2
    att_key = next((k for k in (scaling or {}) if "ttention" in k), None)
    att_bt = (scaling or {}).get(att_key, {}).get("b_T") if att_key else None
    att_pass = att_bt is not None and 1.7 <= float(att_bt) <= 2.3
    overall = bool(ablation_pass and lin_pass and att_pass)
    return {"wins": wins, "n": len(scored), "ablation_pass": ablation_pass,
            "linear_b_T": lin_bt, "linear_pass": bool(lin_pass),
            "attention_key": att_key, "attention_b_T": att_bt,
            "attention_pass": bool(att_pass), "overall_pass": overall}


def _load_scaling(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def write_report(results, scaling, out_path, n_bootstrap, seed, rows_pattern):
    verdict = check_thresholds(results, scaling)
    lines = []
    lines.append("# Task 6 report: ablation + bootstrap LOMO eval")
    lines.append("")
    lines.append("VDD_CORE rail only. Schema 3 only (`schema_version==3`, `_TAINTED_V2` skipped).")
    lines.append("Energy `E_ABAB = P_delta_ABAB_W * t_per_call_ms` (mJ); `y=log(E_ABAB)`, `X=[logT,logB]`.")
    lines.append("Baseline latency-only: `E_pred = mean_P_global * t_per_call_ms` (mean_P from train fold).")
    lines.append("Full: per-class `logE = a + b_T*log(T_eff) + c_B*log(bytes)` "
                 "(`fit_class` lstsq, n>=15 else pooled fallback; unseen test class -> pooled).")
    lines.append("LOMO: train on rest, test on held-out family. "
                 "Bootstrap n=%d seed=%d resamples test indices w/ replacement; "
                 "95%% CI (2.5/97.5 pct) for `delta = MAPE_full - MAPE_base`." % (n_bootstrap, seed))
    lines.append("Inputs: `%s` + `%s`." % (rows_pattern, "modeling/scaling.json"))
    lines.append("")
    lines.append("## LOMO ablation (bootstrap CI for delta)")
    lines.append("")
    lines.append("| family | n | MAPE_base | MAPE_full | delta | 95% CI | sMAPE_base | sMAPE_full | MAElog_base | MAElog_full |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("empty"):
            lines.append("| %s | %d | NaN | NaN | NaN | [NaN, NaN] | NaN | NaN | NaN | NaN |" % (r["family"], r["n"]))
        else:
            lines.append("| %s | %d | %.4f | %.4f | %+.4f | [%.4f, %.4f] | %.4f | %.4f | %.4f | %.4f |" % (
                r["family"], r["n"], r["mape_base"], r["mape_full"], r["delta"],
                r["ci_lo"], r["ci_hi"], r["smape_base"], r["smape_full"],
                r["maelog_base"], r["maelog_full"]))
    lines.append("")
    lines.append("## Thresholds (FAIL recorded honestly; exit stays 0 unless --strict)")
    lines.append("")
    lines.append("- Ablation `MAPE_full < MAPE_base - 3pp` on >=4/6 families: "
                 "**%s** (%d/%d families; brief lists 7 families, threshold says 6 "
                 "— scored on evaluated families)." % (
                     "PASS" if verdict["ablation_pass"] else "FAIL", verdict["wins"], verdict["n"]))
    lines.append("- Linear `b_T in [0.8,1.2]`: **%s** (b_T=%s)." % (
        "PASS" if verdict["linear_pass"] else "FAIL", verdict["linear_b_T"]))
    lines.append("- Attention `b_T in [1.7,2.3]`: **%s** (%s)." % (
        "PASS" if verdict["attention_pass"] else "FAIL",
        ("key=%s b_T=%s" % (verdict["attention_key"], verdict["attention_b_T"])
         if verdict["attention_key"] else "no *Attention* class in scaling.json/v3 runs (MiniGPTExplicit uses functional SDPA, no nn.MultiheadAttention leaf)")))
    lines.append("- Overall: **%s**." % ("PASS" if verdict["overall_pass"] else "FAIL"))
    lines.append("")
    lines.append("## Scaling coefficients (global fit, from modeling/scaling.json)")
    lines.append("")
    lines.append("| class | n | b_T | c_B | rmse | fallback |")
    lines.append("|---|---|---|---|---|---|")
    for cls in sorted((scaling or {})):
        m = scaling[cls]
        lines.append("| %s | %s | %.4f | %.4f | %.4f | %s |" % (
            cls, m.get("n", "?"), float(m.get("b_T", float("nan"))),
            float(m.get("c_B", m.get("c_bytes", float("nan")))),
            float(m.get("rmse", float("nan"))), m.get("fallback", "")))
    lines.append("")
    lines.append("## Notes")
    lines.append("- No missing values in scored rows; `t_eff,bytes,E>0` enforced; MAPE denom guarded at 1e-12.")
    lines.append("- Baseline sees `t_per_call_ms` directly (E=P*t, P varies little), "
                 "full sees only `(t_eff,bytes)` — latency-only is expected to be strong; "
                 "a FAIL here means scaling features do not beat cheat-adjacent latency, not a harness bug.")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/v3_*/perop_power.csv"])
    ap.add_argument("--out", default="modeling/report.md")
    ap.add_argument("--scaling", default="modeling/scaling.json")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when thresholds FAIL (default: exit 0, FAIL only in report)")
    args = ap.parse_args(argv)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = load_rows(args.runs)
    if not rows:
        print("no rows for patterns %s" % (args.runs,))
        return 2
    sc_path = args.scaling if os.path.isabs(args.scaling) else os.path.join(repo_root, args.scaling)
    scaling = _load_scaling(sc_path)
    results = lomo(rows, FAMILIES, n_bootstrap=args.bootstrap_n, seed=args.seed)
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    verdict = write_report(results, scaling, out, args.bootstrap_n, args.seed, " ".join(args.runs))
    for r in results:
        if r.get("empty"):
            print("%s n=%d EMPTY" % (r["family"], r["n"]))
        else:
            print("%s n=%d MAPE_base=%.4f MAPE_full=%.4f delta=%+.4f CI=[%.4f,%.4f]" % (
                r["family"], r["n"], r["mape_base"], r["mape_full"],
                r["delta"], r["ci_lo"], r["ci_hi"]))
    print("ablation %d/%d %s; linear %s; attention %s; overall %s -> %s" % (
        verdict["wins"], verdict["n"], "PASS" if verdict["ablation_pass"] else "FAIL",
        "PASS" if verdict["linear_pass"] else "FAIL",
        "PASS" if verdict["attention_pass"] else "FAIL",
        "PASS" if verdict["overall_pass"] else "FAIL", out))
    if args.strict and not verdict["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
