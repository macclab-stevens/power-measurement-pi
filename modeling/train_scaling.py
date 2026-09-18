"""Per-class log scaling-law trainer: logE = a + b_T*log(T_eff) + c_B*log(bytes).

Consumes runs/v3_*/perop_power.csv (schema_version==3 only, skip _TAINTED_V2).
Energy is ABAB-corrected: E_ABAB = P_delta_ABAB_W * t_per_call_ms (mJ).
CPU numpy only, lstsq only, no sklearn.
"""
import csv
import glob
import json
import os

import numpy as np


def fit_class(logT, logB, logE):
    X = np.stack([np.ones_like(logT), logT, logB], axis=1)
    coef, *_ = np.linalg.lstsq(X, logE, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((logE - pred) ** 2)))
    return {"a": float(coef[0]), "b_T": float(coef[1]), "c_B": float(coef[2]), "rmse": rmse}


def predict(m, t_eff, bytes_mv):
    import math
    c = m.get("c_B", m.get("c_bytes"))
    return math.exp(m["a"] + m["b_T"] * math.log(max(1, t_eff)) + c * math.log(max(1, bytes_mv)))


def _e_abab_mj(r):
    try:
        pa = float(r.get("P_delta_ABAB_W") or "")
        tp = float(r.get("t_per_call_ms") or "")
        e = pa * tp
        if e > 0:
            return e
    except (TypeError, ValueError):
        pass
    try:
        e = float(r.get("E_per_call_mJ") or "")
        return e if e > 0 else None
    except (TypeError, ValueError):
        return None


def load_rows(pattern="runs/v3_*/perop_power.csv"):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pats = [pattern, os.path.join(base, pattern)]
    seen = []
    for p in pats:
        seen += [f for f in glob.glob(p) if f not in seen]
    groups = {}
    for f in sorted(seen):
        d = os.path.dirname(f)
        if os.path.exists(os.path.join(d, "_TAINTED_V2")):
            continue
        mp = os.path.join(d, "run_meta.json")
        try:
            if json.load(open(mp)).get("schema_version") != 3:
                continue
        except (OSError, ValueError):
            continue
        for r in csv.DictReader(open(f)):
            if str(r.get("SKIPPED")).strip() == "True":
                continue
            try:
                t = int(float(r.get("t_eff") or 0))
                b = int(float(r.get("bytes_moved") or 0))
            except (TypeError, ValueError):
                continue
            e = _e_abab_mj(r)
            if t <= 0 or b <= 0 or not e or e <= 0:
                continue
            groups.setdefault(r.get("class") or "?", []).append((t, b, e))
    return groups


def main(out="modeling/scaling.json"):
    groups = load_rows()
    all_t = [t for v in groups.values() for t, _, _ in v]
    all_b = [b for v in groups.values() for _, b, _ in v]
    all_e = [e for v in groups.values() for _, _, e in v]
    pm = fit_class(np.log(all_t), np.log(all_b), np.log(all_e))
    out_d = {}
    for cls, v in sorted(groups.items()):
        lt = np.log([t for t, _, _ in v])
        lb = np.log([b for _, b, _ in v])
        le = np.log([e for _, _, e in v])
        if len(v) >= 15:
            m = fit_class(lt, lb, le)
            m.update({"c_bytes": m["c_B"], "n": len(v)})
        else:
            coef = np.array([pm["a"], pm["b_T"], pm["c_B"]])
            proj = np.stack([np.ones_like(lt), lt, lb], axis=1) @ coef
            rmse = float(np.sqrt(np.mean((le - proj) ** 2)))
            m = {"a": pm["a"], "b_T": pm["b_T"], "c_B": pm["c_B"]}
            m.update({"c_bytes": pm["c_B"], "rmse": rmse})
            m.update({"n": len(v), "fallback": "pooled"})
        out_d[cls] = m
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op = os.path.join(base, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    json.dump(out_d, open(op, "w"), indent=2)
    print("wrote %s classes=%d pooled_rmse=%.4f" % (op, len(out_d), pm["rmse"]))
    return out_d


if __name__ == "__main__":
    main()
