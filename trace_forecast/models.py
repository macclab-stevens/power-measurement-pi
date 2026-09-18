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
