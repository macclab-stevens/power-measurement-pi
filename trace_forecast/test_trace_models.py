# trace_forecast/test_models.py
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
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
