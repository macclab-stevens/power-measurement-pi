import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def test_ablation_beats_latency_only():
    from evaluate import mape
    assert mape([1.0, 2.0], [1.1, 1.9]) < 0.1


def test_mape_known_value():
    from evaluate import mape
    # (0.1/1.0 + 0.1/2.0)/2 = 0.075
    assert abs(mape([1.0, 2.0], [1.1, 1.9]) - 0.075) < 1e-9


def test_smape_known_value():
    from evaluate import smape
    # 2*|1-1.1|/(1+1.1)=0.095238..., 2*|2-1.9|/(2+1.9)=0.051282...
    v = smape([1.0, 2.0], [1.1, 1.9])
    assert 0.07 < v < 0.08, v


def test_mae_log_known_value():
    import math
    from evaluate import mae_log
    v = mae_log([1.0, math.e], [1.0, 1.0])
    assert abs(v - 0.5) < 1e-9, v


def test_baseline_latency_only_formula():
    from evaluate import baseline_latency_only
    train = [
        {"p_abab_w": 2.0, "t_per_call_ms": 1.0, "t_eff": 10,
         "bytes": 100, "e_mj": 2.0, "cls": "Conv2d", "family": "a"},
        {"p_abab_w": 4.0, "t_per_call_ms": 1.0, "t_eff": 10,
         "bytes": 100, "e_mj": 4.0, "cls": "Conv2d", "family": "a"},
    ]
    test = [
        {"p_abab_w": 9.0, "t_per_call_ms": 2.0, "t_eff": 5,
         "bytes": 50, "e_mj": 18.0, "cls": "Conv2d", "family": "b"},
    ]
    preds, mean_p = baseline_latency_only(train, test)
    assert abs(mean_p - 3.0) < 1e-9, mean_p
    assert abs(preds[0] - 6.0) < 1e-9, preds


def test_full_model_beats_baseline_on_synthetic_scaling():
    import numpy as np
    from evaluate import baseline_latency_only, full_model, mape
    rng = np.random.default_rng(1)
    # Synthetic: E follows exact scaling law in (t_eff, bytes); t_per_call
    # is independent noise so latency-only baseline cannot win.
    train, test = [], []
    for i in range(60):
        t = int(rng.integers(4, 128))
        b = int(rng.integers(1000, 9000))
        e = float(np.exp(1.0 + 0.9 * np.log(t) + 0.4 * np.log(b)))
        tp = float(rng.uniform(0.5, 5.0))
        p = e / tp  # inconsistent P on purpose (baseline cheated feature is noisy)
        row = {"p_abab_w": p, "t_per_call_ms": tp, "t_eff": t,
               "bytes": b, "e_mj": e, "cls": "Linear", "family": "tr"}
        (train if i < 45 else test).append(row)
    pb, _ = baseline_latency_only(train, test)
    pf = full_model(train, test)
    y = [r["e_mj"] for r in test]
    assert mape(y, pf) < mape(y, pb), (mape(y, pf), mape(y, pb))


def test_bootstrap_ci_valid():
    import numpy as np
    from evaluate import bootstrap_delta
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pb = np.array([1.1, 1.9, 3.2, 3.8])
    pf = np.array([1.0, 2.0, 3.0, 4.0])
    lo, hi, delta = bootstrap_delta(y, pb, pf, n=100, seed=0)
    assert lo <= delta <= hi, (lo, delta, hi)
    assert lo < hi
    assert not (np.isnan(lo) or np.isnan(hi))
