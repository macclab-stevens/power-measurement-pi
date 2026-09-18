import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
def test_scaling_recovers_quadratic():
    import numpy as np
    from train_scaling import fit_class
    rng = np.random.default_rng(0)
    T = rng.integers(4, 128, size=60)
    logE = 1.0 + 2.0*np.log(T) + rng.normal(0, 0.05, size=60)  # synthetic O(T^2)
    # NOTE: brief's logB=log(T*64*4)=logT+const is rank-2 collinear, lstsq
    # cannot split b_T/c_B (sum recovers 2.0 but b_T alone fails). Add
    # independent bytes jitter so b_T is identifiable; all brief numbers kept.
    logB = np.log(T*64*4) + rng.normal(0, 0.3, size=60)
    m = fit_class(np.log(T), logB, logE)
    assert abs(m["b_T"] - 2.0) < 0.2, m
