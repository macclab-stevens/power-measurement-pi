import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
def _load_sibling_evaluate():
    path = os.path.join(os.path.dirname(__file__), "evaluate.py")
    spec = importlib.util.spec_from_file_location("trace_forecast_evaluate_sibling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def test_forward_weighted_metric():
    forward_mape = _load_sibling_evaluate().forward_mape
    rows = [{"phase": "forward", "E": 10.0, "E_pred": 10.5}, {"phase": "gap", "E": 1.0, "E_pred": 5.0}]
    assert forward_mape(rows) < 0.1  # gap error must not dominate
def test_throttle_abort():
    check_throttle = _load_sibling_evaluate().check_throttle
    assert check_throttle(0x0) is False and check_throttle(0x1) is True  # R6 synthetic: bit0 = currently throttled
