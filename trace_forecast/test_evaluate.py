import sys, os; sys.path.insert(0, os.path.dirname(__file__))
def test_forward_weighted_metric():
    from evaluate import forward_mape
    rows = [{"phase": "forward", "E": 10.0, "E_pred": 10.5}, {"phase": "gap", "E": 1.0, "E_pred": 5.0}]
    assert forward_mape(rows) < 0.1  # gap error must not dominate
def test_throttle_abort():
    from evaluate import check_throttle
    assert check_throttle(0x0) is False and check_throttle(0x1) is True  # R6 synthetic: bit0 = currently throttled
