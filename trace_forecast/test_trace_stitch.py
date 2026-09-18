# trace_forecast/test_stitch.py
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
def test_stitch_cumsum():
    from stitch import stitch
    plan = [{"phase": "forward", "frames": 10}, {"phase": "gap", "frames": 0, "dur_s": 0.5}]
    tr = stitch(plan, 0.1, lambda t: 1.0, lambda ph: (2.0, 0.1))
    assert abs(tr[-1]["E_cum"] - (3.0*1.0 + 3.0*0.5)) < 1e-6 and tr[0]["P_lo"] < tr[0]["P_mean"] < tr[0]["P_hi"]
    assert tr[0]["E_lo"] < tr[0]["E_cum"] < tr[0]["E_hi"]  # duration term strictly widens E bands
