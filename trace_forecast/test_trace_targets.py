# trace_forecast/test_targets.py
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
def test_phase_target_energy(tmp_path):
    s = tmp_path/"samples.csv"
    s.write_text("t_ms,phase,window_id,I_A,V_V,P_W\n0,baseline_A,0,1,0.8,0.8\n100,baseline_A,0,1,0.8,0.8\n200,forward,1,2,2.0,4.0\n300,forward,1,2,2.0,4.0\n")
    (tmp_path/"run_meta.json").write_text('{"schema_version":3}')
    (tmp_path/"context.csv").write_text("t_ms,temp_C,arm_MHz,throttle_bits\n50,50.0,2400,327680\n250,55.0,2400,327680\n")
    from targets import build_phase_targets, leakage_inputs
    rows = build_phase_targets(str(tmp_path))
    fwd = [r for r in rows if r["phase"]=="forward"][0]
    # E_mJ = P_W * t_ms: P_mean 4.0W x dur 100ms = 400 mJ
    assert abs(fwd["E_mJ"] - 400.0) < 1e-6 and fwd["n"] == 2 and abs(fwd["P_mean"] - 4.0) < 1e-9
    P, T = leakage_inputs(str(tmp_path))
    assert len(P) == 2 and T == [50.0, 50.0]  # baseline_A/B rows only, never gap/bench
