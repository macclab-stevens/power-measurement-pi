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

def test_lm_phase_targets_sane(tmp_path):
    # Task C: synthetic sub-bin LM phases. 6-sample decode (29ms span, like
    # runs/v3_lm128 decode n=6) must give sane E>0; 1-sample phase pins the
    # explicit keep+guard policy (E==0/dur==0, invalid target: train drops
    # E<=0 in evaluate._build_train_items, eval forward_mape raises on E==0).
    s = tmp_path / 'samples.csv'
    lines = ['t_ms,phase,window_id,I_A,V_V,P_W']
    lines.append('0,baseline_A,0,1,0.8,0.8')
    lines.append('100,baseline_A,0,1,0.8,0.8')
    base = 1000
    for i in range(6):
        lines.append(f'{base + i * 5},decode,1,2,2.0,4.5')
    lines.append(f'{base + 6 * 5 + 50},lonely,2,1,0.8,1.0')
    s.write_text('\n'.join(lines) + '\n')
    (tmp_path / 'run_meta.json').write_text('{"schema_version":3}')
    (tmp_path / 'context.csv').write_text('t_ms,temp_C,arm_MHz,throttle_bits\n0,50.0,2400,327680\n')
    from targets import build_phase_targets, is_valid_phase_target
    rows = build_phase_targets(str(tmp_path))
    dec = [r for r in rows if r['phase'] == 'decode'][0]
    assert dec['n'] == 6 and dec['dur_s'] > 0 and dec['E_mJ'] > 0
    assert abs(dec['E_mJ'] - dec['P_mean'] * 25.0) < 1e-6
    assert abs(dec['P_mean'] - 4.5) < 1e-9
    assert is_valid_phase_target(dec) is True
    one = [r for r in rows if r['phase'] == 'lonely'][0]
    assert one['n'] == 1 and one['dur_s'] == 0.0 and one['E_mJ'] == 0.0
    assert is_valid_phase_target(one) is False
