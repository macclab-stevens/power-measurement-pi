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
def test_full_hookup_beats_fallback_on_synthetic(tmp_path):
    import csv
    import json
    import math
    import random
    mod = _load_sibling_evaluate()
    rng = random.Random(0)
    a_coef, b_coef, c_coef = 1.0, 0.9, 0.2
    def make_run(parent, name, teff, nbytes):
        d = parent / name
        d.mkdir()
        logE = a_coef + b_coef * math.log(teff) + c_coef * math.log(nbytes)
        E_true = math.exp(logE) * (1.0 + rng.uniform(-0.02, 0.02))
        Pf = E_true / 1000.0
        with open(d / "samples.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "phase", "window_id", "I_A", "V_V", "P_W"])
            w.writerow([0, "baseline_A", 0, 1, 0.8, 0.8])
            w.writerow([100, "baseline_A", 0, 1, 0.8, 0.8])
            w.writerow([200, "forward", 1, 2, 2.0, Pf])
            w.writerow([1200, "forward", 1, 2, 2.0, Pf])
            w.writerow([1300, "baseline_B", 2, 1, 0.8, 0.85])
            w.writerow([1400, "baseline_B", 2, 1, 0.8, 0.85])
        with open(d / "layer_table.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "config_id", "class", "config_json", "input_shapes", "input_dtypes", "out_shapes", "callsite", "mode", "leaf", "params", "macs", "bytes_moved", "arith_intensity", "t_eff", "seq_len", "count", "lat_mean_ms", "lat_median_ms", "lat_std_ms"])
            for i, pth in enumerate(["p0", "p1"]):
                cls = "Conv2d" if i == 0 else "BatchNorm2d"
                cid = "cid%d_%s" % (i, name)
                w.writerow([pth, cid, cls, "{}", "[[1,3,16,16]]", "[torch.float32]", "[[1,16,16,16]]", pth, "forward", "True", 100, 1000, nbytes / 2, 1.0, teff / 2, 256, 1, 1.0, 1.0, 0.1])
        (d / "run_meta.json").write_text(json.dumps({"schema_version": 3, "seq_len": 16, "task": "image"}))
        (d / "context.csv").write_text("t_ms,temp_C,arm_MHz,throttle_bits\n0,55.0,2400,327680\n")
        return str(d)
    dirs = []
    for i, tb in enumerate([(1000, 50000), (2000, 80000), (1500, 65000)]):
        dirs.append(make_run(tmp_path, "v3_yolo11n_fake%d" % i, tb[0], tb[1]))
    for i, tb in enumerate([(1200, 55000), (2200, 85000), (1800, 70000)]):
        dirs.append(make_run(tmp_path, "v3_effb0_fake%d" % i, tb[0], tb[1]))
    res = mod.lomo(dirs, bootstrap_n=100, seed=0)
    assert set(res.keys()) == {"yolo11n", "effb0"}
    for fam, v in res.items():
        assert v["fallbacks"] == 0, "%s fallbacks %s" % (fam, v["fallbacks"])
        assert v["full_mape"] < v["base_mape"], "%s full %s vs base %s" % (fam, v["full_mape"], v["base_mape"])
def test_hillclimb_picks_improving_level_on_synthetic():
    import math
    import random
    mod = _load_sibling_evaluate()
    rng = random.Random(0)
    per_fam = {}
    for fi, fam in enumerate(["famA", "famB", "famC", "famD"]):
        items = []
        for i in range(6):
            teff = 500 + fi * 300 + i * 200
            nbytes = 20000 + fi * 10000 + i * 5000
            logT = math.log(teff)
            logB = math.log(nbytes)
            ai = logT - logB
            y = 1.0 + 0.9 * logT + 0.2 * logB + 2.0 * ai + rng.uniform(-0.01, 0.01)
            feat = {"t_eff": float(teff), "bytes": float(nbytes), "temp0": 55.0, "freq": 2400.0, "attn_flops": 0.0, "fam_idx": float(fi), "class": "forward", "phase": "forward", "unseen_cid": False}
            items.append((feat, y, math.exp(y), "forward"))
        per_fam[fam] = items
    res = mod._choose_level_loto(per_fam, max_rounds=5)
    assert res["level"] >= 1, "hill-climb should pick level>=1 when truth needs AI term, got %s" % (res,)
    curve = res.get("curve", res.get("rounds", []))
    assert res.get("stale", 0) >= 2 or len(curve) < 5, "stale-stop should fire, got %s" % (res,)

