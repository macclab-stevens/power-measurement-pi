# host/tests/test_verify_cell.py
import json

import numpy as np
import pandas as pd

from verify_cell import TRAIN_STEPS, verify

PHASES = ["load_weights", "first:train", "warmup", "sync:start", "baseline_A", "train",
          "baseline_B", "sync:end"]


def _run(tmp_path, meta_over=None, drop_phase=None, drop_step=None, p_spike=None,
         ali_over=None):
    d = tmp_path / "cell"
    d.mkdir(parents=True)
    meta = {"schema_version": 4, "task": "train", "modes": ["train"], "warmup_passes": 5,
            "serve_requests": 0, "bench_skipped": True, "cpu_mhz_pinned": 2400,
            "fan_full": True}
    meta.update(meta_over or {})
    (d / "run_meta.json").write_text(json.dumps(meta))
    ali = {"pass": True, "slip_rate_hz": 5000.0, "max_residual_ms": 0.2,
           "slip_warning": False, "drift_warning": False}
    ali.update(ali_over or {})
    (d / "alignment.json").write_text(json.dumps(ali))
    rows = []
    for ph in PHASES:
        if ph == drop_phase:
            continue
        if ph == "train":
            for it in range(2):
                for st in TRAIN_STEPS:
                    if st == drop_step:
                        continue
                    rows += [(ph, str(it), st, 7.0)] * 50
        else:
            rows += [(ph, "", "", 5.0)] * 100
    df = pd.DataFrame(rows, columns=["phase", "iter", "step", "P_W"])
    df["P_W"] = df["P_W"].astype(np.float32)
    if p_spike:
        df.loc[: len(df) // 50, "P_W"] = p_spike
    df["t_s"] = np.arange(len(df)) / 5000.0
    df["ctx_read"] = False
    df.to_parquet(d / "power_trace.parquet")
    return d


def test_good_cell_passes_and_writes_phase_energy(tmp_path):
    d = _run(tmp_path)
    r = verify(d, 2400)
    assert r["ok"], r["errors"]
    assert r["mean_P_W"] == {"train": 7.0} and r["energy_closure"] < 1e-6
    pe = pd.read_csv(d / "phase_energy.csv", keep_default_na=False)
    row = pe[(pe["phase"] == "train") & (pe["iter"].astype(str) == "0")
             & (pe["step"] == "backward")].iloc[0]
    assert row["n_samples"] == 50 and abs(row["E_mJ"] - 7.0 * 10.0) < 1e-6


def test_missing_phase_step_or_wrong_clock_fail(tmp_path):
    assert any("load_weights" in e for e in verify(_run(tmp_path / "a", drop_phase="load_weights"))["errors"])
    assert any("backward" in e for e in verify(_run(tmp_path / "b", drop_step="backward"))["errors"])
    r = verify(_run(tmp_path / "c"), 1500)
    assert not r["ok"] and any("cpu_mhz_pinned" in e for e in r["errors"])


def test_power_range_and_alignment_fail(tmp_path):
    assert not verify(_run(tmp_path / "a", p_spike=40.0))["ok"]
    r = verify(_run(tmp_path / "b", ali_over={"pass": False, "fail_reasons": ["x"]}))
    assert any("alignment" in e for e in r["errors"])


def test_slip_is_a_warning(tmp_path):
    r = verify(_run(tmp_path, ali_over={"slip_warning": True, "slip_samples_total": 40}))
    assert r["ok"] and any("slip" in w for w in r["warnings"])


def test_inference_cell_expects_serve_and_bench(tmp_path):
    d = _run(tmp_path, meta_over={"task": "image", "serve_requests": 10, "bench_skipped": False})
    errs = verify(d)["errors"]
    assert any("serve" in e for e in errs) and any("bench" in e for e in errs)


def _trace(d, p):
    df = pd.read_parquet(d / "power_trace.parquet")
    df["P_W"] = p(df["P_W"].to_numpy().copy())
    df.to_parquet(d / "power_trace.parquet")


def test_energy_counts_only_observed_samples(tmp_path):
    d = _run(tmp_path)

    def nan_some(P):
        P[::200] = np.nan                       # 0.5 %: below the 1 % NaN limit
        return P
    _trace(d, nan_some)
    r = verify(d)
    assert r["ok"], r["errors"]
    assert r["energy_closure"] < 1e-6
    pe = pd.read_csv(d / "phase_energy.csv", keep_default_na=False)
    P = pd.read_parquet(d / "power_trace.parquet")["P_W"].to_numpy(dtype=np.float64)
    assert abs(pe["E_mJ"].sum() - np.nansum(P) * 0.2) < 1e-3     # 0.2 ms per sample


def test_all_nan_trace_is_an_error_not_an_exception(tmp_path):
    d = _run(tmp_path)
    _trace(d, lambda P: np.full_like(P, np.nan))
    r = verify(d)
    assert not r["ok"] and any("finite" in e or "NaN" in e for e in r["errors"])


def test_infinite_power_fails(tmp_path):
    d = _run(tmp_path)

    def one_inf(P):
        P[5] = np.inf
        return P
    _trace(d, one_inf)
    r = verify(d)
    assert not r["ok"] and any("infinite" in e for e in r["errors"])
