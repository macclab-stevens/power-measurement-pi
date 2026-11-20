# host/tests/test_verify_cell.py
import json

import numpy as np
import pandas as pd

from verify_cell import TRAIN_STEPS, derived_metrics, phase_energy, verify

PHASES = ["load_weights", "first:train", "warmup", "sync:start", "baseline_A", "train",
          "baseline_B", "sync:end"]


def _run(tmp_path, meta_over=None, drop_phase=None, drop_step=None, p_spike=None,
         ali_over=None, df_over=None, files=None):
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
    if df_over:
        df = df_over(df)
    df.to_parquet(d / "power_trace.parquet")
    for name, content in (files or {}).items():
        (d / name).write_text(content)
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


# ---- v2 checks -------------------------------------------------------------------------

V2 = {"dataset_version": 2, "governor": "pinned", "fan_mode": "full", "fan_full": True,
      "perf_counters": {"enabled": True, "available": True}, "max_temp_C": 61.0,
      "thermal_abort": None, "model": "yolo11n.pt", "variant": "yolo11n", "imgsz": 640,
      "map_reference": {"map50_95": 0.39, "measured_on": {"machine": "aarch64"}}}


def _v2(tmp_path, meta_over=None, **kw):
    return _run(tmp_path, meta_over={**V2, **(meta_over or {})}, **kw)


def _cols(**cols):
    def f(df):
        for k, v in cols.items():
            df[k] = v(df) if callable(v) else v
        return df
    return f


def _ctr_cov(frac):
    """ctr_instructions non-null for the first `frac` of every phase's samples."""
    def f(df):
        pos = df.groupby("phase").cumcount() / df.groupby("phase")["P_W"].transform("size")
        return np.where(pos < frac, 1.0, np.nan)
    return f


def test_v2_good_cell_passes_with_no_warnings(tmp_path):
    r = verify(_v2(tmp_path, df_over=_cols(ctr_instructions=_ctr_cov(1.0))), 2400)
    assert r["ok"], r["errors"]
    assert r["warnings"] == [] and r["counter_coverage"] == 1.0 and r["max_temp_C"] == 61.0


def test_fan_modes(tmp_path):
    for i, mode in enumerate(("auto", "off")):
        r = verify(_run(tmp_path / str(i), meta_over={"fan_mode": mode, "fan_full": False}))
        assert r["ok"], r["errors"]
    r = verify(_run(tmp_path / "x", meta_over={"fan_mode": "full", "fan_full": False}))
    assert any("fan_full not set" in e for e in r["errors"])
    r = verify(_run(tmp_path / "y", meta_over={"fan_mode": "none", "fan_full": False}))
    assert any("fan_full not set" in e for e in r["errors"])
    r = verify(_run(tmp_path / "z", meta_over={"fan_full": False}))          # v1 meta
    assert any("fan_full not set" in e for e in r["errors"])


def _dvfs(khz):
    return _cols(**{f"sys_cpu{i}_kHz": khz for i in range(4)})


def test_dvfs_frequency_range(tmp_path):
    meta = {"governor": "ondemand"}
    ok = verify(_run(tmp_path / "a", meta_over=meta, df_over=_dvfs(1500000.0)), 2400)
    assert ok["ok"], ok["errors"]
    assert verify(_run(tmp_path / "b", meta_over=meta, df_over=_dvfs(2425000.0)))["ok"]
    for name, khz in (("c", 2426000.0), ("d", 1474000.0)):
        r = verify(_run(tmp_path / name, meta_over=meta, df_over=_dvfs(khz)))
        assert any("cpu clock outside" in e for e in r["errors"])
    r = verify(_run(tmp_path / "e", meta_over=meta))
    assert any("no cpu frequency readings" in e for e in r["errors"])
    r = verify(_run(tmp_path / "f", meta_over=meta, df_over=_dvfs(np.nan)))
    assert any("no cpu frequency readings" in e for e in r["errors"])
    r = verify(_run(tmp_path / "g", meta_over={**meta, "cpu_mhz_pinned": 1800},
                    df_over=_dvfs(2000000.0)), 2400)           # sync clock equality is kept
    assert any("cpu_mhz_pinned" in e for e in r["errors"])


def test_frequency_not_checked_for_pinned_governor(tmp_path):
    assert verify(_run(tmp_path, df_over=_dvfs(100000.0)))["ok"]


def test_counter_coverage(tmp_path):
    r = verify(_v2(tmp_path / "a", df_over=_cols(ctr_instructions=_ctr_cov(0.96))))
    assert r["ok"], r["errors"]
    assert 0.95 <= r["counter_coverage"] < 1.0
    r = verify(_v2(tmp_path / "b", df_over=_cols(ctr_instructions=_ctr_cov(0.90))))
    assert not r["ok"] and any(e.startswith("counter coverage 0.9") and "< 0.95" in e
                               for e in r["errors"])
    r = verify(_v2(tmp_path / "c"))                          # no ctr column at all
    assert r["counter_coverage"] == 0.0 and any("counter coverage" in e for e in r["errors"])
    r = verify(_v2(tmp_path / "d", meta_over={"perf_counters": {"enabled": False}},
                   df_over=_cols(ctr_instructions=1.0)))
    assert any("perf counters not enabled" in e for e in r["errors"])


def test_coverage_counts_only_mode_phases(tmp_path):
    def f(df):                       # counters only in the train phase
        df["ctr_instructions"] = np.where(df["phase"] == "train", 1.0, np.nan)
        return df
    assert verify(_v2(tmp_path, df_over=f))["ok"]


def test_v1_cell_has_no_counter_check(tmp_path):
    r = verify(_run(tmp_path))
    assert r["ok"] and "counter_coverage" not in r and "max_temp_C" not in r
    assert r["warnings"] == []


def test_thermal(tmp_path):
    r = verify(_v2(tmp_path / "a", df_over=_cols(ctr_instructions=1.0),
                   files={"thermal_abort.json": json.dumps({"temp_C": 86.5, "level": "abort"})}))
    assert not r["ok"] and "thermal abort at 86.5 C" in r["errors"]
    r = verify(_v2(tmp_path / "b", df_over=_cols(ctr_instructions=1.0),
                   meta_over={"thermal_abort": {"temp_C": 85.2, "level": "abort"}}))
    assert "thermal abort at 85.2 C" in r["errors"]
    r = verify(_v2(tmp_path / "c", df_over=_cols(ctr_instructions=1.0),
                   meta_over={"max_temp_C": 85.0}))
    assert any("max_temp_C" in e for e in r["errors"])
    r = verify(_v2(tmp_path / "d", df_over=_cols(ctr_instructions=1.0),
                   meta_over={"max_temp_C": 84.9}))
    assert r["ok"] and r["max_temp_C"] == 84.9


def test_aborted_run_without_meta_still_reports_the_abort(tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "thermal_abort.json").write_text(json.dumps({"temp_C": 91.0, "level": "hard_stop"}))
    r = verify(d)
    assert not r["ok"] and "thermal abort at 91.0 C" in r["errors"]


def test_map_reference_warnings(tmp_path):
    ctr = _cols(ctr_instructions=1.0)
    r = verify(_v2(tmp_path / "a", meta_over={"map_reference": None}, df_over=ctr))
    assert r["ok"] and "mAP reference missing for yolo11n@640" in r["warnings"]
    r = verify(_v2(tmp_path / "b", df_over=ctr, meta_over={
        "map_reference": {"map50_95": 0.39, "measured_on": {"machine": "x86_64"}}}))
    assert r["ok"] and any("mAP reference not measured on the Pi" in w for w in r["warnings"])
    r = verify(_v2(tmp_path / "c", df_over=ctr, meta_over={"map_reference": {"map50_95": 0.39}}))
    assert any("not measured on the Pi" in w for w in r["warnings"])
    r = verify(_v2(tmp_path / "d", df_over=ctr, meta_over={"model": "gpt2", "map_reference": None}))
    assert not any("mAP" in w for w in r["warnings"])
    r = verify(_run(tmp_path / "e", meta_over={"model": "yolo11n.pt"}))        # v1 meta: no variant
    assert not any("mAP" in w for w in r["warnings"])


def test_bench_not_required_when_forced_off(tmp_path):
    base = {"task": "image", "bench_skipped": False}
    assert any("bench" in e for e in verify(_run(tmp_path / "a", meta_over=base))["errors"])
    r = verify(_run(tmp_path / "b", meta_over={**base, "bench_forced_off": "quant"}))
    assert not any("bench" in e for e in r["errors"])


def test_derived_metrics_and_file(tmp_path):
    d = _v2(tmp_path, meta_over={"batch": 8, "holdout": "primary-s"},
            df_over=_cols(ctr_instructions=1.0))
    r = verify(d)
    assert r["ok"], r["errors"]
    # frame 1 only: 6 steps x 50 samples x 0.2 ms = 60 ms at 7 W; baseline 5 W
    x = r["derived"]
    assert abs(x["energy_per_frame_mJ"] - 420.0) < 1e-6
    assert abs(x["dyn_energy_per_frame_mJ"] - 120.0) < 1e-6
    assert abs(x["latency_ms_per_frame"] - 60.0) < 1e-6
    assert abs(x["energy_per_image_mJ"] - 52.5) < 1e-6
    assert x["n_frames"] == 1 and x["batch"] == 8 and x["baseline_P_W"] == 5.0
    assert x["variant"] == "yolo11n" and x["holdout"] == "primary-s"
    assert x["map_reference"]["map50_95"] == 0.39
    assert json.loads((d / "derived.json").read_text()) == x


def test_derived_not_written_without_write_and_v1_gets_it(tmp_path):
    d = _run(tmp_path)
    r = verify(d, write=False)
    assert not (d / "derived.json").exists() and r["derived"]["energy_per_frame_mJ"] is not None
    verify(d)
    assert (d / "derived.json").exists()


def test_derived_multi_mode_has_per_mode_values_only(tmp_path):
    d = _run(tmp_path, meta_over={"modes": ["train"]})
    meta = json.loads((d / "run_meta.json").read_text())
    df = pd.read_parquet(d / "power_trace.parquet")
    pe = phase_energy(df, 0.0002)
    two = derived_metrics(pe, {**meta, "modes": ["train", "decode"]}, 5.0)
    assert two["energy_per_frame_mJ"] is None and two["energy_per_image_mJ"] is None
    assert two["by_mode"]["train"]["n_frames"] == 1 and two["by_mode"]["decode"]["n_frames"] == 0


def test_no_measured_iterations_warns(tmp_path):
    def only_frame0(df):
        return df[~((df["phase"] == "train") & (df["iter"] == "1"))].reset_index(drop=True)
    r = verify(_run(tmp_path, df_over=only_frame0))
    assert r["derived"]["energy_per_frame_mJ"] is None
    assert any("no measured iterations" in w for w in r["warnings"])
