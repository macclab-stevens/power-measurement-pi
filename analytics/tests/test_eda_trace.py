"""Exploration report, trace level: computations on synthetic data."""
import numpy as np
import pandas as pd
import pytest

import eda_trace as et


def test_ols_slope_se_known():
    rng = np.random.default_rng(0)
    x = rng.uniform(40, 70, 4000)
    y = 3 + 0.05 * x + rng.normal(0, 0.3, 4000)
    r = et.ols(x, y)
    assert r["slope"] == pytest.approx(0.05, abs=3 * r["se"]) and r["se"] == pytest.approx(0.3 / (x.std() * 63.2), rel=0.05)
    assert r["resid_sd"] == pytest.approx(0.3, rel=0.05) and 0 < r["r2"] < 1 and r["n"] == 4000
    assert np.isnan(et.ols([1, 1, 1], [1, 2, 3])["slope"])


def test_within_slope_removes_between_cell_offsets():
    rng = np.random.default_rng(1)
    rows = []
    for c in range(6):                                   # cells at different temperatures AND different power levels
        t = 50 + 4 * c + rng.normal(0, 0.5, 300)
        rows.append(pd.DataFrame({"g": c, "x": t, "y": 5 - 0.5 * (4 * c) + 0.04 * (t - 50 - 4 * c) + rng.normal(0, 0.05, 300)}))
    d = pd.concat(rows)
    assert et.ols(d["x"], d["y"])["slope"] < 0           # the between-cell effect flips the pooled sign
    w = et.within_slope(d["x"], d["y"], d["g"])
    assert w["slope"] == pytest.approx(0.04, abs=0.005) and w["k"] == 6 and w["se"] > 0


def test_temp_slopes_per_config_and_low_range_flag():
    rng = np.random.default_rng(2)
    parts = []
    for cid, lo in (("a0", 40), ("a1", 55), ("b0", 60)):
        t = lo + rng.uniform(0, 8 if cid != "b0" else 0.5, 500)
        parts.append(pd.DataFrame({"cell_id": cid, "phase": "forward", "P_W": 6 + 0.1 * t + rng.normal(0, .05, 500), "temp_C": t}))
    parts.append(pd.DataFrame({"cell_id": "a0", "phase": "gap", "P_W": 1.0, "temp_C": 99.0}, index=[0]))
    info = pd.DataFrame({"cell_id": ["a0", "a1", "b0"], "family": ["A", "A", "B"], "config": ["ca", "ca", "cb"]})
    t = et.temp_slopes(pd.concat(parts), info).set_index("config")
    assert t.loc["ca", "n_cells"] == 2 and t.loc["ca", "n_windows"] == 1000          # gap windows excluded
    assert t.loc["ca", "slope_pooled"] == pytest.approx(0.1, abs=0.01) and t.loc["ca", "slope_within"] == pytest.approx(0.1, abs=0.01)
    assert t.loc["cb", "low_range"] and not t.loc["ca", "low_range"]
    s = et.slope_family_summary(t.reset_index()).set_index("family")
    assert s.loc["A", "n_used"] == 1 and s.loc["B", "n_used"] == 0


def test_time_to_95_exponential_within_1_percent():
    tau, a, p_inf = 300.0, 0.3, 10.0
    t = np.arange(0, 3600, 0.1)
    p = p_inf * (1 - a * np.exp(-t / tau)) + np.random.default_rng(3).normal(0, 0.01, len(t))
    r = et.time_to_fraction(t, p)
    analytic = tau * np.log(a / 0.05)                    # 1 - a e^{-t/tau} = 0.95
    assert r["t_frac_s"] == pytest.approx(analytic, rel=0.01)
    assert r["final_mean"] == pytest.approx(p_inf, rel=1e-3)


def test_time_to_fraction_series_starting_above_is_zero_and_short_is_nan():
    t = np.arange(0, 100, 0.1)
    assert et.time_to_fraction(t, np.linspace(10, 9.8, len(t)), 0.95, 5)["t_frac_s"] == 0.0
    assert np.isnan(et.time_to_fraction(t[:2], np.array([1.0, 2.0]))["t_frac_s"])


def test_soak_summary_cold_hot():
    t = np.arange(0, 1800, 0.1)
    s = pd.DataFrame({"t_s": t, "P_W": 9 + 1 * (1 - np.exp(-t / 200)), "temp_C": 40 + 30 * (1 - np.exp(-t / 200))})
    r = et.soak_summary(s)
    k = 60 / 200                                         # first-minute mean of 1 - exp(-t/200) is 1 - (1 - e^-k)/k
    assert r["delta_P_W"] == pytest.approx((1 - 0) - (1 - (1 - np.exp(-k)) / k), abs=0.03) and r["hot_T_C"] > r["cold_T_C"] and r["n_cold"] == 600


def test_demeaned_correlation_removes_pure_between_cell_effect():
    rng = np.random.default_rng(4)
    rows = []
    for c in range(20):                                   # between cells: P and feature rise together; within: independent
        lvl = c * 1.0
        rows.append(pd.DataFrame({"cell_id": f"c{c}", "phase": "forward", "counted": 1,
                                  "P_W": 5 + lvl + rng.normal(0, 0.1, 200), "busy": 1 + lvl + rng.normal(0, 0.1, 200)}))
    w = pd.concat(rows)
    r = et.counter_correlations(w, features=["busy"]).set_index("phase").loc["forward"]
    assert r["spearman_raw"] > 0.95 and r["pearson_raw"] > 0.95
    assert abs(r["spearman_dm"]) < 0.1 and abs(r["pearson_dm"]) < 0.1 and r["n_dm"] == 4000


def test_counter_correlations_ratio_features_use_counted_only_and_pool_phases():
    rng = np.random.default_rng(5)
    n = 300
    d = pd.DataFrame({"cell_id": "c", "phase": ["forward"] * n, "counted": 1, "P_W": rng.normal(8, 1, n)})
    d["ipc"] = d["P_W"] + rng.normal(0, 0.1, n)
    d.loc[:99, "counted"] = 0
    d.loc[:99, "ipc"] = 5.0 + rng.normal(0, 1, 100)       # junk values in uncounted windows must not count
    g = d.assign(phase="gap")
    r = et.counter_correlations(pd.concat([d, g]), features=["ipc"]).set_index("phase")
    assert r.loc["forward", "n_raw"] == 200 and r.loc["forward", "spearman_raw"] > 0.9
    assert r.loc["pooled", "n_raw"] == 400 and r.loc["idle", "n_raw"] == 0


def test_acf_ar1_and_lag():
    rng = np.random.default_rng(6)
    phi, n = 0.9, 40000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    a = et.acf(x, 60)
    assert a[0] == pytest.approx(1.0) and a[5] == pytest.approx(phi ** 5, abs=0.03)
    lag = et.first_lag_below(a, 0.1)
    assert lag == int(np.ceil(np.log(0.1) / np.log(phi))) or abs(lag - np.log(0.1) / np.log(phi)) < 2   # about 22
    assert et.first_lag_below(np.array([1.0, 0.9, 0.8]), 0.1) is None


def test_acf_summary_seconds_and_cell_choice():
    n = 2000
    t0 = 10 ** 18
    w = pd.DataFrame({"cell_id": "x", "phase": "forward", "t_start_ns": t0 + np.arange(n) * 100_000_000,
                      "P_W": np.sin(np.arange(n) * 2 * np.pi / 40)})
    r = et.acf_summary(w, "x")
    assert r["dt_s"] == pytest.approx(0.1) and r["lag_windows"] == 10 and r["lag_s"] == pytest.approx(1.0)   # cos(2 pi k / 40) < 0.1 first at k = 10
    info = pd.DataFrame({"cell_id": ["a_r01", "a_r00", "b_r00", "b2_r00"], "family": ["A", "A", "B", "B"],
                         "base_model": ["yolo11n", "yolo11n", "yolo11s", "yolo11n"], "repeat": [1, 0, 0, 0]})
    assert et.acf_cells(info, {"a_r01", "a_r00", "b_r00", "b2_r00"}) == ["a_r00", "b2_r00"]
    assert et.acf_cells(info, {"b_r00"}) == ["b_r00"]


def test_order_residuals_and_spearman():
    rng = np.random.default_rng(7)
    rows, started = [], {}
    for k in range(40):                                   # 40 configs x 3 cells, energy drifts +0.1 % per start slot
        for r in range(3):
            cid = f"c{k}_{r}"
            slot = k * 3 + r
            rows.append(dict(cell_id=cid, config_key=f"k{k}", energy_mJ_per_image=1000 * (1 + 0.001 * slot) * 1.0))
            started[cid] = pd.Timestamp("2026-11-21 12:00:00") + pd.Timedelta(minutes=slot)
    rows.append(dict(cell_id="solo", config_key="solo", energy_mJ_per_image=5.0))
    started["solo"] = pd.Timestamp("2026-11-22 10:00")
    c = et.order_residuals(pd.DataFrame(rows), pd.Series(started))
    assert len(c) == 120 and c["config_key"].nunique() == 40                     # single-cell config dropped
    s = et.order_spearman(c).set_index("day")
    assert s.loc["2026-11-21", "n_cells"] == 120 and "pooled" in s.index
    assert abs(s.loc["2026-11-21", "rho"]) < 0.3                                  # within-config drift is tiny and symmetric
    drift = c.assign(resid=c["order"] * 0.01 + rng.normal(0, 0.01, len(c)))
    assert et.order_spearman(drift).set_index("day").loc["pooled", "rho"] > 0.8


def test_pmic_fits_per_group():
    rng = np.random.default_rng(8)
    n = 400
    pm = rng.uniform(4, 9, n)
    pw = pd.DataFrame({"cell_id": ["a"] * n, "phase": "forward", "pmic_W": pm, "P_W": 1.0 + 1.2 * pm + rng.normal(0, 0.1, n)})
    pw = pd.concat([pw, pw.assign(cell_id="z")])
    g = et.pmic_fits(pw, pd.Series({"a": "v2 A"}))
    assert list(g["group"]) == ["v2 A"] and g.loc[0, "n_windows"] == n          # "z" has no group: dropped
    assert g.loc[0, "b"] == pytest.approx(1.2, abs=0.03) and g.loc[0, "a"] == pytest.approx(1.0, abs=0.2)
    assert g.loc[0, "r2"] > 0.98 and g.loc[0, "resid_sd_W"] == pytest.approx(0.1, rel=0.15)


def test_splice_trace_replaces_marker_between_sections():
    import eda
    md = f"## 5. x\n\n{eda.TRACE_MARKER}\n\n## 9. y\n"
    out = eda.splice_trace(md, "## 6. t\n\n## 7. c\n\n## 8. p\n")
    assert eda.TRACE_MARKER not in out
    assert [h for h in out.splitlines() if h.startswith("## ")] == ["## 5. x", "## 6. t", "## 7. c", "## 8. p", "## 9. y"]


def test_write_eda_trace_on_real_data_has_sections_and_figures(real_data, tmp_path):
    import data
    import deploy
    import derive
    import eda
    import eda_trace_write
    import settings
    qc_path = settings.HERE / "reports" / "qc_cells.csv"
    if not qc_path.is_file() or not (settings.derived_dir() / "windows.parquet").is_file():
        pytest.skip("qc_cells.csv or the windows cache is not present")
    cat = data.catalogue()
    md = eda_trace_write.write_eda_trace(cat, derive.derive_all(cat), pd.read_csv(qc_path), tmp_path)
    heads = [h for h in md.splitlines() if h.startswith("## ")]
    assert heads[0].startswith("## 6.") and heads[1].startswith("## 7.") and heads[2].startswith("## 8.")
    full = eda.splice_trace(f"a\n{eda.TRACE_MARKER}\nb\n", md)
    assert "yolo11n_soak30_640_t4_2400_r00" in full and "n = 1" in full and "dataset v2" in full
    assert {p.name for p in (tmp_path / "fig").glob("*.png")} == {"eda_temp_slopes.png", "eda_soak.png", "eda_acf.png",
                                                                  "eda_pmic_scatter.png"}


def test_md_table_escapes_pipes_so_rows_keep_the_header_width():
    import eda_write as ew
    df = pd.DataFrame({"config": ["yolo11m|yolo11m|fp32|320|1|4|2400|pinned|full", "a|b"], "a|b": [1.5, float("nan")],
                       "txt": ["x\ny", "z"]})
    lines = ew.md_table(df).splitlines()

    def n_cells(line):
        return len(line.strip().strip("|").replace("\\|", "").split("|"))
    assert len({n_cells(ln) for ln in lines}) == 1 and n_cells(lines[0]) == 3
    assert "yolo11m\\|yolo11m\\|fp32" in lines[2] and "\\|" in lines[0] and len(lines) == 4
    assert ew.plural(1, "configuration") == "1 configuration" and ew.plural(2, "configuration") == "2 configurations"


def test_prov_singular_configuration():
    import eda_write as ew
    p = ew.prov(pd.DataFrame({"rig": ["nov2026"], "n_cells": [3]}))
    assert p.startswith("n = 1 configuration / 3 cells")
