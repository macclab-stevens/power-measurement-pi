import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import data
import derive
import qc
import settings
from conftest import make_cell
from test_derive import T0, phase_rows

BOUNDS = {"q1_ctx_frac_flag": 0.01, "q2_max_residual_ms": 1.0, "q3_slip_frac_flag": 0.01,
          "q5_pv_rel_flag": 0.02, "q6_sample_ratio_lo": 0.98, "q6_sample_ratio_hi": 1.02}
HZ = 1000.0


def write_trace(cell, n=3000, hz=HZ, scale=1.0, phases=("idle", "forward"), t=None):
    """Trace at `hz` starting at T0: V = 5, I = 1, P = scale * V * I; phase labels cycle in blocks."""
    t = T0 + (np.arange(n) * (1e9 / hz)).astype(np.int64) if t is None else np.asarray(t, np.int64)
    n = len(t)
    ph = np.array(phases, dtype=object)[np.arange(n) * len(phases) // n]
    tbl = pa.table({"t_pi_ns": t, "I_A": np.ones(n), "V_V": np.full(n, 5.0),
                    "P_W": np.full(n, 5.0 * scale), "phase": pa.array(ph.tolist()).dictionary_encode()})
    pq.write_table(tbl, cell / "power_trace.parquet")


def mk(tmp_path, cid="c1", source="v2", meta=None, alignment=None, rows=None, trace=True,
       trace_kw=None, files=None):
    m = {"modes": ["forward"], "achieved_sample_hz": HZ, "throttled_start": 0, "throttled_end": 0,
         "max_temp_C": 60.0, **(meta or {})}
    al = {"slips": [], "slip_rate_hz": HZ, "slip_samples_total": 0, "max_residual_ms": 0.1,
          **(alignment or {})}
    cell = make_cell(tmp_path, source, cid, meta=m, phase_rows=rows or phase_rows(source),
                     alignment=al, files=files)
    if trace:
        write_trace(cell, **(trace_kw or {}))
    return cell


def run(tmp_path, cid="c1"):
    cat = data.catalogue(tmp_path)
    row = cat[cat["cell_id"] == cid].iloc[0]
    d = pd.DataFrame(derive.derive_cell(row["path"]))
    return qc.qc_cell(row, d, BOUNDS)


def test_clean_cell_triggers_nothing(tmp_path):
    mk(tmp_path, files={"context.csv": "t_ms,temp_C,arm_MHz,throttle_bits\n0.1,45.5,2400,0\n5000,46,2400,0\n",
                        "sysfs.csv": "t_real_ns,temp_C,fan_state,uv_alarm\n1,45,4,0\n2,45,4,0\n"})
    r = run(tmp_path)
    assert list(r) == qc.COLUMNS
    assert r["rules"] == "" and r["include"] is True and r["primary"] is True
    assert r["n_samples"] == 3000 and r["sample_ratio"] == pytest.approx(3000 / (2.999 * HZ))
    assert r["monotonic"] is True and r["dup_timestamps"] == 0 and r["unknown_phases"] == ""
    assert r["pv_median_rel"] == 0.0 and r["start_temp_C"] == 45.5 and r["max_temp_C"] == 60.0
    assert r["n_slips"] == 0 and r["slip_scored_frac"] == 0.0 and r["ctx_read_frac_scored"] == 0.0


def test_v1_cell_without_optional_files(tmp_path):
    mk(tmp_path, source="v1")
    r = run(tmp_path)
    assert r["rules"] == "" and np.isnan(r["start_temp_C"])


def ctx_rows(frame0, measured):
    rows = phase_rows()
    for x in rows:
        if x["phase"] == "forward":
            x["ctx_read_frac"] = frame0 if x["iter"] == "0" else measured
    return rows


def test_q1_uses_measured_rows_only_duration_weighted(tmp_path):
    mk(tmp_path, rows=ctx_rows(0.9, 0.0))         # frame 0 is cold: must not count
    assert run(tmp_path)["rules"] == ""
    mk(tmp_path, "c2", rows=ctx_rows(0.9, 0.05))
    r = run(tmp_path, "c2")
    assert r["rules"] == "Q1flag" and r["ctx_read_frac_scored"] == pytest.approx(0.05)
    assert r["include"] and r["primary"]
    rows = ctx_rows(0.9, 0.0)                     # weighted: 0.5 over 2 ms of 68.5 ms is ~0.0146
    for x in rows:
        if x["iter"] == "1" and x["step"] == "preprocess":
            x["ctx_read_frac"] = 0.5
    mk(tmp_path, "c3", rows=rows)
    r = run(tmp_path, "c3")
    assert r["ctx_read_frac_scored"] == pytest.approx(0.5 * 2.0 / (23 + 25.5 + 20))


def test_q1_threshold_is_strict(tmp_path):
    mk(tmp_path, rows=ctx_rows(0.0, 0.01))
    assert "Q1flag" not in run(tmp_path)["rules"]


def test_q2_excludes_at_threshold(tmp_path):
    mk(tmp_path, "a", alignment={"max_residual_ms": 0.999})
    mk(tmp_path, "b", alignment={"max_residual_ms": 1.0})
    assert run(tmp_path, "a")["rules"] == ""
    r = run(tmp_path, "b")
    assert r["rules"] == "Q2" and r["include"] is False and r["primary"] is False


def test_q3_slip_fraction_is_duration_weighted(tmp_path):
    slip = {"t_ns": T0 + 30_000_000_000, "step_ms": 12.0, "samples": 12}     # inside frame 2
    mk(tmp_path, alignment={"slips": [slip], "slip_samples_total": 12})
    r = run(tmp_path)
    assert r["rules"] == "Q3flag" and r["include"] and r["primary"]
    assert r["slip_scored_frac"] == pytest.approx(25.5 / 68.5)
    assert (r["n_slips"], r["slip_samples"], r["slips_in_scored"], r["slip_frames_excluded"]) == (1, 12, 1, 1)


def test_q3_slip_outside_frames_counts_nothing(tmp_path):
    slip = {"t_ns": T0 + 5_000_000_000, "step_ms": 12.0, "samples": 12}       # during load_weights
    mk(tmp_path, alignment={"slips": [slip], "slip_samples_total": 12})
    r = run(tmp_path)
    assert r["rules"] == "" and r["slip_scored_frac"] == 0.0 and r["slips_in_scored"] == 0
    assert r["n_slips"] == 1


@pytest.mark.parametrize("start,end,ctx,sysfs,flag", [
    (0x50000, 0x50000, "0x50000", 0, False),       # history bits only
    (0x50005, 0, "0", 0, True),                    # current under-voltage + history at start
    (0, "0x50005", "0", 0, True),                  # hex string in run_meta
    (0, 0, "0x50004", 0, True),                    # in-run context.csv bits
    (0, 0, "0x50000", 1, True),                    # sysfs uv_alarm
    (0, 0, "0", 0, False),
])
def test_q4_current_bits(tmp_path, start, end, ctx, sysfs, flag):
    files = {"context.csv": f"t_ms,temp_C,arm_MHz,throttle_bits\n0.1,45.5,2400,0\n5000,46,2400,{ctx}\n",
             "sysfs.csv": f"t_real_ns,temp_C,fan_state,uv_alarm\n1,45,4,0\n2,45,4,{sysfs}\n"}
    mk(tmp_path, meta={"throttled_start": start, "throttled_end": end}, files=files)
    r = run(tmp_path)
    assert ("Q4" in r["rules"]) is flag
    assert r["include"] is True and r["primary"] is (not flag)


def test_q4_reports_where_it_came_from(tmp_path):
    mk(tmp_path, meta={"throttled_end": "0x5"})
    r = run(tmp_path)
    assert (r["throttle_start"], r["throttle_end"], r["throttle_current"]) == (False, True, False)


def test_q5_pv_mismatch_flagged(tmp_path):
    mk(tmp_path, "a", trace_kw={"scale": 1.03})
    mk(tmp_path, "b", trace_kw={"scale": 1.01})
    r = run(tmp_path, "a")
    assert r["rules"] == "Q5flag" and r["pv_median_rel"] == pytest.approx(0.03 / 1.03)
    assert r["include"] and r["primary"]
    assert run(tmp_path, "b")["rules"] == ""


def test_q5_uses_every_50th_row(tmp_path):
    cell = mk(tmp_path, trace=False)
    n = 5000
    t = T0 + (np.arange(n) * 1e6).astype(np.int64)
    P = np.full(n, 5.0)
    P[1::50] = 50.0                                   # only rows the stride skips are bad
    pq.write_table(pa.table({"t_pi_ns": t, "I_A": np.ones(n), "V_V": np.full(n, 5.0), "P_W": P,
                             "phase": ["forward"] * n}), cell / "power_trace.parquet")
    assert run(tmp_path)["pv_median_rel"] == 0.0


def test_q6_non_monotonic_and_duplicates(tmp_path):
    t = T0 + (np.arange(3000) * 1e6).astype(np.int64)
    t2 = t.copy()
    t2[1500] = t2[1499]
    mk(tmp_path, "dup", trace_kw={"t": t2})
    t3 = t.copy()
    t3[[1500, 1501]] = t3[[1501, 1500]]
    mk(tmp_path, "swap", trace_kw={"t": t3})
    r = run(tmp_path, "dup")
    assert r["rules"] == "Q6" and r["dup_timestamps"] == 1 and r["monotonic"] is False
    assert r["include"] is False and r["primary"] is False
    r = run(tmp_path, "swap")
    assert r["rules"] == "Q6" and r["monotonic"] is False and r["dup_timestamps"] == 0


def test_q6_sample_ratio_bounds(tmp_path):
    # 2999 ms span at 1000 Hz: n / (span * hz) = n / 2999; samples missing -> ratio low
    mk(tmp_path, "lo", trace_kw={"t": np.r_[T0 + np.arange(0, 3000) * 1_000_000][::2], })
    mk(tmp_path, "hi", meta={"achieved_sample_hz": HZ / 1.05})
    mk(tmp_path, "ok", meta={"achieved_sample_hz": HZ / 1.015})
    assert run(tmp_path, "lo")["rules"] == "Q6"
    r = run(tmp_path, "hi")
    assert r["rules"] == "Q6" and r["sample_ratio"] > 1.02
    assert run(tmp_path, "ok")["rules"] == ""


def test_q6_ratio_falls_back_to_alignment_rate(tmp_path):
    mk(tmp_path, meta={"achieved_sample_hz": "__drop__"})
    r = run(tmp_path)
    assert r["rules"] == "" and r["expected_samples"] == pytest.approx(2.999 * HZ)


def test_q6_unknown_phase_but_known_prefixes_pass(tmp_path):
    mk(tmp_path, "ok", trace_kw={"phases": ("", "first:forward", "bench:x", "sync:start", "concurrent_drain")})
    assert run(tmp_path, "ok")["rules"] == ""
    mk(tmp_path, "bad", trace_kw={"phases": ("idle", "mystery", "first")})
    r = run(tmp_path, "bad")
    assert r["rules"] == "Q6" and r["unknown_phases"] == "first;mystery" and r["include"] is False


def test_missing_trace_is_quarantined(tmp_path):
    mk(tmp_path, trace=False)
    r = run(tmp_path)
    assert r["rules"] == "Q6" and r["n_samples"] == 0 and r["include"] is False


def test_rules_are_sorted_and_combined(tmp_path):
    mk(tmp_path, meta={"throttled_start": 1}, alignment={"max_residual_ms": 2.0}, trace_kw={"scale": 1.1})
    r = run(tmp_path)
    assert r["rules"] == "Q2;Q4;Q5flag" and r["include"] is False and r["primary"] is False


def test_vocabulary_helper():
    for lab in ("", "idle", "first:forward", "bench:foo", "gap", "train"):
        assert qc.is_known_phase(lab)
    for lab in ("first", "bench", "decode2", "Idle"):
        assert not qc.is_known_phase(lab)
    assert qc.parse_bits("0x50005") == 0x50005 and qc.parse_bits(5.0) == 5 and qc.parse_bits(None) == 0
    assert qc.parse_bits(float("nan")) == 0 and qc.parse_bits("") == 0


def test_run_qc_over_catalogue(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    mk(tmp_path, "a")
    mk(tmp_path, "b", source="v1", alignment={"max_residual_ms": 5.0})
    cat = data.catalogue(tmp_path)
    der = derive.derive_all(cat)
    df = qc.run_qc(cat, der, BOUNDS)
    assert list(df.columns) == qc.COLUMNS and len(df) == 2
    assert dict(zip(df["cell_id"], df["rules"])) == {"a": "", "b": "Q2"}
    assert df.attrs["phase_labels"] == {"idle": 2, "forward": 2}
    assert df.attrs["phase_distinct"] == {"idle": 1, "forward": 1}


def test_run_qc_collapses_hashed_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    mk(tmp_path, "a", trace_kw={"phases": ("idle", "bench:aa", "bench:bb")})
    mk(tmp_path, "b", trace_kw={"phases": ("idle", "bench:cc", "first:dd")})
    cat = data.catalogue(tmp_path)
    df = qc.run_qc(cat, derive.derive_all(cat), BOUNDS)
    assert df.attrs["phase_labels"] == {"idle": 2, "bench:*": 2, "first:*": 1}
    assert df.attrs["phase_distinct"] == {"idle": 1, "bench:*": 3, "first:*": 1}
    md = qc.report_md(df, cat, BOUNDS)
    assert "bench:*" in md and "## Excluded cells" in md and "none." in md


def test_config_has_protocol_bounds():
    # protocol amendment: Q1 flag bound raised from 0.01 to 0.03; all other bounds unchanged
    assert settings.cfg()["qc"] == {**BOUNDS, "q1_ctx_frac_flag": 0.03}


def test_real_cells(real_data):
    names = ["yolo11n_soak30_640_t4_2400_r00"]
    cat = data.catalogue(real_data)
    v1 = cat[cat["source"] == "v1"].iloc[0]["cell_id"]
    v2 = cat[(cat["source"] == "v2") & ~cat["cell_id"].isin(names) & (cat["soak_minutes"].isna())
             ].iloc[0]["cell_id"]
    sub = cat[cat["cell_id"].isin(names + [v1, v2])]
    assert len(sub) == 3
    der = pd.concat([pd.DataFrame(derive.derive_cell(p)) for p in sub["path"]])
    df = qc.run_qc(sub, der, qc.bounds_from_cfg())
    assert len(df) == 3 and set(df["cell_id"]) == set(sub["cell_id"])
    assert df["n_samples"].gt(0).all()
