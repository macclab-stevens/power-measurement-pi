import json
import math

import numpy as np
import pandas as pd
import pytest

import data
import derive
import runlog
import settings
from conftest import make_cell

T0 = 1_000_000_000


def _row(phase, it, step, E, dur, t0, layout="v2", P=None):
    """One phase_energy row; layout v2 has n_finite, v1 does not and orders columns differently."""
    n = max(int(dur), 1)
    r = {"phase": phase, "iter": it, "step": step, "n_samples": n, "n_finite": n,
         "mean_P_W": E / dur if P is None else P, "ctx_read_frac": 0.0,
         "t_start_s": t0, "t_end_s": t0 + dur / 1000.0, "dur_ms": dur, "E_mJ": E}
    if layout == "v1":
        keep = ["phase", "iter", "step", "n_samples", "mean_P_W", "ctx_read_frac",
                "t_start_s", "t_end_s", "dur_ms", "E_mJ"]
        r = {k: r[k] for k in keep}
    return r


def phase_rows(layout="v2", mode="forward", frames=None):
    """Frame 0 (+ a v2 profiler row), frames 1..3, baselines 4 W and 6 W. Frame k starts at 10*k s."""
    L = layout
    rows = [_row("load_weights", "", "", 500.0, 9000.0, 0.0, L),
            _row(mode, "0", "forward", 300.0, 30.0, 10.0, L)]
    if L == "v2":
        rows.append(_row(mode, "0", "profiler", 50.0, 5.0, 10.1, L))
    rows += [_row("baseline_A", "", "", 40.0, 10.0, 5.0, L, P=4.0),
             _row(mode, "1", "preprocess", 10.0, 2.0, 20.0, L),
             _row(mode, "1", "forward", 100.0, 20.0, 20.01, L),
             _row(mode, "1", "postprocess", 5.0, 1.0, 20.03, L),
             _row(mode, "2", "preprocess", 10.0, 2.0, 30.0, L),
             _row(mode, "2", "forward", 110.0, 22.0, 30.01, L),
             _row(mode, "2", "postprocess", 5.0, 1.0, 30.04, L),
             _row(mode, "2", "", 1.0, 0.5, 30.05, L),
             _row(mode, "3", "forward", 100.0, 20.0, 40.0, L),
             _row("baseline_B", "", "", 60.0, 10.0, 50.0, L, P=6.0)]
    return rows


def _cell(tmp_path, cid="c1", source="v2", meta=None, rows=None, slips=None, **kw):
    m = {"modes": ["forward"], **(meta or {})}
    return make_cell(tmp_path, source, cid, meta=m, phase_rows=rows or phase_rows(source),
                     alignment={"slips": slips or [], "slip_rate_hz": 5000.0}, **kw)


def test_frame_table_sums_and_extents():
    pe = pd.DataFrame(phase_rows())
    pe["iter"] = pe["iter"].astype(object)
    t = derive.frame_table(pe, "forward")
    assert list(t.index) == [1, 2, 3]
    assert t.loc[2, "E_mJ"] == pytest.approx(126.0)      # includes the empty-step row
    assert t.loc[2, "dur_ms"] == pytest.approx(25.5)
    assert t.loc[2, "t_start_s"] == 30.0
    assert t.loc[2, "t_end_s"] == pytest.approx(30.05 + 0.0005)
    assert derive.frame_table(pe, "nope").empty


def test_frame_table_ignores_non_numeric_iter():
    pe = pd.DataFrame(phase_rows())
    pe.loc[pe["phase"] == "forward", "iter"] = pe.loc[pe["phase"] == "forward", "iter"].astype(str)
    pe.loc[len(pe)] = _row("forward", "", "", 999.0, 99.0, 60.0)       # empty iter
    pe.loc[len(pe)] = _row("forward", "abc", "forward", 999.0, 99.0, 61.0)
    assert list(derive.frame_table(pe, "forward").index) == [1, 2, 3]


def test_derive_cell_hand_computation(tmp_path):
    (r,) = derive.derive_cell(_cell(tmp_path))
    assert r["mode"] == "forward" and r["cell_id"] == "c1" and r["batch"] == 1
    assert r["n_frames"] == 3 and r["n_frames_slip_excluded"] == 0 and r["n_images"] == 3
    e, d = (115 + 126 + 100) / 3, (23 + 25.5 + 20) / 3
    assert r["energy_mJ_per_image_raw"] == pytest.approx(e)
    assert r["latency_ms_per_image_raw"] == pytest.approx(d)
    assert r["energy_mJ_per_image"] == pytest.approx(e) and r["latency_ms_per_image"] == pytest.approx(d)
    assert r["mean_P_W"] == pytest.approx((115 + 126 + 100) / (23 + 25.5 + 20))
    assert r["baseline_A_W"] == 4.0 and r["baseline_B_W"] == 6.0 and r["baseline_W"] == 5.0
    assert r["dyn_energy_mJ_per_image"] == pytest.approx(e - 5.0 * d)
    # the trace totals cover every row, including load_weights and baselines
    allE = sum(x["E_mJ"] for x in phase_rows())
    allD = sum(x["dur_ms"] for x in phase_rows())
    assert r["trace_E_J"] == pytest.approx(allE / 1000) and r["trace_dur_s"] == pytest.approx(allD / 1000)
    assert math.isnan(r["serve_E_mJ"]) and r["serve_n_requests"] == 0
    assert r["slips_total"] == 0 and r["slips_in_scored"] == 0


def test_batch_halves_per_image(tmp_path):
    (r,) = derive.derive_cell(_cell(tmp_path, meta={"batch": 2}))
    assert r["batch"] == 2 and r["n_images"] == 6
    assert r["energy_mJ_per_image_raw"] == pytest.approx((115 + 126 + 100) / 3 / 2)
    assert r["latency_ms_per_image_raw"] == pytest.approx((23 + 25.5 + 20) / 3 / 2)
    assert r["dyn_energy_mJ_per_image"] == pytest.approx(((115 + 126 + 100) / 3 - 5 * 68.5 / 3) / 2)


def test_v1_layout_matches_v2(tmp_path):
    (a,) = derive.derive_cell(_cell(tmp_path, "a", "v2"))
    (b,) = derive.derive_cell(_cell(tmp_path, "b", "v1"))
    # v1 has no profiler row in frame 0, which is excluded anyway; trace totals differ by that row only
    for k in ("n_frames", "energy_mJ_per_image", "latency_ms_per_image", "mean_P_W",
              "dyn_energy_mJ_per_image", "baseline_W", "energy_mJ_per_image_raw"):
        assert a[k] == pytest.approx(b[k]), k
    assert a["trace_E_J"] - b["trace_E_J"] == pytest.approx(0.05)


def test_slip_in_frame_drops_only_that_frame(tmp_path):
    slips = [{"t_ns": T0 + int(40.005e9), "step_ms": 2.0, "samples": 10}]
    (r,) = derive.derive_cell(_cell(tmp_path, slips=slips))
    assert r["n_frames"] == 3 and r["n_frames_slip_excluded"] == 1
    assert r["slips_total"] == 1 and r["slips_in_scored"] == 1
    assert r["energy_mJ_per_image"] == pytest.approx((115 + 126) / 2)
    assert r["latency_ms_per_image"] == pytest.approx((23 + 25.5) / 2)
    assert r["mean_P_W"] == pytest.approx((115 + 126) / (23 + 25.5))
    assert r["dyn_energy_mJ_per_image"] == pytest.approx((115 + 126) / 2 - 5.0 * (23 + 25.5) / 2)
    # raw still reproduces derived.json (all frames)
    assert r["energy_mJ_per_image_raw"] == pytest.approx((115 + 126 + 100) / 3)
    assert r["n_images"] == 2


def test_slip_in_load_weights_drops_nothing(tmp_path):
    slips = [{"t_ns": T0 + int(5e9), "step_ms": 3.0, "samples": 15}]
    (r,) = derive.derive_cell(_cell(tmp_path, slips=slips))
    assert r["n_frames_slip_excluded"] == 0 and r["slips_total"] == 1 and r["slips_in_scored"] == 0
    assert r["energy_mJ_per_image"] == pytest.approx(r["energy_mJ_per_image_raw"])


def test_slip_step_extends_into_next_frame(tmp_path):
    # starts 1 ms before frame 3 begins and lasts 2 ms: overlaps it
    slips = [{"t_ns": T0 + int(39.999e9), "step_ms": 2.0, "samples": 10}]
    (r,) = derive.derive_cell(_cell(tmp_path, slips=slips))
    assert r["n_frames_slip_excluded"] == 1


def test_slip_times_s():
    al = {"slips": [{"t_ns": T0 + 2_500_000_000, "step_ms": 1.5, "samples": 7}]}
    assert derive.slip_times_s(al, {"t0_real_ns": T0}) == [(2.5, 1.5)]
    assert derive.slip_times_s(al, {}) == []
    assert derive.slip_times_s({}, {"t0_real_ns": T0}) == []
    assert derive.slip_times_s(None, {"t0_real_ns": T0}) == []


def test_slip_overlaps_mask():
    fr = pd.DataFrame({"t_start_s": [0.0, 10.0, 20.0], "t_end_s": [1.0, 11.0, 21.0]})
    m = derive.slip_overlaps(fr, [(10.5, 5.0), (-4.0, 1.0), (0.999, 0.0)])
    assert m.tolist() == [True, True, False] and m.dtype == bool
    assert derive.slip_overlaps(fr, []).tolist() == [False] * 3
    assert derive.slip_overlaps(fr.iloc[:0], [(1, 1)]).shape == (0,)


def test_lm_modes_two_rows(tmp_path):
    rows = (phase_rows(mode="prefill") +
            [r for r in phase_rows(mode="decode") if r["phase"] == "decode"])
    rs = derive.derive_cell(_cell(tmp_path, meta={"task": "lm", "modes": ["prefill", "decode"]},
                                  rows=rows))
    assert [r["mode"] for r in rs] == ["prefill", "decode"]
    assert all(r["n_frames"] == 3 for r in rs)
    assert rs[0]["trace_E_J"] == rs[1]["trace_E_J"]


def test_serve_fields(tmp_path):
    rows = phase_rows()
    t = 60.0
    # stray single-sample row labelled with the last measured iter, then 3 requests
    rows.append(_row("serve", "59", "", 1.0, 0.2, t))
    for i in range(3):
        rows += [_row("serve", str(i), "forward", 100.0, 20.0, t + 1 + i),
                 _row("serve", str(i), "idle_between_requests", 40.0, 10.0, t + 1.5 + i)]
    (r,) = derive.derive_cell(_cell(tmp_path, rows=rows))
    assert r["serve_n_requests"] == 3
    assert r["serve_E_mJ"] == pytest.approx(1 + 3 * 140) and r["serve_dur_ms"] == pytest.approx(0.2 + 3 * 30)
    assert r["serve_idle_E_mJ"] == pytest.approx(120) and r["serve_idle_dur_ms"] == pytest.approx(30)


def test_no_baseline_is_nan(tmp_path):
    rows = [x for x in phase_rows() if not x["phase"].startswith("baseline")]
    (r,) = derive.derive_cell(_cell(tmp_path, rows=rows))
    assert math.isnan(r["baseline_W"]) and math.isnan(r["dyn_energy_mJ_per_image"])
    assert r["energy_mJ_per_image"] == pytest.approx((115 + 126 + 100) / 3)


def test_single_baseline_phase(tmp_path):
    rows = [x for x in phase_rows() if x["phase"] != "baseline_B"]
    (r,) = derive.derive_cell(_cell(tmp_path, rows=rows))
    assert r["baseline_W"] == 4.0 and math.isnan(r["baseline_B_W"])


def test_missing_phase_energy_gives_no_rows(tmp_path):
    cell = make_cell(tmp_path, "v2", "nofile", meta={"modes": ["forward"]})
    assert derive.derive_cell(cell) == []


def test_derive_all_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    _cell(tmp_path, "a")
    _cell(tmp_path, "b", meta={"batch": 2})
    cat = data.catalogue()
    df = derive.derive_all(cat)
    assert sorted(df["cell_id"]) == ["a", "b"]
    cache = settings.derived_dir() / "derive.parquet"
    assert cache.is_file()
    mtime = cache.stat().st_mtime_ns
    df2 = derive.derive_all(cat)
    assert cache.stat().st_mtime_ns == mtime            # served from the cache
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))
    derive.derive_all(cat, force=True)
    assert cache.stat().st_mtime_ns != mtime
    # a different cell set invalidates the cache
    df3 = derive.derive_all(cat[cat["cell_id"] == "a"])
    assert list(df3["cell_id"]) == ["a"]


def test_derive_never_writes_outside_derived(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    _cell(tmp_path, "a")
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    derive.derive_all(data.catalogue())
    new = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert all(p.parent == tmp_path / "derived" for p in new)


def test_cli_derive(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    mfr = {"energy_per_frame_mJ": 113.0, "latency_ms_per_frame": 22.0, "energy_per_image_mJ": 113.0}
    _cell(tmp_path, "a", derived=mfr)
    assert cli.main(["derive"]) == 0
    out = capsys.readouterr().out
    assert "1 rows" in out and "worst" in out.lower()


def test_real_data_matches_derived_json(real_data):
    cat = data.catalogue()
    cat = cat[(cat["source"] == "v2")]
    worst, worst_cell, n = 0.0, None, 0
    for _, c in cat.iterrows():
        dj = data.read_json_opt(c["path"], "derived.json")
        if dj is None:
            continue
        rows = derive.derive_cell(c["path"])
        if len(rows) != 1:
            continue
        r = rows[0]
        for got, want in ((r["energy_mJ_per_image_raw"], dj["energy_per_image_mJ"]),
                          (r["latency_ms_per_image_raw"] * r["batch"], dj["latency_ms_per_frame"])):
            rel = abs(got - want) / abs(want)
            if rel > worst:
                worst, worst_cell = rel, c["cell_id"]
            assert rel < 1e-3, (c["cell_id"], got, want)
        n += 1
    print(f"\nworst relative difference raw vs derived.json over {n} v2 cells: "
          f"{worst:.3e} ({worst_cell})")
    assert n > 100
