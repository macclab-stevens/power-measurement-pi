"""Exploration report, configuration level: computations on synthetic data and the writer's headings."""
import numpy as np
import pandas as pd
import pytest

import data
import derive
import eda
import eda_write
import plotstyle
import static
import tables
from conftest import make_cell


# ------------------------------------------------------------------ ICC

def _icc_frame(rng, k, sd_between, sd_within, sizes):
    rows = []
    for g in range(k):
        mu = 100 + rng.normal(0, sd_between)
        for _ in range(sizes[g % len(sizes)]):
            rows.append({"g": g, "y": mu + rng.normal(0, sd_within)})
    return pd.DataFrame(rows)


def test_icc_known_variance_balanced():
    df = _icc_frame(np.random.default_rng(1), 400, 2.0, 1.0, [3])
    r = eda.icc_oneway(df, "g", "y")
    assert r["icc"] == pytest.approx(4 / 5, abs=0.03)           # 4 / (4 + 1)
    assert r["k"] == 400 and r["n_cells"] == 1200
    assert r["n0"] == pytest.approx(3.0)
    assert r["var_within"] == pytest.approx(1.0, rel=0.1)


def test_icc_unbalanced_and_singletons_dropped():
    df = _icc_frame(np.random.default_rng(2), 600, 1.0, 1.0, [1, 2, 5])
    r = eda.icc_oneway(df, "g", "y")
    assert r["icc"] == pytest.approx(0.5, abs=0.05)
    assert r["k"] == 400                                        # the 1-cell groups are not used
    assert 2 < r["n0"] < 5


def test_icc_degenerate():
    assert np.isnan(eda.icc_oneway(pd.DataFrame({"g": [1, 2], "y": [1.0, 2.0]}), "g", "y")["icc"])


# ------------------------------------------------------------------ scaling fit

def test_fit_power_law_recovers_noiseless():
    f = np.array([1.2, 1.5, 1.8, 2.0, 2.2, 2.4])
    r = eda.fit_power_law(f, 1.5 + 3.0 * f ** 1.8)
    assert r["converged"] and r["max_abs_resid"] < 1e-6
    assert (r["P_static"], r["k"], r["alpha"]) == pytest.approx((1.5, 3.0, 1.8), rel=1e-4)


def test_fit_power_law_three_points_exact_and_bounded():
    f = np.array([1.5, 2.0, 2.4])
    r = eda.fit_power_law(f, np.array([5.98, 8.20, 9.86]))      # concave: free fit has P_static < 0
    assert r["converged"] and r["max_abs_resid"] < 1e-6 and r["P_static"] < 0
    b = eda.fit_power_law(f, np.array([5.98, 8.20, 9.86]), bounded=True)
    assert b["P_static"] >= 0 and b["max_abs_resid"] > 1e-3 and b["at_bound"]


def test_fit_power_law_reports_failure():
    r = eda.fit_power_law([1.5, 2.0, 2.4], [5.0, float("nan"), 9.0])
    assert not r["converged"] and "did not converge" in r["message"]


def test_scaling_series_requires_three_clocks():
    rows = [dict(base_model="yolo11n", variant="yolo11n", precision="fp32", imgsz=640, batch=1, threads=4, rig="nov2026",
                 cpu_mhz=m, mean_P_W=p, n_cells=3) for m, p in ((1500, 6.0), (2000, 8.0), (2400, 9.8))]
    cfgs = pd.DataFrame(rows + [{**rows[0], "base_model": "yolo11s", "variant": "yolo11s"}])
    idle = pd.DataFrame({"rig": ["oct2026"] * 3, "cpu_key": ["1500", "2000", "2400"], "fan_mode": "full",
                         "P_idle_W": [4.0, 4.5, 5.0], "n": [5, 5, 5], "iqr_W": 0.0})
    s = eda.scaling_series(cfgs, idle)
    assert [(x["label"], x["rig"]) for x in s] == [("yolo11n", "nov2026"), ("idle", "oct2026")]
    t = eda.scaling_table(s)
    assert len(t) == 4 and t["converged"].all()


# ------------------------------------------------------------------ shares

def _pe():
    def r(phase, it, step, E, d):
        return {"phase": phase, "iter": it, "step": step, "E_mJ": E, "dur_ms": d}
    return pd.DataFrame([r("load_weights", "", None, 10, 100), r("first:forward", 0, None, 5, 50),
                         r("baseline_A", "", None, 20, 200), r("baseline_B", "", None, 20, 200),
                         r("forward", 0, "forward", 99, 99),
                         r("forward", 1, "preprocess", 10, 10), r("forward", 1, "forward", 80, 80),
                         r("forward", 1, None, 10, 10), r("bench:abc", "", None, 5, 50),
                         r("bench:def", "", None, 5, 50), r("sync:end", "", None, 1, 10), r("idle", "", None, 1, 10)])


def test_phase_group():
    m = ["forward"]
    got = {p: eda.phase_group(p, m) for p in ("forward", "first:forward", "bench:x1", "sync:start", "baseline_B",
                                              "gap", "idle", "warmup")}
    assert got == {"forward": "measured", "first:forward": "first", "bench:x1": "bench", "sync:start": "sync",
                   "baseline_B": "baseline", "gap": "gap", "idle": "other", "warmup": "warmup"}


def test_phase_and_step_shares():
    ph = eda.phase_shares(_pe(), ["forward"])
    assert ph["E_share"].sum() == pytest.approx(1.0) and ph["dur_share"].sum() == pytest.approx(1.0)
    tot = _pe()["E_mJ"].sum()
    assert ph.loc["measured", "E_share"] == pytest.approx((99 + 100) / tot)
    assert ph.loc["bench", "E_share"] == pytest.approx(10 / tot)
    st = eda.step_shares(_pe(), ["forward"])                    # frame 1 only: iter 0 is the cold call
    assert st["E_share"].to_dict() == pytest.approx({"preprocess": 0.1, "forward": 0.8, "(none)": 0.1})
    assert eda.step_shares(_pe(), ["train"]).empty


# ------------------------------------------------------------------ rig shift and interactions

def _cell(cid, rig, family, e, lat=100.0, p=9.0, **kw):
    return {"cell_id": cid, "rig": rig, "family": family, "base_model": "yolo11n", "variant": "yolo11n",
            "precision": "fp32", "imgsz": 640, "batch": 1, "threads": 4, "cpu_mhz": 2400,
            "energy_mJ_per_image": e, "latency_ms_per_image": lat, "mean_P_W": p, **kw}


def test_rig_shift_pairs_ref_with_v1_only():
    cells = pd.DataFrame([_cell("a", "oct2026", "v1", 1000), _cell("b", "oct2026", "v1", 1010),
                          _cell("c", "nov2026", "ref", 1050), _cell("d", "nov2026", "ref", 1060),
                          _cell("e", "nov2026", "A", 5000),                       # other family: not a ref twin
                          _cell("f", "nov2026", "ref", 3000, imgsz=320)])         # no v1 twin
    r = eda.rig_shift(cells)
    assert len(r) == 1 and r.loc[0, "n_v1"] == 2 and r.loc[0, "n_ref"] == 2
    assert r.loc[0, "energy_pct"] == pytest.approx(100 * (1055 / 1005 - 1))
    assert r.loc[0, "latency_pct"] == pytest.approx(0.0) and r.loc[0, "diff_se_pct"] > 0


def _cfg(model="yolo11n", variant=None, precision="fp32", imgsz=640, batch=1, threads=4, mhz=2400, e=1000.0, rig="nov2026"):
    variant = variant or model
    key = f"{model}|{variant}|{precision}|{imgsz}|{batch}|{threads}|{mhz}|{rig}"
    return dict(config_key=key, base_model=model, variant=variant, precision=precision, imgsz=imgsz, batch=batch,
                threads=threads, cpu_mhz=mhz, rig=rig, energy_mJ=e, latency_ms=e / 10, mean_P_W=9.0, n_cells=3,
                cell_ids="x;y;z")


def test_thread_grid_batch_and_precision():
    cfgs = pd.DataFrame([_cfg(threads=4, e=1000), _cfg(threads=1, e=900), _cfg(imgsz=320, threads=4, e=300),
                         _cfg(batch=2, e=980), _cfg(variant="yolo11n_int8s", precision="int8", e=650),
                         _cfg(variant="yolo11n_fp16", precision="fp16", e=1300),
                         _cfg(variant="yolo11n_int8s", precision="int8", mhz=1500, e=500)])   # no fp32 twin at 1500
    g = eda.thread_grid(cfgs)
    assert g.set_index(["imgsz", "threads"]).loc[(640, 1), "pct_vs_4t"] == pytest.approx(-10.0)
    b = eda.batch_effect(cfgs).set_index("batch")
    assert b.loc[2, "pct_vs_batch1"] == pytest.approx(-2.0)
    p = eda.precision_ratios(cfgs).set_index("variant")
    assert p.loc["yolo11n_int8s", "energy_ratio"] == pytest.approx(0.65) and len(p) == 2
    assert p.loc["yolo11n_fp16", "energy_ratio"] == pytest.approx(1.3)


def test_energy_accuracy_pareto_and_filters():
    cfgs = pd.DataFrame([_cfg(e=1000, imgsz=320) | {"cell_ids": "a"}, _cfg(e=2000, imgsz=480) | {"cell_ids": "b"},
                         _cfg(e=3000, imgsz=640) | {"cell_ids": "c"}, _cfg(e=900, imgsz=960, threads=1) | {"cell_ids": "d"},
                         _cfg(e=500, rig="oct2026") | {"cell_ids": "e"}])
    maps = {"a": (0.3, 0.4), "b": (0.4, 0.5), "c": (0.35, 0.45), "d": (0.9, 0.9)}
    ea = eda.energy_accuracy(cfgs, maps)
    assert list(ea["imgsz"]) == [320, 480, 640]                  # threads 1 and oct2026 excluded
    assert list(ea["pareto"]) == [True, True, False]


# ------------------------------------------------------------------ writer

def _rows(e0, frames=3):
    def r(phase, it, step, E, d, P=None):
        return {"phase": phase, "iter": it, "step": step, "n_samples": 10, "n_finite": 10,
                "mean_P_W": E / d if P is None else P, "ctx_read_frac": 0.0, "t_start_s": 0.0, "t_end_s": 1.0,
                "dur_ms": d, "E_mJ": E}
    rows = [r("load_weights", "", "", 500.0, 9000.0), r("baseline_A", "", "", 40.0, 10.0, P=4.0),
            r("baseline_B", "", "", 50.0, 10.0, P=5.0), r("first:forward", "0", "", 30.0, 30.0)]
    for it in range(1, frames + 1):
        rows += [r("forward", str(it), "preprocess", 0.1 * e0, 2.0), r("forward", str(it), "forward", e0, 20.0),
                 r("forward", str(it), "postprocess", 0.05 * e0, 1.0)]
    return rows


def _lm_rows(modes=("prefill", "decode"), e=10.0):
    base = _rows(e)
    out = [{**r, "phase": r["phase"].replace("forward", modes[0])} for r in base]
    for m in modes[1:]:
        out += [{**r, "phase": m} for r in base if r["phase"] == "forward"]
    return out


FEATS = {"macs": 1e9, "params": 2e6, "weight_bytes": 8e6, "act_bytes": 5e7, "depth": 10, "n_leaf": 50, "n_module_calls": 120.0,
         **{f"n_{g}": 1 for g in static.GROUPS}, **{f"frac_macs_{g}": 1 / 6 for g in static.GROUPS}}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "root"))
    root = tmp_path / "root"
    mr = {"map50_95": 0.4, "map50": 0.55}
    for i, e in enumerate((100.0, 101.0, 99.5)):
        make_cell(root, "v1", f"yolo11n_2400_r0{i}", phase_rows=_rows(e))
        make_cell(root, "v2", f"v1ref_yolo11n_2400_r0{i}", phase_rows=_rows(e * 1.01),
                  meta={"map_reference": mr}, index={"family": "ref"})
        make_cell(root, "v2", f"yolo11n_int8s_640_t4_2400_r0{i}", phase_rows=_rows(e * 0.7),
                  meta={"variant": "yolo11n_int8s", "quant": "int8-static", "map_reference": mr},
                  index={"model_key": "yolo11n", "family": "B"})
        make_cell(root, "v1", f"lm64_2400_r0{i}", phase_rows=_lm_rows(e=e / 10),
                  meta={"task": "lm", "modes": ["prefill", "decode"], "seq_len": 64},
                  index={"model_key": "lm64", "task": "lm"})
        make_cell(root, "v1", f"train_lm64_sgd_2400_r0{i}", phase_rows=_lm_rows(("train",), e=e),
                  meta={"task": "train", "modes": ["train"], "batch": 8, "optimizer": "sgd"},
                  index={"model_key": "train_lm64_sgd", "task": "train"})
    cat = data.catalogue(root)
    der = derive.derive_all(cat)
    qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
    st = pd.DataFrame([{"cell_id": "x", **FEATS}], index=pd.MultiIndex.from_tuples([("yolo11n", 640)],
                                                                                 names=["base_model", "imgsz"]))
    return cat, der, qc, tables.mode_a_table(cat, der, qc, st)


def test_write_eda_config_headings_and_figures(synthetic, tmp_path):
    cat, der, qc, cfgs = synthetic
    out = tmp_path / "rep"
    md = eda.write_eda_config(cat, der, qc, cfgs, out)
    assert (out / "eda.md").read_text() == md
    for h in ("## 1. Repeatability", "## 2. Clock scaling and idle power", "## 3. Rig shift", "## 4. Interactions",
              "### 4.1 Threads x image size", "### 4.2 Batch", "### 4.3 INT8 and FP16 against FP32",
              "## 5. Phase and step shares", "## 9. Energy against accuracy", "## 10. MiniGPT and training workloads",
              "### 10.1 MiniGPT energy per token", "### 10.2 Training energy per step", "ICC(1,1)", "Noise floor",
              "exact by construction, descriptive only", eda.TRACE_MARKER):
        assert h in md, h
    assert md.index("## 5.") < md.index(eda.TRACE_MARKER) < md.index("## 9.")
    assert "dataset v1" in md and "dataset v2" in md
    for fig in ("eda_cv_hist", "eda_phase_shares", "eda_energy_vs_map"):
        assert (out / "fig" / f"{fig}.png").stat().st_size > 1000


def test_lm_tokens_and_train_step_units(synthetic):
    cat, der, qc, _ = synthetic
    lm = eda.lm_per_token(cat, der, qc).set_index(["mode", "cpu_mhz"])
    assert lm.loc[("prefill", 2400), "tokens_per_frame"] == 64 and lm.loc[("decode", 2400), "tokens_per_frame"] == 1
    assert lm.loc[("prefill", 2400), "n_cells"] == 3
    tr = eda.train_per_step(cat, der, qc)
    assert tr.loc[0, "workload"] == "lm64" and tr.loc[0, "optimiser"] == "sgd" and tr.loc[0, "batch"] == 8
    assert tr.loc[0, "E_step_mJ"] == pytest.approx(1.15 * 100.5 * 1.0, rel=0.2)    # frame energy, not per sample


def test_write_eda_config_on_empty_tables(tmp_path):
    empty = pd.DataFrame(columns=["cell_id", "rig", "task", "path", "mode"])
    cat = pd.DataFrame({c: pd.Series(dtype=object) for c in data.CATALOGUE_COLUMNS})
    der = pd.DataFrame({c: pd.Series(dtype=object) for c in derive.COLUMNS})
    qc = pd.DataFrame({"cell_id": pd.Series(dtype=object), "primary": pd.Series(dtype=bool)})
    cfgs = pd.DataFrame({c: pd.Series(dtype=object) for c in ["config_key", "base_model", "variant", "precision", "imgsz",
                         "batch", "threads", "cpu_mhz", "rig", "n_cells", "energy_cv", "cell_ids", "energy_mJ",
                         "latency_ms", "mean_P_W"]})
    md = eda.write_eda_config(cat, der, qc, cfgs, tmp_path)
    assert "## 1. Repeatability" in md and "## 10. MiniGPT" in md


def test_splice_trace_and_md_table():
    md = f"a\n{eda.TRACE_MARKER}\nb"
    out = eda.splice_trace(md, "## 6. Thermal\n")
    assert "## 6. Thermal" in out and eda.TRACE_MARKER not in out and out.endswith("b")
    t = eda_write.md_table(pd.DataFrame({"a": [1, 2], "b": [0.123456, float("nan")], "c": [True, False]}))
    assert t.splitlines()[0] == "| a | b | c |" and "| 1 | 0.1235 | yes |" in t and "| 2 |  | no |" in t
    assert eda_write.md_table(pd.DataFrame()) == "_No data._\n"


def test_plotstyle_save(tmp_path):
    fig, ax = plotstyle.subplots()
    ax.plot([0, 1], [0, 1])
    p = plotstyle.save(fig, "x", tmp_path)
    assert p == tmp_path / "x.png" and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert plotstyle.DPI == 150


# ------------------------------------------------------------------ data-driven narrative (Task CU item 7)

def test_fmt_icc_never_rounds_up_to_one():
    f = eda_write.fmt_icc
    assert f(0.99996) == "0.99996" and f(0.999951) == "0.99995" and f(0.5) == "0.50000"
    assert f(0.9999999) == "> 0.99999" and f(1.0) == "> 0.99999"
    assert f(float("nan")) == "n/a" and "1.0000" not in f(0.99996)


def test_icc_sentence_uses_the_computed_span_and_value():
    near = eda_write.icc_sentence(0.99996, 37.4)
    assert "37-fold" in near and "50-fold" not in near and "near 1" in near
    low = eda_write.icc_sentence(0.4, 3.0)
    assert "near 1" not in low and "0.40000" in low and "3-fold" in low
    assert "-fold" not in eda_write.icc_sentence(0.99, float("nan"))
    multi = pd.DataFrame({"energy_mJ": [100.0, 450.0, 1200.0]})
    assert eda_write.energy_span(multi) == pytest.approx(12.0)
    assert np.isnan(eda_write.energy_span(multi.iloc[0:0]))


def _st(rows):
    cols = ["series", "kind", "rig", "fit", "alpha", "converged", "at_bound"]
    return pd.DataFrame(rows, columns=cols)


def test_alpha_bound_sentence_only_when_a_free_fit_is_at_bound():
    none = _st([("yolo11n", "forward power", "nov2026", "free", 0.8, True, False),
                ("idle", "idle power", "nov2026", "free", 0.5, True, False),
                ("yolo11n", "forward power", "nov2026", "P_static in [0, min P]", 1.1, True, True)])  # restricted: by design
    assert eda_write.bound_sentence(none) == ""
    hit = _st([("idle", "idle power", "oct2026", "free", 0.05, True, True)])
    s = eda_write.bound_sentence(hit)
    assert "idle (v1 rig)" in s and "not interpretable" in s
    assert eda_write.bound_sentence(_st([])) == ""
    both = _st([("idle", "idle power", "nov2026", "free", 0.3, True, False),
                ("yolo11n", "forward power", "nov2026", "free", 0.7, True, False)])
    assert "below" in eda_write.idle_alpha_sentence(both, 0.7) and "0.30-0.30" in eda_write.idle_alpha_sentence(both, 0.7)
    assert "above" in eda_write.idle_alpha_sentence(both, 0.1)


def test_threads_sentence_handles_both_signs():
    less = eda_write.threads_sentence("yolo11n", 320, -14.1, 814.0, 113.0)
    assert "14.1 % less energy" in less and "-14.1" not in less
    more = eda_write.threads_sentence("yolo11s", 320, 6.3, 900.0, 140.0)
    assert "6.3 % more energy" in more and "less" not in more
    assert "the same energy" in eda_write.threads_sentence("m", 320, 0.0, 1.0, 1.0)


def test_report_has_no_hard_coded_claims(synthetic, tmp_path):
    cat, der, qc, cfgs = synthetic
    md = eda.write_eda_config(cat, der, qc, cfgs, tmp_path / "rep")
    assert "50-fold" not in md and "sits on the alpha bound" not in md
    assert "ICC = 1.0000" not in md
