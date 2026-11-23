"""Results report and model cards on synthetic run outputs: headings, provenance of every table, exploratory
labels, claim logic and clear failures for missing inputs. Nothing here touches analytics/reports/."""
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import report
import report_inputs as ri
import report_tables as rt
import report_tables_l as tl
from synth_report import MASTER, family_map, make_configs, make_inputs, overhead_frame

N_BOOT = 15


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    paths = make_inputs(root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        files = report.build(paths, seed=MASTER, n_boot=N_BOOT, family=family_map(), overhead=overhead_frame())
    report.write(paths, files)
    return paths, files


@pytest.fixture(scope="module")
def analysis(built):
    return report.analyse(built[0], seed=MASTER, n_boot=N_BOOT, family=family_map())


def tables_of(text: str):
    """(preceding paragraph, header cells, n body rows) for every markdown table."""
    lines, out = text.splitlines(), []
    for i, ln in enumerate(lines):
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(---\|)+$", lines[i + 1]) and not lines[i - 1].startswith("|"):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            hdr = [c.strip() for c in re.split(r"(?<!\\)\|", ln)[1:-1]]
            k = i + 2
            while k < len(lines) and lines[k].startswith("|"):
                k += 1
            out.append((lines[j], hdr, k - i - 2))
    return out


def test_every_required_heading_and_file_is_present(built):
    paths, files = built
    assert set(files) == set(report.FILES)
    for h in report.HEADINGS:
        assert h in files["results.md"], h
    for split in ri.SPLITS:
        assert f"### {split}" in files["results.md"]
    for target in ri.TARGETS:
        assert f"#### {target}" in files["results.md"]
    for f in report.FILES:
        assert (paths.out_dir / f).is_file()


@pytest.mark.parametrize("name", report.FILES)
def test_every_table_carries_n_and_a_run_id(built, name):
    _, files = built
    ts = tables_of(files[name])
    assert len(ts) >= 3
    for cap, hdr, nrows in ts:
        assert nrows > 0, f"empty table under {cap!r}"
        has_n = any(h == "n" or h.startswith("n ") or h == "n_cells" for h in hdr) or "n = " in cap
        has_run = "run_id" in hdr or "run_id" in cap
        assert has_n, (hdr, cap)
        assert has_run, (hdr, cap)


def test_exploratory_rung_is_labelled_everywhere_it_appears(built):
    _, files = built
    res = files["results.md"]
    assert "L1-cell (exploratory)" in res
    assert re.search(r"amendment of\s+2026-11-23", res)
    # no table cell names the rung without the label
    for ln in res.splitlines():
        if ln.startswith("|") and re.search(r"(?<![\w-])L1-cell(?! \(exploratory\))", ln):
            pytest.fail(f"unlabelled exploratory rung: {ln}")
    assert "L1-cell (exploratory)" in files["model_card_mode_l.md"]


def test_negative_results_are_stated_in_findings(built):
    _, files = built
    head = files["results.md"].split("## 2.")[0]
    assert "Findings at a glance" in head
    assert re.search(r"\*\*(indistinguishable|not significant|worse|better)", head)
    assert "does not beat Mode A" in head or "the probe beats Mode A" in head
    assert "coverage" in head


def test_figures_written_and_small(built):
    paths, files = built
    pngs = sorted(paths.fig_dir.glob("report_*.png"))
    assert {p.name for p in pngs} >= {"report_error_vs_novelty.png", "report_learning_curve.png", "report_mode_l_ape.png",
                                     "report_probe_vs_mode_a.png"}
    assert sum(p.stat().st_size for p in pngs) < 3e6
    for p in pngs:
        assert f"fig/{p.name}" in files["results.md"]


def test_cards_have_the_section_11_headings_and_fixed_text(built):
    _, files = built
    for name in ("model_card_mode_a.md", "model_card_mode_l.md"):
        t = files[name]
        for h in ("## Intended use", "## Out of scope", "## Data and split", "## Features", "## Headline metrics", "## Coverage",
                  "## Worst subgroups", "## Limitations", "## Hashes and runs"):
            assert h in t, (name, h)
        assert "Other boards" in t and "SoC alone" in t and "lockbox" in t and "single device" in t
    assert "perf_event_perf" not in files["model_card_mode_l.md"]
    assert "perf_event_paranoid=2" in files["model_card_mode_l.md"]
    assert "System-wide or multi-process" in files["model_card_mode_l.md"]
    assert "e" * 64 in files["model_card_mode_a.md"]            # final model sha256


def test_seed_spread_lists_the_seeds_present(tmp_path):
    paths = make_inputs(tmp_path, seeds=(MASTER,))
    a = report.analyse(paths, seed=MASTER, n_boot=N_BOOT, family=family_map())
    assert sorted(a.seed_runs) == [MASTER]
    text = "\n".join(report.section_seeds(a))
    assert "Only one seed is present" in text and "1 of 3 present" in text


def test_missing_master_run_names_the_command(tmp_path):
    paths = make_inputs(tmp_path)
    lines = [ln for ln in paths.runs_log.read_text().splitlines()
             if not ('"split": "S-LSO"' in ln and '"target": "latency_ms"' in ln)]
    paths.runs_log.write_text("\n".join(lines) + "\n")
    with pytest.raises(ri.ReportInputError, match=r"cli\.py mode-a --split S-LSO --target latency_ms --seed 20261122"):
        report.analyse(paths, seed=MASTER, n_boot=N_BOOT)


@pytest.mark.parametrize("what,match", [("deploy", "deploy-validate"), ("configs", "cli.py tables"), ("qc", "cli.py qc")])
def test_missing_tables_name_their_command(tmp_path, what, match):
    paths = make_inputs(tmp_path)
    {"deploy": paths.deploy_csv, "configs": paths.configs_csv, "qc": paths.qc_csv}[what].unlink()
    with pytest.raises(ri.ReportInputError, match=match):
        report.analyse(paths, seed=MASTER, n_boot=N_BOOT)


def test_missing_mode_l_and_extras_name_their_commands(tmp_path):
    paths = make_inputs(tmp_path)
    shutil.rmtree(paths.mode_l_dir)
    paths.mode_l_dir.mkdir()
    with pytest.raises(ri.ReportInputError, match=r"cli\.py mode-l"):
        report.analyse(paths, seed=MASTER, n_boot=N_BOOT)
    paths = make_inputs(tmp_path / "b")
    shutil.rmtree(paths.extras_dir)
    paths.extras_dir.mkdir()
    with pytest.raises(ri.ReportInputError, match=r"cli\.py mode-a-extras"):
        report.analyse(paths, seed=MASTER, n_boot=N_BOOT)


def test_missing_probe_output_explains_how_to_make_it(tmp_path):
    paths = make_inputs(tmp_path)
    for f in paths.mode_l_dir.glob("*_probe.parquet"):
        f.unlink()
    with pytest.raises(ri.ReportInputError, match=r"probe.*mode-a --split S-LOMO"):
        report.analyse(paths, seed=MASTER, n_boot=N_BOOT)


def test_runs_whose_parquet_is_gone_are_skipped(tmp_path):
    paths = make_inputs(tmp_path)
    recs = ri.read_records(paths.runs_log)
    old = [r for r in recs if r.get("command") == "mode-a" and r.get("split") == "S-LOMO" and r.get("target") == "energy_mJ" and r.get("seed") == MASTER][-1]
    new = dict(old, run_id="f" * 32)                      # a later log line without a file
    with open(paths.runs_log, "a") as fh:
        fh.write(__import__("json").dumps(new) + "\n")
    assert ri.load_mode_a(paths, ri.read_records(paths.runs_log), MASTER)[("S-LOMO", "energy_mJ")].run_id == old["run_id"]


# ------------------------------------------------------------------ claim logic

def _frame(errs: dict, n=12, noise_free=False) -> pd.DataFrame:
    rows = []
    for cand, e in errs.items():
        y = np.linspace(100, 200, n)
        rows.append(pd.DataFrame({"fold": [f"m{i % 4}" for i in range(n)], "config_key": [f"c{i}" for i in range(n)],
                                  "base_model": [f"m{i % 4}" for i in range(n)], "candidate": cand, "y_true": y,
                                  "y_pred": y * (1 + np.asarray(e)), "lo90": y * 0.5, "hi90": y * 2.0}))
    return pd.concat(rows, ignore_index=True)


def test_best_baseline_is_the_lowest_pooled_mdape_among_b0_to_b3():
    d = _frame({"B0Median": [0.3] * 12, "B3LogLinear": [0.05] * 12, "B1MacsProportional": [0.2] * 12, "M1Ridge": [0.01] * 12})
    raw = rt.pooled_table(d, "S-LOMO", 30, 1)
    assert rt.best_baseline(raw) == "B3LogLinear"          # M1 is better but is not a baseline
    assert list(raw["candidate"]) == ["B0Median", "B1MacsProportional", "B3LogLinear", "M1Ridge"]


def test_bootstrap_clusters_are_base_models_and_config_keys_for_random():
    assert rt.group_col("S-LOMO") == "base_model" and rt.group_col("RANDOM") == "config_key"


def test_paired_difference_below_noise_floor_is_indistinguishable():
    rng = np.random.default_rng(0)
    base = rng.normal(0.10, 0.03, 40)
    d = _frame({"B3LogLinear": base, "M1Ridge": base - 0.002}, n=40)   # 0.2 pp better, consistently
    p = rt.paired_table(d, "B3LogLinear", noise=0.02)
    r = p.iloc[0]
    assert r["p_holm"] < 0.05 and r["verdict"].startswith("indistinguishable")
    q = rt.paired_table(d, "B3LogLinear", noise=0.0005)               # same data, tiny floor: better
    assert q.iloc[0]["verdict"] == "better"
    w = rt.paired_table(_frame({"B3LogLinear": base, "M1Ridge": base + 0.1}, n=40), "B3LogLinear", 0.001)
    assert w.iloc[0]["verdict"] == "worse"


def test_holm_is_applied_across_the_learned_candidates():
    rng = np.random.default_rng(1)
    base = rng.normal(0.1, 0.05, 30)
    d = _frame({"B3LogLinear": base, "M1Ridge": base * 0.9, "M2Trees": base * 1.1, "AUTO": base * 0.95}, n=30)
    p = rt.paired_table(d, "B3LogLinear", 0.0)
    assert (p["p_holm"] >= p["p"] - 1e-12).all() and len(p) == 3


def test_verdict_rules():
    assert rt.verdict(float("nan"), 0.01, 0.01) == "n/a"
    assert rt.verdict(-0.005, 0.001, 0.01).startswith("indistinguishable")
    assert rt.verdict(-0.05, 0.2, 0.01) == "not significant"
    assert rt.verdict(-0.05, 0.01, float("nan")) == "better"


def test_noise_floor_uses_repeat_configurations_only():
    c = make_configs()
    c.loc[0, ["n_cells", "energy_cv"]] = [1, 0.5]
    assert rt.noise_floor(c, "energy_mJ") == pytest.approx(0.02)
    assert np.isnan(rt.noise_floor(c, "mean_P_W"))


def test_worst_table_orders_by_ape_and_joins_qc(analysis):
    a = analysis
    w = rt.worst_table(a.master.df, a.configs, a.qc)
    assert len(w) == 10 and w["config_key"].is_unique
    ape = w["APE %"].astype(float)
    assert (np.diff(ape) <= 1e-9).all()
    assert {"QC rules", "max slip frac", "max residual ms", "n_cells"} <= set(w.columns)
    assert (w["n_cells"] == 3).all()


def test_v2_vs_pooled_scores_pooled_run_on_the_nov2026_test_configs_only(analysis):
    a = analysis
    t, n, note = rt.v2_vs_pooled(a.extras["pooled"].df, a.extras["v2only"].df, a.configs, N_BOOT, MASTER)
    nov = set(a.configs.loc[a.configs["rig"] == "nov2026", "config_key"])
    assert n == a.extras["v2only"].df.query("candidate == 'AUTO'").shape[0]
    assert set(a.extras["v2only"].df["config_key"]) <= nov
    assert len(t) == 8 and "Wilcoxon p" in note


def test_mode_l_ladder_rows_follow_the_protocol_order_with_exploratory_label(analysis):
    a = analysis
    raw = tl.ladder_raw(a.ml["lomo"], N_BOOT, MASTER)
    assert list(raw["rung"]) == ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees"]
    assert list(tl.show_ladder(raw)["rung"])[2] == "L1-cell (exploratory)"
    assert (raw["n"] == 15).all()


def test_no_output_is_written_outside_the_given_paths(built, tmp_path):
    paths, _ = built
    repo_reports = report.settings.HERE / "reports"
    assert not (repo_reports / "results.md").exists()
    assert not list((repo_reports / "fig").glob("report_*.png"))
    assert (paths.out_dir / "results.md").is_file()


# ------------------------------------------------------------------ CLI

def test_cli_registers_the_new_commands_and_report_writes_to_the_given_paths(tmp_path, monkeypatch):
    import cli
    import data
    paths = make_inputs(tmp_path)
    monkeypatch.setattr(ri.Paths, "default", classmethod(lambda cls: paths))
    monkeypatch.setattr(data, "catalogue", lambda: pd.DataFrame(columns=["cell_id", "family"]))
    args = cli.build_parser().parse_args(["report", "--n-boot", "10"])
    assert args.n_boot == 10 and args.seed is None
    assert cli.build_parser().parse_args(["mode-a-extras", "--seed", "5"]).seed == 5
    assert args.fn(args) == 0
    assert (paths.out_dir / "results.md").is_file() and (paths.out_dir / "model_card_mode_l.md").is_file()
    assert ri.read_records(paths.runs_log)[-1]["command"] == "report"


def test_cli_report_fails_clearly_when_a_run_is_missing(tmp_path, monkeypatch, capsys):
    import cli
    import data
    paths = make_inputs(tmp_path)
    paths.deploy_csv.unlink()
    monkeypatch.setattr(ri.Paths, "default", classmethod(lambda cls: paths))
    monkeypatch.setattr(data, "catalogue", lambda: pd.DataFrame(columns=["cell_id", "family"]))
    args = cli.build_parser().parse_args(["report"])
    assert args.fn(args) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "cli.py deploy-validate" in err
    assert not (paths.out_dir / "results.md").exists()
