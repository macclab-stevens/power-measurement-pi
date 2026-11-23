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
    assert re.search(r"Probe mode, L\d.*\*\*(indistinguishable|not significant|worse|better)", head)
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


def test_v2_vs_pooled_scores_both_arms_on_exactly_the_same_nov2026_pairs(analysis):
    a = analysis
    pooled, v2 = a.extras["pooled"].df, a.extras["v2only"].df
    nov = set(a.configs.loc[a.configs["rig"] == "nov2026", "config_key"])
    pairs = lambda d: set(zip(d.loc[d["candidate"] == "AUTO", "fold"], d.loc[d["candidate"] == "AUTO", "config_key"]))
    # the synthetic data has oct2026 configurations of yolov8n and yolo11n, as the real data does
    assert pairs(pooled) - {p for p in pairs(pooled) if p[1] in nov}, "fixture must have v1 rows in the pooled test sets"
    restricted = rt.restrict_to_nov(pooled, a.configs)
    assert pairs(restricted) == pairs(v2) and pairs(restricted) < pairs(pooled)
    t, n, note = rt.v2_vs_pooled(pooled, v2, a.configs, N_BOOT, MASTER)
    assert n == len(pairs(v2)) < len(pairs(pooled)) and "Wilcoxon p" in note and len(t) == 8
    # the pooled column is computed on the restricted rows, not on all of them
    want = rt.pooled_table(restricted, "S-LOMO", N_BOOT, MASTER).set_index("candidate")
    auto = t[t["candidate"] == "AUTO"].iloc[0]
    assert auto["pooled v1+v2 MdAPE % [95 % CI]"].startswith(rt.pct(want.loc["AUTO", "mdape"]))
    unrestricted = rt.pooled_table(pooled, "S-LOMO", N_BOOT, MASTER).set_index("candidate")
    assert rt.pct(unrestricted.loc["AUTO", "mdape"]) != rt.pct(want.loc["AUTO", "mdape"])
    assert set(v2["config_key"]) <= nov
    # arms that score different pairs are refused
    with pytest.raises(ValueError, match="same .fold, configuration. pairs"):
        rt.v2_vs_pooled(pooled, v2[v2["config_key"] != sorted(v2["config_key"])[0]], a.configs, N_BOOT, MASTER)


def test_pooled_arm_trained_on_all_rows_and_v2_arm_on_nov2026_rows(analysis, tmp_path):
    a = analysis
    n_pool, n_v2 = report.check_arms(a)
    assert n_pool == len(a.configs) and n_v2 == int((a.configs["rig"] == "nov2026").sum()) < n_pool
    assert f"trained on all {n_pool} configurations" in "\n".join(report.section_v2(a))
    a.extras["pooled"].rec["n_configs_train_table"] = n_v2        # an arm that did not train on every row
    try:
        with pytest.raises(ri.ReportInputError, match="mode-a-extras"):
            report.check_arms(a)
    finally:
        a.extras["pooled"].rec["n_configs_train_table"] = n_pool


def test_headline_table_is_one_claim_and_its_best_baseline(analysis, built):
    h = report.headline_rows(analysis)
    assert list(h["id"]) == ["H1", "H2"]
    r = h.iloc[0]
    assert (r["split"], r["target"], r["metric"], r["claim"]) == ("S-LOMO", "energy_mJ", "MdAPE %", "Mode A AUTO")
    assert re.fullmatch(r"\[[\d.]+, [\d.]+\]", r["95 % CI"]) and r["run_id"] == analysis.master.run_id and r["n"] > 0
    assert h.iloc[1]["claim"].startswith("Mode A B") and h.iloc[1]["split"] == "S-LOMO"
    text = built[1]["results.md"]
    head1 = text.split("### Mode L ladder (whole-cell energy APE; no single pre-registered rung)")[0]
    assert "Mode L" not in head1.split("## 1. Headline claims")[1].split("###")[0].split("| H2 |")[1].split("\n\n")[0]
    assert "### Mode L ladder (whole-cell energy APE; no single pre-registered rung)" in text
    ladder = text.split("### Mode L ladder (whole-cell energy APE; no single pre-registered rung)")[1].split("### Findings")[0]
    for rung in ("L0", "L1 ", "L1-cell (exploratory)", "L4-trees"):
        assert f"| {rung}" in ladder
    assert "H3" not in text


def test_every_error_table_states_the_noise_floor(built):
    _, files = built
    res = files["results.md"].split("## 12. Counter overhead")[0]
    ts = tables_of(res)
    assert len(ts) > 60
    for cap, hdr, _ in ts:
        assert "noise floor" in cap.lower(), (hdr, cap[:160])
    mode_a_part = res.split("## 2. Mode A")[1].split("## 9.")[0]
    for cap, hdr, _ in tables_of(mode_a_part):
        assert re.search(r"noise floor \(median repeat CV|noise floor not defined", cap, re.I), cap[:200]
    mode_l_part = res.split("## 9. Mode L")[1]
    for cap, hdr, _ in tables_of(mode_l_part):
        assert "Mode A's energy noise floor" in cap and "reused" in cap, cap[:200]
    for name in ("model_card_mode_a.md", "model_card_mode_l.md"):
        for cap, hdr, _ in tables_of(files[name].split("## Hashes and runs")[0]):
            assert "noise floor" in cap.lower(), (name, cap[:160])


def test_caption_requires_an_explicit_floor():
    with pytest.raises(TypeError):
        rt.caption(split="s", target="t", prov="n = 1", run_ids="r")
    assert "Noise floor" not in rt.caption(split="s", target="t", prov="n = 1", run_ids="r", floor=None)


def test_probe_caption_n_is_the_per_rung_pair_count_and_matches_the_table(built):
    _, files = built
    text = files["results.md"].split("## 11. Probe mode")[1].split("## 12.")[0]
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| rung | n pairs"))
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[i + 2:i + 10]]
    ns = {r[1] for r in rows}
    assert len(rows) == 8 and len(ns) == 1
    cap = next(ln for ln in reversed(lines[:i]) if ln.startswith("*Split"))
    assert f"n = {ns.pop()} ordered pairs per rung (8 rungs) of repeats in 6 configurations" in cap
    assert "n = 96" not in cap


def test_probe_verdict_applies_the_noise_floor_rule(analysis):
    probe = analysis.ml["probe"].copy()
    probe["rung"] = "L2"
    probe["exploratory"] = False
    probe["mode_a_mJ"] = probe["true_mJ"] * 1.10
    probe["mode_a_ape"] = 0.10
    probe["probe_mJ"] = probe["true_mJ"] * 1.09                    # 1 pp better, consistently
    probe["ape"] = 0.09
    probe = probe[probe["j"] < probe["k"]]
    probe["ape"] += np.linspace(0, 0.0005, len(probe))
    near = tl.probe_raw(probe, N_BOOT, MASTER, noise=0.02)
    assert near.iloc[0]["verdict"] == "indistinguishable (below noise floor)"
    assert tl.probe_raw(probe, N_BOOT, MASTER, noise=0.0).iloc[0]["verdict"] in ("better", "not significant")
    assert "verdict" in tl.show_probe(near).columns and "probe better?" not in tl.show_probe(near).columns
    worse = probe.assign(probe_mJ=probe["true_mJ"] * 1.5, ape=0.5)
    assert tl.probe_raw(worse, N_BOOT, MASTER, noise=0.02).iloc[0]["median_d"] > 0


def test_mode_l_ladder_rows_follow_the_protocol_order_with_exploratory_label(analysis):
    a = analysis
    raw = tl.ladder_raw(a.ml["lomo"], N_BOOT, MASTER)
    assert list(raw["rung"]) == ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees"]
    assert list(tl.show_ladder(raw)["rung"])[2] == "L1-cell (exploratory)"
    assert (raw["n"] == 15).all()


def test_no_output_is_written_outside_the_given_paths(built, tmp_path):
    paths, _ = built
    repo_reports = report.settings.HERE / "reports"
    made = (paths.out_dir / "results.md").stat().st_mtime       # the committed report may exist: it must be older
    assert not (repo_reports / "results.md").exists() or (repo_reports / "results.md").stat().st_mtime < made
    assert all(f.stat().st_mtime < made for f in (repo_reports / "fig").glob("report_*.png"))
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
