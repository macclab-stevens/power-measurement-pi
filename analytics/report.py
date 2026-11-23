"""Results report (`results.md`), model cards and figures from the saved evaluation outputs.

Claim discipline: every headline maps to one split, one target, one metric, an interval and a run_id;
exploratory items are labelled; negative results are shown as prominently as positive ones. Inputs are
the outputs of `mode-a`, `mode-a-extras`, `deploy-validate`, `mode-l`, `tables` and `qc`; a missing one
raises `report_inputs.ReportInputError` naming the command that makes it. Output paths come from
`report_inputs.Paths`, so tests never touch analytics/reports/.

`report.py` (this file) assembles the Mode A part; `report_l.py` the Mode L part; `report_cards.py` the
two model cards; tables and figures live in `report_tables*.py` and `report_figs.py`."""
from dataclasses import dataclass, field

import pandas as pd

import eda_write as ew
import report_cards
import report_figs as figs
import report_inputs as ri
import report_l
import report_tables as rt
import settings
import report_tables_l as tl
from report_inputs import Paths, ReportInputError

N_BOOT = 2000
SPLIT_TEXT = {
    "S-LOMO": "leave one base model out (9 folds; YOLO folds `sibling-seen`, classifier folds `new-architecture`); the headline split",
    "S-LSO": "leave one YOLO size pair out (n and m pairs `extrapolation`, s pair `interpolation`)",
    "S-LVO": "leave one INT8 variant out (`variant-seen-on-other-model`); FP16 (one configuration) is unsupported and not scored",
    "S-LCO": "leave one clock out (1500, 2000, 2400 MHz) plus train 1500 + 2000 and test 2400; the two 2400 MHz folds share their test configurations, so pooled counts repeat them",
    "RANDOM": "random 5-fold over configurations; pipeline sanity only, never a headline",
}
HEADINGS = ["## 1. Headline claims", "## 2. Mode A results by split and target", "## 3. Seed spread of the headline",
            "## 4. Error against novelty", "## 5. Worst-10 configurations", "## 6. Learning curve",
            "## 7. v2-only against pooled training", "## 8. Deployment validation", "## 9. Mode L: live power ladder",
            "## 10. PMIC-only check", "## 11. Probe mode against Mode A", "## 12. Counter overhead and stated limits",
            "## 13. Runs and provenance"]
FILES = ["results.md", "model_card_mode_a.md", "model_card_mode_l.md"]


@dataclass
class Analysis:
    paths: Paths
    seed: int
    n_boot: int
    configs: pd.DataFrame
    qc: pd.DataFrame
    recs: list
    ma: dict
    seed_runs: dict
    extras: dict
    ml: dict
    deploy: pd.DataFrame
    deploy_rec: dict
    family: pd.Series
    overhead: pd.DataFrame | None
    pooled: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    @property
    def boot_seed(self) -> int:
        """Bootstrap seed: the config.toml master seed, whatever seed's runs are reported."""
        return settings.seed()

    @property
    def master(self) -> ri.RunRef:
        return self.ma[("S-LOMO", "energy_mJ")]


def analyse(paths: Paths, *, seed: int | None = None, n_boot: int = N_BOOT, family: pd.Series | None = None,
            overhead: pd.DataFrame | None = None) -> Analysis:
    """Load every input (raising ReportInputError if one is missing) and compute the Mode A tables once."""
    seed = settings.seed() if seed is None else seed
    recs = ri.read_records(paths.runs_log)
    configs = ri.load_table(paths.configs_csv, "configuration table", "uv run python cli.py tables")
    qc = ri.load_table(paths.qc_csv, "QC table", "uv run python cli.py qc")
    dep = ri.load_table(paths.deploy_csv, "deployment validation", "uv run python cli.py deploy-validate")
    a = Analysis(paths, seed, n_boot, configs, qc, recs, ri.load_mode_a(paths, recs, seed),
                 ri.load_seed_runs(paths, recs, seed), ri.load_extras(paths, recs, seed), ri.load_mode_l(paths, recs, seed),
                 dep, ri.latest_record(recs, command="deploy-validate") or {},
                 family if family is not None else pd.Series(dtype=object), overhead)
    a.noise = {t: rt.noise_floor(configs, t) for t in ri.TARGETS}
    for (split, target), ref in a.ma.items():
        raw = rt.pooled_table(ref.df, split, n_boot, settings.seed())
        a.pooled[(split, target)] = raw
        a.base[(split, target)] = rt.best_baseline(raw)
        a.paired[(split, target)] = rt.paired_table(ref.df, a.base[(split, target)], a.noise[target])
    return a


# ------------------------------------------------------------------ rendering helpers

def md(df: pd.DataFrame, fmt=None) -> str:
    return ew.md_table(df, fmt) + "\n"


def nf_energy(a: Analysis) -> str:
    return rt.floor_text(a.noise["energy_mJ"], "energy_mJ")


def prov_ma(a: Analysis, split: str, target: str) -> str:
    ref = a.ma[(split, target)]
    keys = ref.df["config_key"].unique()
    return rt.version_text(a.configs, keys)


def headline_rows(a: Analysis) -> pd.DataFrame:
    """The headline claim H1 (AUTO, S-LOMO, energy_mJ, MdAPE with its interval) and its comparator H2 (best
    baseline, same split, target and metric). Mode L has no single pre-registered rung: it is a separate table."""
    rows = []
    raw = a.pooled[("S-LOMO", "energy_mJ")].set_index("candidate")
    ref = a.master
    for cid, cand, status in (("H1", "AUTO", "pre-registered headline"), ("H2", a.base[("S-LOMO", "energy_mJ")], "best baseline (by MdAPE)")):
        r = raw.loc[cand]
        rows.append({"id": cid, "claim": f"Mode A {rt.SHORT[cand]}", "split": "S-LOMO", "target": "energy_mJ",
                     "metric": "MdAPE %", "estimate": rt.pct(r["mdape"]), "95 % CI": f"[{rt.pct(r['mdape_lo'])}, {rt.pct(r['mdape_hi'])}]",
                     "n": int(r["n"]), "run_id": ref.run_id, "status": status})
    return pd.DataFrame(rows)


def findings(a: Analysis) -> list:
    """Plain-language verdicts, positive and negative alike, computed from the tables."""
    out = []
    base, p = a.base[("S-LOMO", "energy_mJ")], a.paired[("S-LOMO", "energy_mJ")]
    for r in p.itertuples():
        out.append(f"Mode A, S-LOMO energy: {rt.SHORT[r.candidate]} against the best baseline ({rt.SHORT[base]}): "
                   f"**{r.verdict}** (median paired APE difference {100 * r.median_dape:+.2f} pp, Holm p = {r.p_holm:.3g}, n = {r.n}; "
                   f"run_id `{a.master.run_id}`).")
    auto = a.pooled[("S-LOMO", "energy_mJ")].set_index("candidate").loc["AUTO"]
    out.append(f"Mode A, S-LOMO energy: AUTO 90 % interval coverage {auto['coverage90']:.2f} against the nominal 0.90"
               + (" (below nominal: the intervals under-cover)" if auto["coverage90"] < 0.9 else "")
               + f", n = {int(auto['n'])}; coarse with 9 models.")
    flagged = a.master.df.query("candidate == 'AUTO'")["flagged"].astype(bool)
    out.append(f"Mode A, S-LOMO energy: {int(flagged.sum())} of {len(flagged)} AUTO predictions carry the `no-similar-model` flag.")
    out += report_l.findings(a)
    return out


def section_headline(a: Analysis) -> list:
    r = a.master
    return ["## 1. Headline claims\n",
            f"Every row names one split, one target, one metric, an interval and a run_id. Intervals are cluster-bootstrap 95 % "
            f"({a.n_boot} resamples, seed {a.boot_seed}; clusters are base models). Percentages are percent absolute error unless stated. "
            f"Data: {prov_ma(a, 'S-LOMO', 'energy_mJ')}. Run `{r.rec.get('commit', '')[:10]}`, data hash `{str(r.rec.get('data_hash', ''))[:12]}`. "
            f"Nothing here is a lockbox result: the lockbox has not been collected.\n",
            rt.caption(split="S-LOMO", target="energy_mJ", prov=prov_ma(a, "S-LOMO", "energy_mJ"), run_ids=r.run_id,
                       floor=rt.floor_text(a.noise["energy_mJ"], "energy_mJ"), extra="H1 is the headline; H2 is its best-baseline comparator."),
            md(headline_rows(a)),
            "### Mode L ladder (whole-cell energy APE; no single pre-registered rung)\n",
            "Mode L is a ladder: no single rung is the pre-registered headline, so every rung is listed with the same metric and interval. "
            f"{report_l.overhead_text(a)}\n",
            report_l.ladder_caption(a), md(report_l.ladder_headline(a)),
            "### Findings at a glance (positive and negative)\n", *[f"- {x}" for x in findings(a)], "\n"]


def section_mode_a(a: Analysis) -> list:
    L = ["## 2. Mode A results by split and target\n",
         "Candidates in protocol order: B0 median, B1 energy proportional to MACs per clock, B2 most similar seen model scaled by MACs, "
         "B3 log-linear OLS, M1 ridge, M2 gradient-boosted trees, M3 factored latency x power (energy only), AUTO (best of M1 to M3 by inner "
         "leave-one-model-out MdAPE). Baselines have no prediction interval, so their coverage is blank. Coverage rests on few clusters and is coarse. "
         "Paired tests use absolute log errors per test configuration; \"indistinguishable\" means the median paired APE difference is below the noise floor.\n"]
    for split in ri.SPLITS:
        L += [f"### {split}\n", f"{SPLIT_TEXT[split]}.\n"]
        for target in ri.TARGETS:
            ref, key = a.ma[(split, target)], (split, target)
            base, nf = a.base[key], a.noise[target]

            def cap(extra, target=target, split=split, ref=ref, nf=nf):
                return rt.caption(split=split, target=f"{target} ({rt.NOUN[target]}, {rt.UNITS[target]})",
                                  prov=prov_ma(a, split, target), run_ids=ref.run_id, floor=rt.floor_text(nf, target),
                                  extra=f"Seed {ref.seed}. {extra}")
            L += [f"#### {target}\n", cap(f"Best baseline {rt.SHORT[base]}; bootstrap clusters: {rt.group_col(split)}."),
                  md(rt.show_pooled(a.pooled[key], target)),
                  cap("Paired Wilcoxon against the best baseline, Holm-adjusted over the learned candidates."),
                  md(rt.show_paired(a.paired[key], base)),
                  cap("By similarity class; n counts predictions."), md(rt.similarity_table(ref.df)),
                  cap("By base model."), md(rt.model_table(ref.df, base), {"novelty": "{:.3f}"})]
            if target == "energy_mJ":
                L += [cap("By precision, image size, threads, batch, clock and rig (fixed subgroups)."),
                      md(rt.factor_table(ref.df, a.configs, base))]
    return L


def section_seeds(a: Analysis) -> list:
    t, raw = rt.seed_table(a.seed_runs, a.configs, a.n_boot, a.boot_seed)
    cap = rt.caption(floor=nf_energy(a), split="S-LOMO", target="energy_mJ", prov=prov_ma(a, "S-LOMO", "energy_mJ"),
                     run_ids=list(t["run_id"]),
                     extra=f"Protocol seeds {list(ri.SEED_SPREAD)}; {len(t)} of {len(ri.SEED_SPREAD)} present in runs.jsonl.")
    rng = (f"AUTO MdAPE ranges from {rt.pct(raw['auto'].min())} % to {rt.pct(raw['auto'].max())} % over the {len(raw)} seeds present."
           if len(raw) > 1 else "Only one seed is present, so no seed spread can be stated.")
    return ["## 3. Seed spread of the headline\n", cap, md(t), rng + "\n"]


def section_novelty(a: Analysis) -> list:
    ref = a.master
    auto = ref.df[ref.df["candidate"] == "AUTO"]
    name = figs.error_vs_novelty(auto, a.paths.fig_dir)
    return ["## 4. Error against novelty\n",
            rt.caption(split="S-LOMO", target="energy_mJ", prov=prov_ma(a, "S-LOMO", "energy_mJ"), run_ids=ref.run_id,
                       floor=rt.floor_text(a.noise["energy_mJ"], "energy_mJ"), extra=f"n = {len(auto)} AUTO predictions plotted. Hollow markers carry the `no-similar-model` flag."),
            figs.fig_link(a.paths, name, "AUTO absolute percentage error against novelty, S-LOMO energy")]


def section_worst(a: Analysis) -> list:
    ref = a.master
    w = rt.worst_table(ref.df, a.configs, a.qc)
    n_auto = int((ref.df["candidate"] == "AUTO").sum())
    return ["## 5. Worst-10 configurations\n",
            rt.caption(floor=nf_energy(a), split="S-LOMO", target="energy_mJ", prov=f"n = {len(w)} of {n_auto} AUTO predictions", run_ids=ref.run_id,
                       extra="Largest AUTO absolute percentage errors, joined to the QC rows of the configuration's cells "
                             "(the narrative autopsy is in `analytics/reports/autopsy.md`)."), md(w, {"measured": "{:.1f}", "predicted": "{:.1f}", "novelty": "{:.3f}"})]


def section_learning(a: Analysis) -> list:
    lc = a.extras["learning_curve"]
    rec = lc.rec
    name = figs.learning_curve(lc.df, a.paths.fig_dir)
    return ["## 6. Learning curve\n",
            rt.caption(floor=nf_energy(a), split="S-LOMO YOLO folds", target="energy_mJ", prov=f"n = {len(lc.df)} fits",
                       run_ids=lc.run_id, extra=f"Candidate {rt.SHORT.get(rec.get('candidate'), rec.get('candidate'))} with the final-fit "
                                                f"hyperparameters (no inner search); seed {rec.get('seed')}; at most 10 training subsets per k, all other "
                                                f"configurations (non-YOLO) always in training."),
            md(rt.learning_table(lc.df)), figs.fig_link(a.paths, name, "MdAPE against number of training YOLO models")]


def check_arms(a: Analysis) -> tuple:
    """Training sizes of the two arms from their run records; the pooled arm must have trained on every row of the
    configuration table and the v2-only arm on its nov2026 rows (else `mode-a-extras` ran on another table)."""
    ex = a.extras
    n_pool, n_v2 = ex["pooled"].rec.get("n_configs_train_table"), ex["v2only"].rec.get("n_configs_train_table")
    want = (len(a.configs), int((a.configs["rig"] == "nov2026").sum()))
    if (n_pool, n_v2) != want:
        raise ReportInputError(f"mode-a-extras trained the pooled / v2-only arms on {n_pool} / {n_v2} configurations but the table has "
                               f"{want[0]} / {want[1]} (all / nov2026); rerun `uv run python cli.py mode-a-extras`")
    return n_pool, n_v2


def section_v2(a: Analysis) -> list:
    ex = a.extras
    t, n, note = rt.v2_vs_pooled(ex["pooled"].df, ex["v2only"].df, a.configs, a.n_boot, a.boot_seed)
    n_pool, n_v2 = check_arms(a)
    return ["## 7. v2-only against pooled training\n",
            "S-LOMO YOLO folds, energy. Both runs are scored on the same test configurations (the nov2026 rig ones); only the training "
            "data differ. The v2-only run drops every v1 (oct2026) configuration, including the three non-YOLO models.\n",
            rt.caption(floor=nf_energy(a), split="S-LOMO YOLO folds", target="energy_mJ", prov=f"n = {n} test configurations (dataset v2, nov2026 rig)",
                       run_ids=[ex["pooled"].run_id, ex["v2only"].run_id],
                       extra=f"First run id: pooled, trained on all {n_pool} configurations; second: v2-only, trained on the {n_v2} nov2026 ones."),
            md(t), note + "\n"]


def section_deploy(a: Analysis) -> list:
    d = a.deploy
    ids = a.deploy_rec.get("run_id")
    mids = [a.ma[("S-LOMO", t)].run_id for t in ("energy_mJ", "latency_ms")]
    run_ids = ([ids] if ids else []) + mids
    cap = lambda extra: rt.caption(floor=nf_energy(a), split="held-out S-LOMO AUTO predictions", target="whole-deployment energy, J",
                                   prov=f"n = {len(d)} validation rows (dataset v1 and v2 cells as listed)", run_ids=run_ids, extra=extra)
    return ["## 8. Deployment validation\n",
            "The calculator is fed only held-out Mode A predictions. The fan rows fit their offsets on the same cells and are not a validation; "
            "the soak rows use the soak factor of the other soak only. Run ids: deploy-validate run (if logged), then the Mode A energy and "
            "latency runs it used (the latest at the time the report was built).\n",
            cap("Summary by case and phase."), md(tl.deploy_summary(d)),
            cap("Rows other than `serve-default` (see `reports/deploy_validation.csv` for all)."),
            md(tl.deploy_rows(d), {"measured J": "{:.2f}", "predicted J": "{:.2f}", "lo90 J": "{:.2f}", "hi90 J": "{:.2f}"})]


def section_provenance(a: Analysis) -> list:
    rows = []

    def add(role, ref):
        r = ref.rec
        rows.append({"role": role, "run_id": ref.run_id, "command": r.get("command"), "split": r.get("split", ""),
                     "target": r.get("target", ""), "seed": r.get("seed"), "n": len(ref.df),
                     "commit": str(r.get("commit", ""))[:10], "dirty": r.get("dirty"), "ts": r.get("ts")})
    for (s, t), ref in a.ma.items():
        add(f"Mode A {s}", ref)
    for s, ref in a.seed_runs.items():
        add("seed spread (S-LOMO energy)", ref)
    for k in ("learning_curve", "pooled", "v2only"):
        add(f"extras {k}", a.extras[k])
    for k, key in (("lomo", "lomo"), ("pmic_v1", "pmic_v1"), ("probe", "probe")):
        rec = a.ml["rec"] if k == "lomo" else a.ml.get(f"{k}_rec", {})
        rows.append({"role": f"Mode L {k}", "run_id": a.ml["run_id"], "command": "mode-l", "split": "", "target": "", "seed": rec.get("seed"),
                     "n": len(a.ml[key]), "commit": str(rec.get("commit", ""))[:10], "dirty": rec.get("dirty"), "ts": rec.get("ts")})
    sp = {r["split"]: r for r in (x.rec for x in a.ma.values())}
    hashes = "; ".join(f"{s} `{str(r.get('split_sha256', ''))[:12]}`" for s, r in sp.items())
    return ["## 13. Runs and provenance\n",
            f"Table sha256 `{str(a.master.rec.get('table_sha256', ''))[:16]}`; split hashes (sha256, 12 characters): {hashes}. "
            f"Run seed {a.seed}, bootstrap seed {a.boot_seed}. Dataset: v1 (branch `dataset-2026-10-matrix`, 224 cells) and v2 (branch `dataset-2026-11-v2`, 179 cells). "
            "Protocol: `analytics/PROTOCOL.md`; its amendment log applies (the `L1-cell` rung is exploratory).\n",
            rt.caption(split="all", target="all", prov=f"n = {len(rows)} run outputs used", run_ids="see table", floor=None), md(pd.DataFrame(rows))]


def render_results(a: Analysis) -> str:
    parts = ["# Results report: Mode A and Mode L\n",
             "Generated by `cd analytics && uv run python cli.py report`. Sections follow the order of the protocol; "
             "each table is preceded by a line giving its split, target, n, dataset version(s) and run_id.\n",
             *section_headline(a), *section_mode_a(a), *section_seeds(a), *section_novelty(a), *section_worst(a),
             *section_learning(a), *section_v2(a), *section_deploy(a), *report_l.sections(a), *section_provenance(a)]
    return "\n".join(parts).rstrip() + "\n"


def build(paths: Paths, **kw) -> dict:
    """Render every output; figures are written to `paths.fig_dir`. Returns {file name: text}."""
    a = analyse(paths, **kw)
    return {"results.md": render_results(a), "model_card_mode_a.md": report_cards.mode_a_card(a),
            "model_card_mode_l.md": report_cards.mode_l_card(a)}


def write(paths: Paths, files: dict) -> list:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in files.items():
        (paths.out_dir / name).write_text(text)
        out.append(paths.out_dir / name)
    return out
