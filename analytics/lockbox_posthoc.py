"""Post hoc (R2) scoring of the lockbox: "the lockbox was seen; not a confirmatory test".

The frozen R1 lockbox result (`reports/lockbox.md`, `derived/lockbox/<run>_*.parquet`) is never touched; this module only READS
it, to put the R1 numbers beside the R2 ones. Everything R2 is fitted on the TRAINING tables alone:
  * Mode A: every candidate of `mode_a.CANDIDATES_BY_TARGET` and AUTO, per target, in the regime that fits each lockbox
    configuration ("seen-model" when its base model is a training base model, else "new-model"; `mode_a.fit_candidates`),
    predicting the lockbox configurations from the static features exactly as `lockbox.feature_rows` builds them, with the
    90 % interval of the regime's fold-weighted inner residuals;
  * calculator: serve-phase energy of the lockbox cells from the R2 AUTO energy and latency predictions through the closed-loop
    gap scenario (`lockbox_score.calculator_posthoc_gap`);
  * Mode L: R2 finals of every rung (`mode_l.fit_final`, pickles `mode_l_<rung>_r2.pkl`, the R1 pickles are never overwritten) on
    the QC-primary training YOLO counter cells, scored on the slip-corrected lockbox windows with the CMOS features;
  * probe: Mode L power x the measured latency of another repeat against the Mode A R2 AUTO prediction.
Isolation: `check_training_only` refuses a training table that holds a lockbox configuration or cell, and a training window table
that holds a lockbox cell. Outputs go to `reports/r2/lockbox_posthoc*`."""
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
import deploy
import lockbox as lb
import lockbox_score as ls
import mode_a
import mode_l
import mode_l_models as mlm
import report_inputs as ri
import runlog
import settings
import windows

LABEL = "post hoc (R2): the lockbox was seen; not a confirmatory test"
SUFFIX = "_r2"
TARGETS = ri.TARGETS
NEW, SEEN, ALL = "new model", "seen model", "all"
GROUPS = (ALL, NEW, SEEN)
HEADLINE_RUNGS = ("L0", "L3-trees", "L5-cmos", "L5-cmos-trees")
FAMILIES = ("YOLO", "non-YOLO")
MODE_A_COLS = ["config_key", "model", "imgsz", "regime", "group", "target", "candidate", "chosen", "pred", "lo90", "hi90",
               "truth", "ape", "inside", "score", "params"]


# ------------------------------------------------------------------ isolation

def check_training_only(t: pd.DataFrame, feats: pd.DataFrame | None = None, w: pd.DataFrame | None = None,
                        lockbox_cells=()) -> None:
    """LockboxError unless `t` (the Mode A training table) and `w` (the training windows) hold no lockbox row: no `source`
    column marking one, no `cell_ids` entry naming a lockbox cell or one of `lockbox_cells`, no configuration of `feats` (the
    lockbox feature rows), no window of a lockbox cell."""
    bad = set()
    if "source" in t.columns and t["source"].isin(data.HOLDOUT_SOURCES).any():
        bad.add("a row with source lockbox")
    cells = {str(c) for c in lockbox_cells}
    if "cell_ids" in t.columns:
        ids = {c for s in t["cell_ids"].astype(str) for c in s.split(";")}
        if any(c.startswith("lockbox") for c in ids) or ids & cells:
            bad.add("a lockbox cell id in cell_ids")
    if feats is not None and set(feats["config_key"]) & set(t["config_key"]):
        bad.add("a lockbox configuration " + ", ".join(sorted(set(feats["config_key"]) & set(t["config_key"]))))
    if w is not None and len(w) and (w["cell_id"].astype(str).str.startswith("lockbox").any() or set(w["cell_id"]) & cells):
        bad.add("a window of a lockbox cell")
    if bad:
        raise lb.LockboxError("post hoc R2 fits use training data only, but the training input has " + "; ".join(sorted(bad)))


# ------------------------------------------------------------------ Mode A

def regime_of_model(model: str, train_models) -> str:
    """"seen-model" if `model` is a training base model, else "new-model"."""
    return "seen-model" if model in set(train_models) else "new-model"


def mode_a_predictions(t: pd.DataFrame, feats: pd.DataFrame, desc: pd.DataFrame, seed: int, grids: dict | None = None,
                       targets=TARGETS, fit=None) -> pd.DataFrame:
    """One row per lockbox configuration x target x candidate (plus AUTO): the prediction of the finals fitted on `t` only in the
    regime of that configuration, its 90 % interval (the regime's inner-residual quantiles), the chosen grid point and, for AUTO,
    the candidate it chose (`chosen`). `desc` must hold the descriptors of the lockbox models (static features only).
    `fit` replaces `mode_a.fit_candidates` (tests)."""
    check_training_only(t, feats)
    fit = fit or mode_a.fit_candidates
    train_models = set(t["base_model"])
    reg = {m: regime_of_model(m, train_models) for m in sorted(set(feats["base_model"]))}
    rows = []
    for target in targets:
        for regime in sorted(set(reg.values())):
            sub = feats[feats["base_model"].map(reg) == regime].reset_index(drop=True)
            fits = fit(t, target, seed, desc, regime, grids)
            for cand, f in fits.items():
                p = np.asarray(f["model"].predict(sub), float)
                for i, r in enumerate(sub.itertuples()):
                    rows.append({"config_key": r.config_key, "model": r.base_model, "imgsz": int(r.imgsz), "regime": regime,
                                 "group": NEW if regime == "new-model" else SEEN, "target": target, "candidate": cand,
                                 "chosen": f["chosen"], "pred": float(p[i]), "lo90": float(p[i] * np.exp(f["q5"])),
                                 "hi90": float(p[i] * np.exp(f["q95"])), "score": float(f["score"]),
                                 "params": json.dumps(f["params"], sort_keys=True)})
    return pd.DataFrame(rows)


def score_mode_a(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """`pred` with the measured truth of its configuration, the APE and whether the truth is inside [lo90, hi90]."""
    tr = truth.set_index("config_key")
    y = pred.apply(lambda r: float(tr.loc[r["config_key"], r["target"]]) if r["config_key"] in tr.index else np.nan, axis=1)
    out = pred.assign(truth=y.to_numpy())
    out["ape"] = np.where(out["truth"] > 0, (out["pred"] - out["truth"]).abs() / out["truth"], np.nan)
    out["inside"] = (out["lo90"] <= out["truth"]) & (out["truth"] <= out["hi90"])
    return out[MODE_A_COLS]


def summarise(scores: pd.DataFrame, keys=("target", "candidate")) -> pd.DataFrame:
    """Per `keys` and group (all / new model / seen model, from the `group` column): n, MdAPE, MAPE, MaxAPE, bias (mean signed
    error), inside_90 and the median width hi90 / lo90 of the interval. Rows without a finite truth are not counted."""
    rows = []
    for k, g in scores.groupby(list(keys), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        for label in GROUPS:
            s = g if label == ALL else g[g["group"] == label]
            s = s[np.isfinite(s["truth"])]
            if s.empty:
                continue
            ape = s["ape"].to_numpy(float)
            rows.append({**dict(zip(keys, k)), "group": label, "n": len(s), "mdape": float(np.median(ape)),
                         "mape": float(ape.mean()), "maxape": float(ape.max()),
                         "bias_pct": float(((s["pred"] - s["truth"]) / s["truth"]).mean()),
                         "inside_90": float(s["inside"].mean()), "width": float((s["hi90"] / s["lo90"]).median())})
    return pd.DataFrame(rows)


def auto_primary(pred: pd.DataFrame, novelty: dict) -> pd.DataFrame:
    """The AUTO energy and latency rows of `mode_a_predictions` in the schema `lockbox_score.calculator_posthoc_gap` reads
    (role primary; `novelty` {model: value}; `flagged` False: there is no R2 applicability threshold here)."""
    a = pred[(pred["candidate"] == mode_a.AUTO) & pred["target"].isin(["energy_mJ", "latency_ms"])]
    return pd.DataFrame({"model": a["model"].to_numpy(), "imgsz": a["imgsz"].to_numpy(), "role": "primary",
                         "target": a["target"].to_numpy(), "pred": a["pred"].to_numpy(), "lo90": a["lo90"].to_numpy(),
                         "hi90": a["hi90"].to_numpy(), "novelty": a["model"].map(novelty).to_numpy(float), "flagged": False})


# ------------------------------------------------------------------ frozen R1 (read only)

def frozen_run_id(recs: list) -> str | None:
    """Run id of the first (the scored-once) `lockbox-score` line of the runlog, None if there is none."""
    for r in recs:
        if r.get("command") == "lockbox-score":
            return r["run_id"]
    return None


def frozen_frame(recs: list, name: str) -> pd.DataFrame | None:
    """`derived/lockbox/<frozen run>_<name>.parquet` read-only, None if the file does not exist."""
    rid = frozen_run_id(recs)
    p = settings.derived_dir() / "lockbox" / f"{rid}_{name}.parquet"
    return pd.read_parquet(p) if rid and p.is_file() else None


def check_truth_matches_frozen(truth: pd.DataFrame, frozen: pd.DataFrame | None) -> bool | None:
    """True when `truth` equals the frozen truth table (same configurations and targets to 1e-9 relative); LockboxError if it
    differs; None when no frozen table is available."""
    if frozen is None:
        return None
    a, b = truth.sort_values("config_key").reset_index(drop=True), frozen.sort_values("config_key").reset_index(drop=True)
    cols = list(ls.TARGET_COL)
    if list(a["config_key"]) != list(b["config_key"]) or not np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), rtol=1e-9):
        raise lb.LockboxError("the post hoc truth table differs from the frozen one (derived/lockbox/<run>_truth.parquet)")
    return True


def r1_mode_a(frozen: pd.DataFrame) -> pd.DataFrame:
    """The frozen R1 per-configuration scores in the layout of `summarise` (candidate `<name> (primary|secondary)`)."""
    f = frozen.assign(group=np.where(frozen["new_model"], NEW, SEEN),
                      candidate=frozen["candidate"] + " (" + frozen["role"] + ")")
    return f[["config_key", "model", "imgsz", "group", "target", "candidate", "pred", "lo90", "hi90", "truth", "ape", "inside"]]


# ------------------------------------------------------------------ calculator

def compare_calc(r2: pd.DataFrame, r1: pd.DataFrame | None) -> pd.DataFrame:
    """Median / mean / max APE per variant of `lockbox_score.summarise_posthoc_gap`, R2 AUTO beside the R1 frozen predictions
    (`r1`: the cell table of reports/r2/lockbox_calculator_gap.csv, None to leave the R1 columns out)."""
    s2 = ls.summarise_posthoc_gap(r2).set_index("variant")
    out = s2.rename(columns={"median_ape": "R2 median", "mean_ape": "R2 mean", "max_ape": "R2 max"})[["n", "R2 median", "R2 mean", "R2 max"]]
    if r1 is not None and len(r1):
        s1 = ls.summarise_posthoc_gap(r1).set_index("variant")
        out = out.join(s1.rename(columns={"median_ape": "R1 median", "mean_ape": "R1 mean", "max_ape": "R1 max"})[
            ["R1 median", "R1 mean", "R1 max"]])
    return out.reset_index()


# ------------------------------------------------------------------ Mode L

def lockbox_windows_r2(cat_l: pd.DataFrame, qc_l: pd.DataFrame) -> tuple:
    """(windows, {cell_id: slip status}) of the QC-primary lockbox cells with counters: `windows.cell_windows` (slip-corrected
    times, CMOS features) with cpu_key, fan_mode and governor from the catalogue (`windows.assign_settings`)."""
    import data
    ok = set(qc_l.loc[qc_l["primary"].astype(bool), "cell_id"])
    parts, slip = [], {}
    for _, r in cat_l[cat_l["cell_id"].isin(ok) & cat_l["has_counters"].astype(bool)].iterrows():
        d = windows.cell_windows(Path(r["path"]), data.read_meta(r["path"]))
        if d is not None and len(d):
            slip[r["cell_id"]] = (d.attrs.get("slip") or {}).get("status", "none")
            parts.append(windows.assign_settings(d, r))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), slip


def fit_mode_l(w: pd.DataFrame, cat: pd.DataFrame, qc: pd.DataFrame, seed: int, rungs: dict | None = None,
               n_jobs: int = 1) -> dict:
    """{rung: (model, fit info)} of `mode_l.fit_final` for every rung of `rungs` (default `mode_l_models.default_rungs()`),
    pickled as `mode_l_<rung>_r2.pkl`. Lockbox rows in `cat` are dropped and a lockbox window in `w` raises LockboxError."""
    rungs = rungs or mlm.default_rungs()
    lock = set(cat.loc[cat["source"].isin(data.HOLDOUT_SOURCES), "cell_id"]) if "source" in cat.columns else set()
    cat = cat[~cat["source"].isin(data.HOLDOUT_SOURCES)] if "source" in cat.columns else cat
    check_training_only(pd.DataFrame({"config_key": []}), None, w, lock)
    res = Parallel(n_jobs=n_jobs)(delayed(mode_l.fit_final)(w, cat, qc, n, seed, rungs, SUFFIX) for n in rungs)
    return dict(zip(rungs, res))


def family_of_model(model: str) -> str:
    return "YOLO" if str(model).startswith("yolo") else "non-YOLO"


def summarise_mode_l(cells: pd.DataFrame) -> pd.DataFrame:
    """Per family and rung: n_cells, the median whole-cell, inference-segment (`forward` phase) and 1 s block energy APE, the
    pooled forward-window MAE (W) and forward / all-window R^2 (`mode_l.pooled_window_metrics`)."""
    c = cells.assign(family=cells["model"].map(family_of_model))
    rows = []
    for (fam, rung), g in c.groupby(["family", "rung"], sort=False):
        pw = mode_l.pooled_window_metrics(g)
        rows.append({"family": fam, "rung": rung, "n_cells": len(g), "whole": float(g["ape"].median()),
                     "inference": float(g["ape_meas"].median()), "block_1s": float(g["ape_1s"].median()),
                     "fwd_mae_W": pw["mae_fwd_W"], "fwd_r2": pw["r2_fwd"], "all_r2": pw["r2_all"]})
    return pd.DataFrame(rows)


def r1_mode_l(frozen_cells: pd.DataFrame) -> pd.DataFrame:
    """Per family and rung the frozen R1 medians (whole, inference, 1 s block), as `summarise_mode_l` names them."""
    c = frozen_cells.assign(family=frozen_cells["model"].map(family_of_model))
    g = c.groupby(["family", "rung"], sort=False)[["ape", "ape_meas", "ape_1s"]].median()
    return g.rename(columns={"ape": "R1 whole", "ape_meas": "R1 inference", "ape_1s": "R1 1 s block"}).reset_index()


# ------------------------------------------------------------------ report

def pct(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Copy of `df` with the fractions of `cols` as percent."""
    return df.assign(**{c: df[c] * 100 for c in cols if c in df.columns})


def tab(df: pd.DataFrame, digits: int = 2) -> str:
    """Markdown table; floats to `digits` decimals (percent columns are made percent by `pct` first)."""
    if df is None or len(df) == 0:
        return "_No data._\n"
    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v).replace("|", "\\|")
    return ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
            + "".join("| " + " | ".join(f(v) for v in row) + " |\n" for row in df.itertuples(index=False)))


A_COLS = ["mdape", "mape", "maxape", "inside_90", "bias_pct"]


def _a_view(s: pd.DataFrame, label: str) -> pd.DataFrame:
    return pct(s, A_COLS).assign(revision=label)


def render(*, prov: dict, a_sum: pd.DataFrame, a_scores: pd.DataFrame, r1_sum: pd.DataFrame | None, calc_cmp: pd.DataFrame,
           calc: pd.DataFrame, l_sum: pd.DataFrame, l_r1: pd.DataFrame | None, l_cells: pd.DataFrame,
           probe: pd.DataFrame | None) -> str:
    """reports/r2/lockbox_posthoc.md: summary tables first, per-configuration and per-cell rows after."""
    out = [f"# Lockbox, scored again with the R2 models ({LABEL})\n",
           f"**{LABEL}.** The frozen R1 lockbox result (`reports/lockbox.md`, scored once) is unchanged and is only read here, for "
           "the R1 columns. Every R2 model below was fitted on the training cells alone; the lockbox configurations enter only as "
           "inputs to `predict` (Mode A) or as scored windows (Mode L). The R2 methods were chosen after the lockbox was seen, "
           "so these numbers are not a pre-registered test. Percent columns are percent; MdAPE is the median, MAPE the mean and "
           "MaxAPE the largest of |pred - truth| / truth over the lockbox configurations (5; 2 of a new model, yolo12n, and 3 of "
           "seen models) or cells; width is the median hi90 / lo90.\n",
           "## Provenance\n"] + [f"- {x}" for x in prov["lines"]] + [""]
    out += ["## Mode A: AUTO and M4Physics, R2 beside R1 (post hoc (R2))\n",
            "R1 is the frozen primary (AUTO-final: M1Ridge for all three targets) and the frozen secondaries from `lockbox.md`. "
            "R2 AUTO is the candidate with the lowest balanced inner error in the regime of the configuration; the regime of "
            "each configuration is in the per-configuration table.\n"]
    head = a_sum[a_sum["candidate"].isin([mode_a.AUTO, "M4Physics"])]
    parts = [_a_view(head, "R2")]
    if r1_sum is not None and len(r1_sum):
        parts.append(_a_view(r1_sum[r1_sum["candidate"].str.endswith("(primary)")], "R1"))
    h = pd.concat(parts, ignore_index=True)
    h["gk"] = h["group"].map({g: i for i, g in enumerate(GROUPS)})
    h["tk"] = h["target"].map({t: i for i, t in enumerate(TARGETS)})
    h = h.sort_values(["tk", "gk", "revision"], kind="stable")
    out += [tab(h[["target", "group", "revision", "candidate", "n", "mdape", "mape", "maxape", "inside_90", "width"]]),
            "## Mode A: every R2 candidate, all five configurations\n",
            tab(_a_view(a_sum[a_sum["group"] == ALL], "R2")[["target", "candidate", "n", "mdape", "mape", "maxape", "inside_90", "width"]]),
            "## Mode A: energy, every R2 candidate by group\n",
            tab(_a_view(a_sum[(a_sum["target"] == "energy_mJ") & (a_sum["group"] != ALL)], "R2")[
                ["group", "candidate", "n", "mdape", "mape", "maxape", "inside_90", "width"]])]
    out += ["## Deployment calculator, serve-phase energy of the 15 lockbox cells (APE %)\n",
            "R2 uses the R2 AUTO energy and latency predictions; R1 the frozen predictions (`reports/r2/lockbox_calculator_gap.md`). "
            "`measured duration` leaks the measured serve duration; the two gap scenarios do not use it.\n",
            tab(pct(calc_cmp, [c for c in calc_cmp.columns if c.startswith(("R1", "R2"))]))]
    out += ["## Mode L by family (post hoc (R2))\n",
            f"R2 finals fitted on QC-primary training YOLO counter cells (slip-corrected windows, CMOS features); {prov['n_l_cells']} "
            "lockbox cells. YOLO = yolo12n and yolov8n (9 cells), non-YOLO = resnet18 and mbv3 (6 cells, outside the training "
            "families). Whole = whole cell, inference = the `forward` phase, 1 s block = median over 10-window blocks, all as median "
            "energy APE in percent over cells; forward MAE (W) and R^2 are over the pooled 100 ms forward windows (the frozen R1 run "
            "did not record window sums, so those are R2 only).\n"]
    ls_ = l_sum.copy()
    if l_r1 is not None:
        ls_ = ls_.merge(l_r1, on=["family", "rung"], how="left")
    ls_ = pct(ls_, ["whole", "inference", "block_1s", "R1 whole", "R1 inference", "R1 1 s block"])
    ls_["rk"] = ls_["rung"].map({r: i for i, r in enumerate(mlm.default_rungs())})
    ls_["fk"] = ls_["family"].map({f: i for i, f in enumerate(FAMILIES)})
    out += [tab(ls_.sort_values(["fk", "rk"]).drop(columns=["rk", "fk"]), 3)]
    if probe is not None and len(probe):
        out += ["## Probe: Mode L power x measured latency of another repeat, against Mode A R2 AUTO\n",
                "Median APE (percent) of the energy per image of repeat k, from the mean predicted power over the measured phase of "
                "repeat j times j's latency (ordered pairs of one configuration), per rung, and the median APE of the R2 AUTO "
                "prediction on the same pairs; R1 is the frozen probe against the frozen primary.\n",
                tab(probe)]
    out += ["## Per configuration (Mode A, AUTO, M4Physics and R1 primary)\n"]
    show = a_scores[a_scores["candidate"].isin([mode_a.AUTO, "M4Physics"])].copy()
    show["candidate"] = np.where(show["candidate"] == mode_a.AUTO, "AUTO=" + show["chosen"], show["candidate"])
    show = pct(show, ["ape"]).assign(inside=show["inside"])
    show["tk"] = show["target"].map({t: i for i, t in enumerate(TARGETS)})
    out += [tab(show.sort_values(["tk", "candidate", "model", "imgsz"])[
        ["target", "model", "imgsz", "regime", "candidate", "pred", "lo90", "hi90", "truth", "ape", "inside"]], 3)]
    out += ["## Per cell (calculator)\n",
            tab(pct(calc[["cell_id", "model", "imgsz", "measured_J", "predicted_J_known", "ape_known", "predicted_J_oh0", "ape_oh0",
                          "predicted_J_est", "ape_est"]], ["ape_known", "ape_oh0", "ape_est"]), 3)]
    lc = l_cells[l_cells["rung"].isin(HEADLINE_RUNGS)]
    out += [f"## Per cell (Mode L, rungs {', '.join(HEADLINE_RUNGS)}; the other rungs are in the CSV)\n",
            tab(pct(lc[["cell_id", "rung", "E_true_J", "E_pred_J", "ape", "ape_meas", "ape_1s", "mae_fwd_W"]],
                    ["ape", "ape_meas", "ape_1s"]), 3)]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ the run

def probe_compare(cells: pd.DataFrame, der: pd.DataFrame, cat_l: pd.DataFrame, auto: pd.DataFrame,
                  r1_probe: pd.DataFrame | None) -> pd.DataFrame:
    """`lockbox_score.probe_scores` on the R2 rung scores and R2 AUTO energy, as median APE in percent per rung, with the R1
    frozen probe beside it (rungs of the same name)."""
    p = ls.summarise_probe(ls.probe_scores(cells, der, cat_l, auto, None))
    p = pct(p.rename(columns={"median_probe_ape": "probe APE", "median_mode_a_ape": "Mode A R2 AUTO APE"}),
            ["probe APE", "Mode A R2 AUTO APE"])
    if r1_probe is not None and len(r1_probe):
        r1 = ls.summarise_probe(r1_probe).rename(columns={"n_pairs": "R1 n_pairs", "median_probe_ape": "R1 probe APE",
                                                          "median_mode_a_ape": "R1 Mode A APE"})
        p = p.merge(pct(r1, ["R1 probe APE", "R1 Mode A APE"]), on="rung", how="left")
    return p


def run_posthoc(paths: ri.Paths, seed: int, *, cat: pd.DataFrame, der: pd.DataFrame, qc_train: pd.DataFrame,
                qc_l: pd.DataFrame, t: pd.DataFrame, desc_train: pd.DataFrame, w_train: pd.DataFrame, static_dir,
                w_lockbox: pd.DataFrame | None = None, slip: dict | None = None, grids: dict | None = None,
                rungs: dict | None = None, n_jobs: int = 1, probe: bool = True, fit=None) -> dict:
    """Fit the R2 finals on training data only, score the lockbox and write reports/r2/lockbox_posthoc.md and its CSVs under
    `paths.out_dir`/r2, plus one runlog line (command `lockbox-posthoc`). `t` is the training configuration table, `desc_train`
    the training descriptors, `w_train` the training windows (`windows.build_windows`), `w_lockbox` the lockbox windows
    (`lockbox_windows_r2`; computed from the catalogue when None). Nothing under derived/lockbox/ or reports/lockbox* is written."""
    cat_l = ls.require_lockbox(cat)
    cat_t = cat[~cat["source"].isin(data.HOLDOUT_SOURCES)]
    recs = ri.read_records(paths.runs_log) if Path(paths.runs_log).is_file() else []
    check_training_only(t, None, w_train, cat_l["cell_id"])
    pairs = sorted({(r.base_model, int(r.imgsz)) for r in cat_l.itertuples()})
    feats, shas = lb.feature_rows(pairs, static_dir)
    truth = ls.truth_table(cat_l, der, qc_l)
    truth_ok = check_truth_matches_frozen(truth, frozen_frame(recs, "truth"))
    feats = feats[feats["config_key"].isin(truth["config_key"])].reset_index(drop=True)
    desc = lb.descriptor(desc_train, feats)
    # Mode A
    pred = mode_a_predictions(t, feats, desc, seed, grids, fit=fit)
    a_scores = score_mode_a(pred, truth)
    a_sum = summarise(a_scores)
    fro = frozen_frame(recs, "mode_a_scores")
    r1_sum = summarise(r1_mode_a(fro)) if fro is not None else None
    # calculator
    nov = lb.novelty_by_model(desc, feats["base_model"], set(t["base_model"]))
    auto = auto_primary(pred, nov)
    idle = deploy.idle_table(cat_t, der, qc_train)
    train_cells = deploy.cell_table(cat_t, der, qc_train)
    calc = ls.calculator_posthoc_gap(cat_l, der, qc_l, auto, idle, train_cells=train_cells)
    r1_gap_path = Path(paths.out_dir) / "r2" / "lockbox_calculator_gap.csv"
    r1_gap = pd.read_csv(r1_gap_path) if r1_gap_path.is_file() else None
    calc_cmp = compare_calc(calc, r1_gap)
    # Mode L
    if w_lockbox is None:
        w_lockbox, slip = lockbox_windows_r2(cat_l, qc_l)
    fits = fit_mode_l(w_train, cat_t, qc_train, seed, rungs, n_jobs)
    w_lockbox = w_lockbox[w_lockbox["cell_id"].isin(qc_l.loc[qc_l["primary"].astype(bool), "cell_id"])] if len(w_lockbox) else w_lockbox
    l_cells = (ls.mode_l_cells(w_lockbox, cat_l, {n: m for n, (m, _) in fits.items()}) if len(w_lockbox) else
               pd.DataFrame(columns=["cell_id", "rung", "model", "ape"]))
    l_cells = l_cells.assign(family=l_cells["model"].map(family_of_model))
    l_sum = summarise_mode_l(l_cells) if len(l_cells) else pd.DataFrame()
    fro_l = frozen_frame(recs, "mode_l_cells")
    l_r1 = r1_mode_l(fro_l) if fro_l is not None else None
    pr = probe_compare(l_cells, der, cat_l, auto, frozen_frame(recs, "probe")) if probe and len(l_cells) else None
    # write
    run_id = uuid.uuid4().hex
    out_dir = Path(paths.out_dir) / "r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    a_scores.to_csv(out_dir / "lockbox_posthoc_mode_a.csv", index=False)
    a_sum.to_csv(out_dir / "lockbox_posthoc_mode_a_summary.csv", index=False)
    l_cells.to_csv(out_dir / "lockbox_posthoc_mode_l.csv", index=False)
    l_sum.to_csv(out_dir / "lockbox_posthoc_mode_l_summary.csv", index=False)
    calc.to_csv(out_dir / "lockbox_posthoc_calc.csv", index=False)
    choices = (a_scores[a_scores["candidate"] == mode_a.AUTO].drop_duplicates(["target", "regime"])
               .sort_values(["target", "regime"]))
    lines = [f"run `{run_id}`; seed {seed}; training configurations {len(t)} (base models: {', '.join(sorted(set(t['base_model'])))}), "
             f"training cells {len(cat_t)}; lockbox cells {len(cat_l)} ({len(qc_l[qc_l['primary'].astype(bool)])} QC-primary)",
             "regimes: " + "; ".join(f"{r.model}:{r.imgsz} {r.regime}" for r in
                                     a_scores.drop_duplicates("config_key").itertuples()),
             "AUTO chose (balanced inner error): " + "; ".join(f"{r.target}/{r.regime}: {r.chosen}" for r in choices.itertuples()),
             "truth equals the frozen truth table (derived/lockbox/<run>_truth.parquet): "
             + {True: "yes", None: "not checked (no frozen table)"}[truth_ok],
             "lockbox cells with a slip correction in the windows: "
             + (", ".join(sorted(c for c, s in (slip or {}).items() if s == "corrected")) or "none"),
             "Mode L finals (pickle, sha256): " + "; ".join(f"{n} `{Path(i['path']).name}` `{i['sha256'][:12]}`" for n, (_, i) in fits.items()),
             f"R1 columns read from the frozen run `{frozen_run_id(recs)}`" if frozen_run_id(recs) else "no frozen R1 run found: R1 columns omitted"]
    text = render(prov={"lines": lines, "n_l_cells": int(l_cells["cell_id"].nunique()) if len(l_cells) else 0}, a_sum=a_sum,
                  a_scores=a_scores, r1_sum=r1_sum, calc_cmp=calc_cmp, calc=calc, l_sum=l_sum, l_r1=l_r1, l_cells=l_cells,
                  probe=pr)
    md_path = out_dir / "lockbox_posthoc.md"
    md_path.write_text(text)
    key = a_sum[(a_sum["candidate"] == mode_a.AUTO) & (a_sum["target"] == "energy_mJ")]
    rec = runlog.append({"run_id": run_id, "command": "lockbox-posthoc", "label": LABEL, "seed": seed,
                         "n_train_configs": len(t), "truth_matches_frozen": truth_ok,
                         "mode_a_auto_energy": {r.group: {"mdape": r.mdape, "inside_90": r.inside_90, "width": r.width}
                                                for r in key.itertuples()},
                         "calculator_median_ape": dict(zip(calc_cmp["variant"], calc_cmp["R2 median"])),
                         "mode_l_models": {n: {"path": i["path"], "sha256": i["sha256"], "params": i["params"]}
                                           for n, (_, i) in fits.items()},
                         "outputs": [str(out_dir / n) for n in ("lockbox_posthoc.md", "lockbox_posthoc_mode_a.csv",
                                                                "lockbox_posthoc_mode_l.csv", "lockbox_posthoc_calc.csv")]},
                        paths.runs_log)
    return {"run_id": run_id, "mode_a": a_scores, "mode_a_summary": a_sum, "calc": calc, "calc_compare": calc_cmp,
            "mode_l": l_cells, "mode_l_summary": l_sum, "probe": pr, "md": md_path, "rec": rec, "fits": fits,
            "r1_mode_a_summary": r1_sum}
