"""Mode A extras for the report: the shipped final models, the v2-only against pooled comparison and, as a
separate step (`run_learning_curve`), the learning curve. Reads the latest S-LOMO runs (to see which candidate
AUTO chose most often, and each outer fold's hyperparameters) and the configuration table; writes parquets under
`derived_dir()/mode_a_extras/` and one runlog line per output."""
import itertools
import json
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate

import mode_a
import report_inputs as ri
import runlog
import splits

ORDER = ["M1Ridge", "M2Trees", "M3Factored", "M4Physics"]
LC_COMMAND = "learning-curve"
YOLO_RIG = "nov2026"


def auto_counts(pred: pd.DataFrame) -> dict:
    """How often AUTO chose each candidate, one vote per outer fold."""
    a = pred[pred["candidate"] == mode_a.AUTO].drop_duplicates("fold")
    return dict(Counter(json.loads(p)["candidate"] for p in a["params"]))


def auto_choice(pred: pd.DataFrame) -> str:
    """AUTO's most often chosen candidate; ties go to M1, M2, M3 in that order."""
    counts = auto_counts(pred)
    if not counts:
        raise ValueError("no AUTO rows in the predictions")
    return max(counts, key=lambda c: (counts[c], -ORDER.index(c) if c in ORDER else -99))


def latest_slomo(paths: ri.Paths, recs: list, target: str, seed: int) -> ri.RunRef:
    """Latest S-LOMO run of `target`, preferring the given seed."""
    for sd in (seed, None):
        match = {"command": "mode-a", "split": "S-LOMO", "target": target, **({"seed": sd} if sd is not None else {})}
        rec = ri.latest(recs, paths.mode_a_dir, **match)
        if rec is not None:
            return ri.ref_of(rec, paths.mode_a_dir)
    raise ri.ReportInputError(f"no S-LOMO Mode A run for target {target}; run `{ri.MODE_A_CMD} --split S-LOMO --target {target}`")


def _yolo_folds(t: pd.DataFrame, seed: int) -> list:
    return [f for f in splits.make_splits(t, seed)["S-LOMO"] if f.fold in splits.YOLO]


def run(t: pd.DataFrame, desc: pd.DataFrame, seed: int, paths: ri.Paths, *, n_jobs: int = 1, grids: dict | None = None,
        max_subsets: int = 10, targets=None) -> dict:
    """Final fits per target and the v2-only against pooled runs. Returns {"fits": {target: dict}, "v2_vs_pooled": {...}}.
    The learning curve is `run_learning_curve`, which can be re-run without these (guarded) fits."""
    recs = ri.read_records(paths.runs_log)
    out_dir = Path(paths.extras_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits, summary = {}, {"fits": {}}
    for target in (targets or ri.TARGETS):
        ref = latest_slomo(paths, recs, target, seed)
        counts = auto_counts(ref.df)
        cand = auto_choice(ref.df)
        _, info = mode_a.fit_final(t, cand, target, seed, desc, grids)
        fits[target] = (cand, info)
        rec = runlog.append({"command": "mode-a-extras", "part": "fit_final", "seed": seed, "target": target, "candidate": cand,
                             "params": info["params"], "regime": info["regime"], "path": info["path"], "sha256": info["sha256"], "auto_counts": counts,
                             "source_run_id": ref.run_id, "n_configs": len(t)}, paths.runs_log)
        summary["fits"][target] = rec
    summary["v2_vs_pooled"] = {}
    for training, table in (("pooled", t), ("v2only", t[t["rig"] == YOLO_RIG].reset_index(drop=True))):
        folds = _yolo_folds(table, seed)
        res = mode_a.run_split(table, folds, "energy_mJ", seed, desc, n_jobs=n_jobs, grids=grids)
        rid = uuid.uuid4().hex
        res.to_parquet(out_dir / f"{rid}.parquet", index=False)
        runlog.append({"run_id": rid, "command": "mode-a-extras", "part": "v2_vs_pooled", "training": training, "split": "S-LOMO",
                       "target": "energy_mJ", "seed": seed, "n_configs_train_table": len(table), "n_folds": len(folds)}, paths.runs_log)
        summary["v2_vs_pooled"][training] = rid
    return summary


def slomo_params(pred: pd.DataFrame, candidate: str) -> dict:
    """{held-out base model: the `candidate` hyperparameters chosen in that model's S-LOMO outer fold}, read from the
    `params` column of the candidate's own rows (S-LOMO folds are named by their held-out base model)."""
    rows = pred[pred["candidate"] == candidate].drop_duplicates("fold")
    if rows.empty:
        raise ValueError(f"no {candidate} rows in the S-LOMO predictions")
    return {r.fold: json.loads(r.params) for r in rows.itertuples()}


def learning_curve_per_fold(t: pd.DataFrame, target: str, seed: int, candidate: str, params_by_heldout: dict,
                            desc: pd.DataFrame, max_subsets: int = 10) -> pd.DataFrame:
    """The learning curve of `mode_a.learning_curve` (each YOLO base model held out in turn, trained on k = 2..5 of the
    others plus every non-YOLO configuration, the same subset draws), but each held-out model is fitted with the
    hyperparameters its own S-LOMO outer fold chose (`params_by_heldout`), never with parameters that saw it. No
    inner search. Raises ValueError if a held-out YOLO model has no entry."""
    mode_a._check_table(t)
    rng = np.random.default_rng(seed)
    yolo = [m for m in splits.YOLO if m in set(t["base_model"])]
    missing = [m for m in yolo if m not in params_by_heldout]
    if missing:
        raise ValueError(f"no outer-fold params for held-out models {missing}")
    non_yolo = t[~t["base_model"].isin(splits.YOLO)]
    rows = []
    for held in yolo:
        test, others = t[t["base_model"] == held], [m for m in yolo if m != held]
        for k in range(2, 6):
            if k > len(others):
                continue
            combos = list(itertools.combinations(others, k))
            if len(combos) > max_subsets:
                combos = [combos[i] for i in sorted(rng.choice(len(combos), size=max_subsets, replace=False))]
            for sub in combos:
                train = pd.concat([t[t["base_model"].isin(sub)], non_yolo], ignore_index=True)
                yhat = mode_a._make(candidate, desc, seed).fit(train, target, params_by_heldout[held]).predict(test)
                rows.append({"heldout": held, "k": k, "subset": ";".join(sub),
                             "mdape": evaluate.metrics(test[target].to_numpy(float), yhat)["mdape"]})
    return pd.DataFrame(rows, columns=["heldout", "k", "subset", "mdape"])


def run_learning_curve(t: pd.DataFrame, desc: pd.DataFrame, seed: int, paths: ri.Paths, *, max_subsets: int = 10,
                       target: str = "energy_mJ") -> dict:
    """Energy learning curve of the candidate the final fit chose (`mode-a-extras` line of `seed`), each held-out YOLO
    model with its own S-LOMO outer-fold hyperparameters from the latest S-LOMO energy run of `seed`; saved as a parquet
    and logged as command `learning-curve` with the source run id and the parameters used. Fits no frozen model."""
    recs = ri.read_records(paths.runs_log)
    fit = ri.latest_record(recs, command="mode-a-extras", part="fit_final", target=target, seed=seed)
    if fit is None:
        raise ri.ReportInputError(f"no final-model fit for target {target} with seed {seed}; run `uv run python cli.py mode-a-extras --seed {seed}`")
    ref = latest_slomo(paths, recs, target, seed)
    cand = fit["candidate"]
    params = {h: p for h, p in slomo_params(ref.df, cand).items() if h in splits.YOLO}
    lc = learning_curve_per_fold(t, target, seed, cand, params, desc, max_subsets)
    out_dir = Path(paths.extras_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex
    lc.to_parquet(out_dir / f"{rid}.parquet", index=False)
    runlog.append({"run_id": rid, "command": LC_COMMAND, "part": "learning_curve", "seed": seed, "target": target, "candidate": cand,
                   "params_by_heldout": params, "params_source_run_id": ref.run_id, "final_fit_run_id": fit["run_id"],
                   "max_subsets": max_subsets, "n_rows": len(lc), "n_configs": len(t)}, paths.runs_log)
    return {"run_id": rid, "n_rows": len(lc), "candidate": cand, "params_source_run_id": ref.run_id, "params_by_heldout": params}


SECONDARY = {"energy_mJ": ["B3LogLinear", "M3Factored"], "latency_ms": ["B3LogLinear"], "mean_P_W": ["B3LogLinear"]}


def run_secondary(t: pd.DataFrame, desc: pd.DataFrame, seed: int, paths: ri.Paths, *, grids: dict | None = None,
                  targets=None) -> dict:
    """Final fits of the pre-declared secondary candidates (B3 for every target, M3 for energy) on every
    configuration of `t`, pickled by `mode_a.fit_final` and logged as command `final-mode-a-secondary`.
    A candidate that is already the AUTO-final model of that target (per the latest `mode-a-extras` fit
    of this seed) is not refitted: its pickle is that model and re-pickling could change its sha256.
    Returns {(target, candidate): runlog record}."""
    recs = ri.read_records(paths.runs_log)
    out = {}
    for target in (targets or ri.TARGETS):
        primary = ri.latest_record(recs, command="mode-a-extras", part="fit_final", target=target, seed=seed)
        for cand in SECONDARY[target]:
            if primary is not None and primary["candidate"] == cand:
                continue
            _, info = mode_a.fit_final(t, cand, target, seed, desc, grids)
            out[(target, cand)] = runlog.append(
                {"command": "final-mode-a-secondary", "part": "fit_final", "seed": seed, "target": target, "candidate": cand,
                 "params": info["params"], "regime": info["regime"], "path": info["path"], "sha256": info["sha256"], "n_configs": len(t)}, paths.runs_log)
    return out
