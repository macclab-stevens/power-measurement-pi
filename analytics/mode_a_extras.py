"""Mode A extras for the report: the shipped final models, the learning curve and the v2-only against
pooled comparison. Reads the latest S-LOMO runs (to see which candidate AUTO chose most often) and the
configuration table; writes parquets under `derived_dir()/mode_a_extras/` and one runlog line per output."""
import json
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd

import mode_a
import report_inputs as ri
import runlog
import splits

ORDER = ["M1Ridge", "M2Trees", "M3Factored"]
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
    """Run the three extras. Returns {"fits": {target: dict}, "learning_curve": run_id, "v2_vs_pooled": {...}}."""
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
                             "params": info["params"], "path": info["path"], "sha256": info["sha256"], "auto_counts": counts,
                             "source_run_id": ref.run_id, "n_configs": len(t)}, paths.runs_log)
        summary["fits"][target] = rec
    cand, info = fits["energy_mJ"]
    lc = mode_a.learning_curve(t, "energy_mJ", seed, cand, info["params"], desc, max_subsets=max_subsets)
    rid = uuid.uuid4().hex
    lc.to_parquet(out_dir / f"{rid}.parquet", index=False)
    runlog.append({"run_id": rid, "command": "mode-a-extras", "part": "learning_curve", "seed": seed, "target": "energy_mJ",
                   "candidate": cand, "params": info["params"], "max_subsets": max_subsets, "n_rows": len(lc)}, paths.runs_log)
    summary["learning_curve"] = rid
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
                 "params": info["params"], "path": info["path"], "sha256": info["sha256"], "n_configs": len(t)}, paths.runs_log)
    return out
