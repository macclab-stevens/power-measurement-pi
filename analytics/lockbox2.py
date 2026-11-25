"""Lockbox 2 (R3): predictions frozen and committed before any lockbox-2 cell is measured, then scored once.

The first lockbox (reports/lockbox.md) was seen when R2 was designed, so R2 has no confirmatory test. Lockbox 2 is
that test for R3, the revision trained on v1 + v2 + v3 (v3 adds the hooks on / off calibration pairs and attention
models). Its five configurations are models never collected, each similar to a training model (the stated goal:
"a model like one we have seen"): resnet34 (resnet18), mobilenet_v3_large (mbv3), efficientnet_b1 (effb0), swin_s
(swin_t) and yolo12m (yolo12n / yolo12s), all hooks OFF (deployment-like), 2400 MHz, 4 threads, fp32, batch 1.

`freeze` (CLI `lockbox2-freeze`) refuses once any lockbox-2 cell exists or the predictions file exists. On the training
data only it fits every Mode A R3 candidate and AUTO in the new-model regime (`mode_a.fit_candidates`), predicts the five
configurations with fold-weighted 90 % intervals, fits every Mode L rung (`mode_l.fit_final`, suffix `_r3`), pickles all
models under derived/models/lockbox2_r3/, and writes reports/r3/lockbox2_predictions.csv and reports/r3/lockbox2_frozen.json
(every pickle's sha256, the predictions' sha256, the code pin, the static CSVs' sha256 and the freeze time). Commit both
and tag the commit before measuring.

`score` (CLI `lockbox2-score`) refuses when reports/r3/lockbox2.md exists (scored once), when a lockbox-2 cell started
before the freeze time, or when a pinned hash differs; a code change is refused unless `allow_code_change`, which labels
the result exploratory. It fits nothing: it loads the frozen pickles and the predictions file.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
import deploy
import evaluate as ev
import lockbox as lb
import lockbox_checks as checks
import lockbox_score as ls
import mode_a
import mode_a_models as mm
import mode_l
import mode_l_models as mlm
import qc as qcm
import report_inputs as ri
import runlog
import settings
import static
import tables

SOURCE = "lockbox2"
CONFIGS = [("resnet34", 224), ("mbv3l", 224), ("effb1", 224), ("swins", 224), ("yolo12m", 640)]
SIMILAR = {"resnet34": "resnet18", "mbv3l": "mbv3", "effb1": "effb0", "swins": "swint", "yolo12m": "yolo12s"}
FIXED = {"precision": "fp32", "batch": 1, "threads": 4, "cpu_mhz": 2400, "rig": "nov2026", "layer_hooks": "off"}
REGIME = "new-model"
SUFFIX = "_r3"
PRED_COLS = ["config_key", "model", "imgsz", "candidate", "role", "chosen", "target", "pred", "lo90", "hi90", "novelty",
             "threshold", "flagged", "model_sha256", "static_csv_sha256"]


def out_dir(paths: ri.Paths) -> Path:
    return Path(paths.out_dir) / "r3"


def pred_path(paths: ri.Paths) -> Path:
    return out_dir(paths) / "lockbox2_predictions.csv"


def manifest_path(paths: ri.Paths) -> Path:
    return out_dir(paths) / "lockbox2_frozen.json"


def report_path(paths: ri.Paths) -> Path:
    return out_dir(paths) / "lockbox2.md"


def static_dir_default() -> Path:
    return settings.HERE / "lockbox2_static"


def cells_exist() -> bool:
    d = settings.data_root() / SOURCE / "cells"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def feature_rows(static_dir=None, configs=CONFIGS) -> tuple:
    """(Mode A feature rows of the lockbox-2 configurations, {(model, imgsz): static CSV sha256}). Static features from
    `static_dir`/<model>_<imgsz>.csv (tools/static_features.py), whole-model MACs from static_flops.csv, hooks off."""
    d = Path(static_dir) if static_dir else static_dir_default()
    rows, shas = [], {}
    for model, imgsz in configs:
        p = checks.static_csv_path(d, model, imgsz)
        if not p.is_file():
            raise lb.LockboxError(f"static features file {p} is missing (tools/static_features.py)")
        rows.append({"base_model": model, "variant": model, "imgsz": imgsz, "quant": "fp32", "n_quantized_modules": 0,
                     **FIXED, **static.features_from_layer_table(pd.read_csv(p))})
        shas[(model, imgsz)] = runlog.file_hash(p)
    st = static.add_flops(pd.DataFrame(rows).set_index(["base_model", "imgsz"])).reset_index()
    t = tables.add_derived_columns(st)
    t.insert(0, "config_key", [tables.config_key(r) for _, r in t.iterrows()])
    return t, shas


def check_training(t: pd.DataFrame, w: pd.DataFrame | None) -> None:
    """LockboxError if the training table or windows hold a holdout cell (any `lockbox*` cell id)."""
    ids = set(";".join(t["cell_ids"].astype(str)).split(";")) if "cell_ids" in t.columns else set()
    bad = sorted(i for i in ids if i.startswith("lockbox"))
    if w is not None and len(w):
        bad += sorted(set(c for c in w["cell_id"].astype(str) if c.startswith("lockbox")))
    if bad:
        raise lb.LockboxError(f"holdout cells in the training data: {bad[:3]}")


def _pickle(obj, name: str) -> dict:
    d = settings.derived_dir() / "models" / "lockbox2_r3"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pkl"
    p.write_bytes(pickle.dumps(obj))
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def predict_mode_a(t: pd.DataFrame, feats: pd.DataFrame, shas: dict, desc_train: pd.DataFrame, seed: int,
                   threshold: float) -> tuple:
    """(prediction rows, {target/candidate: pickle record}) of every R3 candidate and AUTO, fitted on `t` only."""
    desc = lb.descriptor(desc_train, feats)
    nov = lb.novelty_by_model(desc, feats["base_model"], set(t["base_model"]))
    rows, frozen = [], {}
    for target in ri.TARGETS:
        # `desc` = training descriptors plus the lockbox-2 models' (B2NearestModel looks a test model up in it; every
        # standardisation uses the training models only, so this adds no information from the lockbox)
        fits = mode_a.fit_candidates(t, target, seed, desc, REGIME)
        for cand, f in fits.items():
            rec = _pickle(f["model"], f"mode_a_{target}_{cand}")
            frozen[f"{target}/{cand}"] = {**rec, "params": f["params"], "chosen": f["chosen"], "score": f["score"]}
            pred = np.asarray(f["model"].predict(feats), float)
            for i, r in enumerate(feats.itertuples()):
                n = nov[r.base_model]
                rows.append({"config_key": r.config_key, "model": r.base_model, "imgsz": int(r.imgsz), "candidate": cand,
                             "role": "primary" if cand == mode_a.AUTO else "secondary", "chosen": f["chosen"],
                             "target": target, "pred": float(pred[i]), "lo90": float(pred[i] * np.exp(f["q5"])),
                             "hi90": float(pred[i] * np.exp(f["q95"])), "novelty": n, "threshold": threshold,
                             "flagged": bool(n > threshold), "model_sha256": rec["sha256"],
                             "static_csv_sha256": shas[(r.base_model, int(r.imgsz))]})
    return pd.DataFrame(rows, columns=PRED_COLS), frozen


def freeze(paths: ri.Paths, seed: int, *, t: pd.DataFrame, desc_train: pd.DataFrame, w_train: pd.DataFrame,
           cat: pd.DataFrame, qc: pd.DataFrame, threshold: float, static_dir=None, n_jobs: int = 2) -> dict:
    """Write the frozen lockbox-2 predictions and manifest (see the module docstring). `t`: the R3 training table
    (reports/mode_a_configs.csv), `desc_train` its descriptors, `w_train` the training windows, `cat` the catalogue,
    `qc` the training QC table, `threshold` the S-LOMO applicability threshold of the R3 AUTO run."""
    if cells_exist():
        raise lb.LockboxError("lockbox-2 cells already exist: the predictions must be frozen and committed first")
    out = pred_path(paths)
    if out.exists():
        raise lb.LockboxError(f"{out} exists: it is the committed record of the frozen predictions")
    check_training(t, w_train)
    feats, shas = feature_rows(static_dir)
    pred, frozen_a = predict_mode_a(t, feats, shas, desc_train, seed, threshold)
    cat_t = cat[~cat["source"].isin(data.HOLDOUT_SOURCES)]
    from joblib import Parallel, delayed
    rungs = mlm.default_rungs()
    res = Parallel(n_jobs=n_jobs)(delayed(mode_l.fit_final)(w_train, cat_t, qc, n, seed, rungs, SUFFIX) for n in rungs)
    frozen_l = {n: {"path": info["path"], "sha256": info["sha256"], "params": info["params"], "n_cells": info["n_cells"],
                    "base_models": info["base_models"]} for n, (_, info) in zip(rungs, res)}
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    man = {"seed": seed, "frozen_at_ns": time.time_ns(), "regime": REGIME, "configs": [f"{m}:{s}" for m, s in CONFIGS],
           "similar_to": SIMILAR, "fixed": FIXED, "predictions_csv": out.name, "predictions_sha256": runlog.file_hash(out),
           "mode_a": frozen_a, "mode_l": frozen_l, "static_csv_sha256": {f"{m}:{s}": h for (m, s), h in shas.items()},
           "static_flops_sha256": runlog.file_hash(static.FLOPS_CSV), "threshold": threshold,
           "training": {"n_configs": len(t), "base_models": sorted(set(t["base_model"])),
                       "table_sha256": hashlib.sha256(pd.util.hash_pandas_object(t["config_key"]).values.tobytes()).hexdigest()},
           "code": checks.code_pin()}
    mp = manifest_path(paths)
    mp.write_text(json.dumps(man, indent=2, sort_keys=True, default=str) + "\n")
    rec = runlog.append({"command": "lockbox2-freeze", "seed": seed, "predictions_sha256": man["predictions_sha256"],
                         "manifest_sha256": runlog.file_hash(mp), "n_rows": len(pred),
                         "auto_chosen": {k: v["chosen"] for k, v in frozen_a.items() if k.endswith("/AUTO")}}, paths.runs_log)
    return {"pred": pred, "manifest": man, "path": out, "rec": rec}


# ------------------------------------------------------------------ scoring (once)

def load_frozen(man: dict) -> tuple:
    """({target/candidate: model}, {rung: model}) from the manifest's pickles, each sha256-checked."""
    def ld(rec):
        p = Path(rec["path"])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            raise lb.LockboxError(f"frozen model {p} is missing or changed since the freeze")
        return pickle.loads(p.read_bytes())
    return {k: ld(v) for k, v in man["mode_a"].items()}, {k: ld(v) for k, v in man["mode_l"].items()}


def summarise(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, cand), g in s.groupby(["target", "candidate"], sort=False):
        g = g[np.isfinite(g["truth"])]
        m = ev.metrics(g["truth"], g["pred"])
        rows.append({"target": target, "candidate": cand, "n": m["n"], "mdape": m["mdape"], "mape": m["mape"],
                     "maxape": m["maxape"], "inside_90": float(g["inside"].mean()) if len(g) else np.nan,
                     "width": ev.interval_width(g["lo90"], g["hi90"])})
    return pd.DataFrame(rows)


def score(paths: ri.Paths, seed: int, *, cat: pd.DataFrame, der: pd.DataFrame, qc_train: pd.DataFrame,
          allow_code_change: bool = False) -> dict:
    """Score lockbox 2 once with the frozen models (module docstring). Writes reports/r3/lockbox2.md and CSVs."""
    rp = report_path(paths)
    if rp.exists():
        raise lb.LockboxError(f"{rp} exists: lockbox 2 is scored once")
    man = json.loads(manifest_path(paths).read_text())
    pp = pred_path(paths)
    if runlog.file_hash(pp) != man["predictions_sha256"]:
        raise lb.LockboxError(f"{pp} changed since the freeze")
    code = checks.check_code(man["code"], allow_code_change)
    exploratory = bool(code["changed_files"]) or code["relation"] == "not an ancestor of HEAD"
    cat_l = cat[cat["source"] == SOURCE]
    if cat_l.empty:
        raise lb.LockboxError(f"no lockbox-2 cells under {settings.data_root() / SOURCE / 'cells'}")
    early = [r["cell_id"] for _, r in cat_l.iterrows() if int(data.read_meta(r["path"])["t0_real_ns"]) <= int(man["frozen_at_ns"])]
    if early:
        raise lb.LockboxError(f"lockbox-2 cells started before the freeze: {early[:3]}")
    models_a, models_l = load_frozen(man)
    pred = pd.read_csv(pp)
    qc_l = qcm.run_qc(cat_l, der)
    truth = ls.truth_table(cat_l, der, qc_l)
    sa = ls.score_mode_a(pred, truth)
    summ = summarise(sa)
    cat_t = cat[~cat["source"].isin(data.HOLDOUT_SOURCES)]
    idle = deploy.idle_table(cat_t, der, qc_train)
    calc = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle, deploy.cell_table(cat_t, der, qc_train))
    w_l = ls.lockbox_windows(cat_l, qc_l)
    lc = ls.mode_l_cells(w_l, cat_l, models_l) if len(w_l) else pd.DataFrame()
    lsum = (lc.groupby("rung")[["ape", "ape_meas", "ape_1s"]].median().join(lc.groupby("rung").size().rename("n_cells"))
            .reset_index() if len(lc) else pd.DataFrame())
    od = out_dir(paths)
    sa.to_csv(od / "lockbox2_mode_a.csv", index=False)
    summ.to_csv(od / "lockbox2_mode_a_summary.csv", index=False)
    calc.to_csv(od / "lockbox2_calc.csv", index=False)
    lc.to_csv(od / "lockbox2_mode_l.csv", index=False)
    label = ("EXPLORATORY: the scoring code changed since the freeze (" + ", ".join(code["changed_files"]) + ")"
             if exploratory else "confirmatory: frozen models, frozen predictions, unchanged code")
    rp.write_text(render(truth, sa, summ, calc, lsum, man, label))
    rec = runlog.append({"command": "lockbox2-score", "seed": seed, "label": label, "n_cells": len(cat_l),
                         "n_primary": int(qc_l["primary"].sum())}, paths.runs_log)
    return {"summary": summ, "calc": calc, "mode_l": lsum, "label": label, "rec": rec, "md": rp}


def _md(df: pd.DataFrame, pct=()) -> str:
    if df is None or not len(df):
        return "_none_\n"
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = [f"{100 * v:.1f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else "" for v in d[c]]
        elif d[c].dtype.kind == "f":
            d[c] = [f"{v:.4g}" if np.isfinite(v) else "" for v in d[c]]
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in d.itertuples(index=False))


def render(truth, sa, summ, calc, lsum, man, label) -> str:
    L = [f"# Lockbox 2 (R3), scored once\n\n**{label}.** Predictions frozen at "
         f"{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(man['frozen_at_ns'] / 1e9))} "
         f"(`lockbox2_predictions.csv` sha256 `{man['predictions_sha256'][:16]}`), before any lockbox-2 cell. Configurations: "
         + ", ".join(f"{c} (similar to {man['similar_to'][c.split(':')[0]]})" for c in man["configs"])
         + "; hooks off, 2400 MHz, 4 threads, fp32. Percent columns are percent.\n",
         "## Truth\n", _md(truth.drop(columns=["cell_ids"], errors="ignore")),
         "## Mode A, every candidate (AUTO is the primary)\n",
         _md(summ, pct=("mdape", "mape", "maxape", "inside_90")),
         "## Mode A per configuration (AUTO)\n",
         _md(sa[sa["candidate"] == mode_a.AUTO][["model", "imgsz", "target", "pred", "lo90", "hi90", "truth", "ape", "inside",
                                                  "novelty", "flagged"]], pct=("ape",)),
         "## Deployment calculator (serve phase, predicted duration)\n",
         _md(calc[["cell_id", "measured_J", "predicted_J_oh0", "ape_oh0", "predicted_J_est", "ape_est"]] if len(calc) else calc,
             pct=("ape_oh0", "ape_est")),
         "## Mode L (frozen rungs, median over cells)\n", _md(lsum, pct=("ape", "ape_meas", "ape_1s"))]
    return "\n".join(L)
