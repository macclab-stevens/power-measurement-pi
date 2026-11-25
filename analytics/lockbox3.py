"""Lockbox 3 (R4): predictions frozen and committed before any lockbox-3 cell is measured, then scored once.

Lockbox 2 was R3's confirmatory test and is now R4 training data. Lockbox 3 is R4's: six never-collected torchvision
models, each similar to a training model, hooks off, at mixed clocks, pre-registered in host/matrix_v4.py
(docs/review/2026-11-25-r4-plan.md):

  mbv2 (mobilenet_v2) 2400, shufv2 (shufflenet_v2_x1_0) 2000, resnet50 1500, effb2 (efficientnet_b2) 2400,
  vitb32 (vit_b_32) 2000, convnexts (convnext_small) 2400 MHz, 224 px, 4 threads, fp32, batch 1.

`freeze` refuses once any lockbox-3 cell exists or the predictions file exists. It fits on the R4 deployment table only
(reports/r4/mode_a_configs_r4.csv: hooks-off v3, v4 and lockbox 2). Mode A: every candidate, M5Pooled included, and
AUTO, in the new-model regime (`mode_a.fit_candidates(..., r4=True)`). The interval of record is the base-model
conformal one (`lo90`, `hi90`); the R3-style fold-weighted quantiles are kept as `lo90_r3`, `hi90_r3`. Mode L: L0,
L3-trees, L5-cmos-trees and the kernel rungs L3k-trees and L5k-trees, fitted on the v4 counter cells
(`mode_l.fit_final`, suffix `_r4`). All models are pickled under derived/models/lockbox3_r4/. It writes
reports/r4/lockbox3_predictions.csv and reports/r4/lockbox3_frozen.json. Commit both and tag `r4-frozen` before
measuring.

`score` refuses when reports/r4/lockbox3.md exists (scored once), when a lockbox-3 cell started before the freeze,
or when a pinned hash differs. A code change is refused unless `allow_code_change`, which labels the result
exploratory. It fits nothing.

  uv run python lockbox3.py freeze [--n-jobs 4]
  uv run python lockbox3.py score
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
import deploy
import derive
import evaluate as ev
import lockbox as lb
import lockbox2 as lb2
import lockbox_checks as checks
import lockbox_score as ls
import mode_a
import mode_l
import mode_l_models as mlm
import qc as qcm
import r4
import runlog
import settings
import static
import tables
import windows

SOURCE = "lockbox3"
CONFIGS = [("mbv2", 224, 2400), ("shufv2", 224, 2000), ("resnet50", 224, 1500), ("effb2", 224, 2400),
           ("vitb32", 224, 2000), ("convnexts", 224, 2400)]
SIMILAR = {"mbv2": "mbv3 / mbv3l", "shufv2": "mbv3", "resnet50": "resnet18 / resnet34", "effb2": "effb0 / effb1",
           "vitb32": "vitb16", "convnexts": "convnextt"}
FIXED = {"precision": "fp32", "batch": 1, "threads": 4, "rig": "nov2026", "layer_hooks": "off"}
REGIME = "new-model"
SUFFIX = "_r4"
RUNGS = ("L0", "L3-trees", "L5-cmos-trees", "L3k-trees", "L5k-trees")
OUT = settings.HERE / "reports" / "r4"
PRED = OUT / "lockbox3_predictions.csv"
MANIFEST = OUT / "lockbox3_frozen.json"
REPORT = OUT / "lockbox3.md"
STATIC_DIR = settings.HERE / "lockbox3_static"
PRED_COLS = ["config_key", "model", "imgsz", "cpu_mhz", "candidate", "role", "chosen", "target", "pred", "lo90", "hi90",
             "lo90_r3", "hi90_r3", "novelty", "threshold", "flagged", "model_sha256", "static_csv_sha256"]


def cells_exist() -> bool:
    d = settings.data_root() / SOURCE / "cells"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def feature_rows() -> tuple:
    """(feature rows of the lockbox-3 configurations, {(model, imgsz): static CSV sha256})."""
    rows, shas = [], {}
    for model, imgsz, mhz in CONFIGS:
        p = checks.static_csv_path(STATIC_DIR, model, imgsz)
        if not p.is_file():
            raise lb.LockboxError(f"static features file {p} is missing (tools/static_features.py)")
        rows.append({"base_model": model, "variant": model, "imgsz": imgsz, "cpu_mhz": mhz, "quant": "fp32",
                     "n_quantized_modules": 0, **FIXED, **static.features_from_layer_table(pd.read_csv(p))})
        shas[(model, imgsz)] = runlog.file_hash(p)
    st = static.add_flops(pd.DataFrame(rows).set_index(["base_model", "imgsz"])).reset_index()
    t = tables.add_derived_columns(st)
    t.insert(0, "config_key", [tables.config_key(r) for _, r in t.iterrows()])
    return t, shas


def _pickle(obj, name: str) -> dict:
    d = settings.derived_dir() / "models" / "lockbox3_r4"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pkl"
    p.write_bytes(pickle.dumps(obj))
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def predict_mode_a(t, feats, shas, desc_train, seed, threshold) -> tuple:
    desc = lb.descriptor(desc_train, feats)
    nov = lb.novelty_by_model(desc, feats["base_model"], set(t["base_model"]))
    rows, frozen = [], {}
    for target in r4.TARGETS:
        fits = mode_a.fit_candidates(t, target, seed, desc, REGIME, r4=True)
        for cand, f in fits.items():
            rec = _pickle(f["model"], f"mode_a_{target}_{cand}")
            frozen[f"{target}/{cand}"] = {**rec, "params": f["params"], "chosen": f["chosen"], "score": f["score"],
                                          "qcf": f["qcf"]}
            pred = np.asarray(f["model"].predict(feats), float)
            for i, r in enumerate(feats.itertuples()):
                n = nov[r.base_model]
                rows.append({"config_key": r.config_key, "model": r.base_model, "imgsz": int(r.imgsz), "cpu_mhz": int(r.cpu_mhz),
                             "candidate": cand, "role": "primary" if cand == mode_a.AUTO else "secondary",
                             "chosen": f["chosen"], "target": target, "pred": float(pred[i]),
                             "lo90": float(pred[i] * np.exp(-f["qcf"])), "hi90": float(pred[i] * np.exp(f["qcf"])),
                             "lo90_r3": float(pred[i] * np.exp(f["q5"])), "hi90_r3": float(pred[i] * np.exp(f["q95"])),
                             "novelty": n, "threshold": threshold, "flagged": bool(n > threshold),
                             "model_sha256": rec["sha256"], "static_csv_sha256": shas[(r.base_model, int(r.imgsz))]})
    return pd.DataFrame(rows, columns=PRED_COLS), frozen


def _v4_counter_cells(cat):
    return cat[(cat["source"] == "v4") & cat["has_counters"].astype(bool)].reset_index(drop=True)


def freeze(seed: int, n_jobs: int = 4) -> dict:
    if cells_exist():
        raise lb.LockboxError("lockbox-3 cells already exist: the predictions must be frozen and committed first")
    if PRED.exists():
        raise lb.LockboxError(f"{PRED} exists: it is the committed record of the frozen predictions")
    t = pd.read_csv(r4.TABLE)
    cat = data.catalogue()
    v4 = _v4_counter_cells(cat)
    w = windows.build_windows(cat)
    w = w[w["cell_id"].isin(v4["cell_id"])]
    lb2.check_training(t[~t["cell_ids"].str.contains("lockbox2_")], w)   # lockbox-2 cells are R4 training data
    if t["cell_ids"].str.contains("lockbox3").any():
        raise lb.LockboxError("lockbox-3 cells in the R4 table")
    desc_train = static.model_descriptor(r4.static_features(cat))
    slomo = pd.read_parquet(r4.latest_runs("mode-a-r4")[("monsoon", "S-LOMO", "energy_mJ")])
    threshold = lb.threshold_from_slomo(slomo)
    feats, shas = feature_rows()
    pred, frozen_a = predict_mode_a(t, feats, shas, desc_train, seed, threshold)
    q = pd.read_csv(settings.HERE / "reports" / "qc_cells.csv")
    from joblib import Parallel, delayed
    rungs = {k: v for k, v in mlm.default_rungs().items() if k in RUNGS}
    res = Parallel(n_jobs=n_jobs)(delayed(mode_l.fit_final)(w, v4, q, n, seed, rungs, SUFFIX + "_lb3") for n in rungs)
    frozen_l = {n: {"path": info["path"], "sha256": info["sha256"], "params": info["params"], "n_cells": info["n_cells"],
                    "base_models": info["base_models"]} for n, (_, info) in zip(rungs, res)}
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED, index=False)
    man = {"seed": seed, "frozen_at_ns": time.time_ns(), "regime": REGIME,
           "configs": [f"{m}:{s}@{mhz}" for m, s, mhz in CONFIGS], "similar_to": SIMILAR, "fixed": FIXED,
           "interval": "base-model conformal (lo90, hi90); R3-style fold quantiles kept as lo90_r3, hi90_r3",
           "predictions_csv": PRED.name, "predictions_sha256": runlog.file_hash(PRED), "mode_a": frozen_a,
           "mode_l": frozen_l, "static_csv_sha256": {f"{m}:{s}": h for (m, s), h in shas.items()},
           "static_flops_sha256": runlog.file_hash(static.FLOPS_CSV), "threshold": threshold,
           "training": {"table": str(r4.TABLE.relative_to(settings.HERE)), "table_sha256": runlog.file_hash(r4.TABLE),
                        "n_configs": len(t), "base_models": sorted(set(t["base_model"])),
                        "mode_l_cells": int(v4["cell_id"].nunique())},
           "code": checks.code_pin()}
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True, default=str) + "\n")
    runlog.append({"command": "lockbox3-freeze", "seed": seed, "predictions_sha256": man["predictions_sha256"],
                   "manifest_sha256": runlog.file_hash(MANIFEST), "n_rows": len(pred),
                   "auto_chosen": {k: v["chosen"] for k, v in frozen_a.items() if k.endswith("/AUTO")}})
    return {"pred": pred, "manifest": man}


def score(seed: int, allow_code_change: bool = False) -> dict:
    if REPORT.exists():
        raise lb.LockboxError(f"{REPORT} exists: lockbox 3 is scored once")
    man = json.loads(MANIFEST.read_text())
    if runlog.file_hash(PRED) != man["predictions_sha256"]:
        raise lb.LockboxError(f"{PRED} changed since the freeze")
    code = checks.check_code(man["code"], allow_code_change)
    exploratory = bool(code["changed_files"]) or code["relation"] == "not an ancestor of HEAD"
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    cat_l = cat[cat["source"] == SOURCE]
    if cat_l.empty:
        raise lb.LockboxError(f"no lockbox-3 cells under {settings.data_root() / SOURCE / 'cells'}")
    early = [r["cell_id"] for _, r in cat_l.iterrows() if int(data.read_meta(r["path"])["t0_real_ns"]) <= int(man["frozen_at_ns"])]
    if early:
        raise lb.LockboxError(f"lockbox-3 cells started before the freeze: {early[:3]}")
    models_a, models_l = lb2.load_frozen(man)
    pred = pd.read_csv(PRED)
    qc_l = qcm.run_qc(cat_l, der)
    truth = ls.truth_table(cat_l, der, qc_l)
    sa = ls.score_mode_a(pred, truth)
    sa["inside_r3"] = [(r.truth >= p.lo90_r3) and (r.truth <= p.hi90_r3) if np.isfinite(r.truth) else np.nan
                       for r, p in zip(sa.itertuples(), pred.itertuples())]
    summ = lb2.summarise(sa)
    summ["inside_90_r3"] = [float(g["inside_r3"].mean()) for _, g in sa.groupby(["target", "candidate"], sort=False)]
    qc_train = pd.read_csv(settings.HERE / "reports" / "qc_cells.csv")
    cat_t = cat[~cat["source"].isin(data.HOLDOUT_SOURCES)]
    idle = deploy.idle_table(cat_t, der, qc_train)
    calc = ls.calculator_posthoc_gap(cat_l, der, qc_l, pred, idle, deploy.cell_table(cat_t, der, qc_train))
    w_l = ls.lockbox_windows(cat_l, qc_l)
    lc = ls.mode_l_cells(w_l, cat_l, models_l) if len(w_l) else pd.DataFrame()
    lsum = (lc.groupby("rung")[["ape", "ape_meas", "ape_1s"]].median().join(lc.groupby("rung").size().rename("n_cells"))
            .reset_index() if len(lc) else pd.DataFrame())
    sa.to_csv(OUT / "lockbox3_mode_a.csv", index=False)
    summ.to_csv(OUT / "lockbox3_mode_a_summary.csv", index=False)
    calc.to_csv(OUT / "lockbox3_calc.csv", index=False)
    lc.to_csv(OUT / "lockbox3_mode_l.csv", index=False)
    label = ("EXPLORATORY: the scoring code changed since the freeze (" + ", ".join(code["changed_files"]) + ")"
             if exploratory else "confirmatory: frozen models, frozen predictions, unchanged code")
    REPORT.write_text(render(truth, sa, summ, calc, lsum, man, label))
    runlog.append({"command": "lockbox3-score", "seed": seed, "label": label, "n_cells": len(cat_l),
                   "n_primary": int(qc_l["primary"].sum())})
    return {"summary": summ, "calc": calc, "mode_l": lsum, "label": label, "sa": sa}


def render(truth, sa, summ, calc, lsum, man, label) -> str:
    md = lb2._md
    L = [f"# Lockbox 3 (R4), scored once\n\n**{label}.** Predictions frozen at "
         f"{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(man['frozen_at_ns'] / 1e9))} "
         f"(`lockbox3_predictions.csv` sha256 `{man['predictions_sha256'][:16]}`), before any lockbox-3 cell. Configurations: "
         + ", ".join(f"{c} (similar to {man['similar_to'][c.split(':')[0]]})" for c in man["configs"])
         + "; hooks off, 4 threads, fp32. Interval of record: base-model conformal (`inside_90`); R3-style beside it "
           "(`inside_90_r3`). Percent columns are percent.\n",
         "## Truth\n", md(truth.drop(columns=["cell_ids"], errors="ignore")),
         "## Mode A, every candidate (AUTO is the primary)\n",
         md(summ, pct=("mdape", "mape", "maxape", "inside_90", "inside_90_r3")),
         "## Mode A per configuration (AUTO)\n",
         md(sa[sa["candidate"] == mode_a.AUTO][["model", "imgsz", "target", "pred", "lo90", "hi90", "truth", "ape", "inside",
                                                 "novelty", "flagged"]], pct=("ape",)),
         "## Deployment calculator (serve phase, predicted duration)\n",
         md(calc[["cell_id", "measured_J", "predicted_J_oh0", "ape_oh0", "predicted_J_est", "ape_est"]] if len(calc) else calc,
            pct=("ape_oh0", "ape_est")),
         "## Mode L (frozen rungs, trained on v4, median over cells)\n", md(lsum, pct=("ape", "ape_meas", "ape_1s"))]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--n-jobs", type=int, default=4)
    p = sub.add_parser("score")
    p.add_argument("--allow-code-change", action="store_true")
    a = ap.parse_args(argv)
    seed = settings.seed()
    try:
        if a.cmd == "freeze":
            out = freeze(seed, a.n_jobs)
            p = out["pred"]
            print(p[p["role"] == "primary"][["model", "imgsz", "cpu_mhz", "target", "chosen", "pred", "lo90", "hi90",
                                             "flagged"]].to_string(index=False))
            print(f"frozen -> {PRED}; commit reports/r4/ and tag r4-frozen before measuring lockbox 3")
        else:
            out = score(seed, a.allow_code_change)
            print(out["label"])
            s = out["summary"]
            print(s[s["candidate"].isin(["AUTO", "M1Ridge", "M4Physics", "M5Pooled"])].to_string(index=False))
            print(out["mode_l"].to_string(index=False))
    except lb.LockboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
