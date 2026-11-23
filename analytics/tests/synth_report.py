"""Synthetic run outputs for the report tests: configuration table, Mode A / Mode L / extras parquets,
runs.jsonl, QC and deployment tables, all under a tmp directory (`make_inputs`)."""
import json
import uuid

import numpy as np
import pandas as pd

import report_inputs as ri
import splits

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "effb0", "mbv3", "resnet18"]
MASTER = 20261122
CANDS = ["B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear", "M1Ridge", "M2Trees", "M3Factored"]
RUNGS = ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees"]
YOLO_L = ["yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]
NOISE = {"B0Median": 0.6, "B1MacsProportional": 0.4, "B2NearestModel": 0.3, "B3LogLinear": 0.2, "M1Ridge": 0.15,
         "M2Trees": 0.25, "M3Factored": 0.16, "AUTO": 0.15}


def make_configs() -> pd.DataFrame:
    rows = []
    for mi, m in enumerate(MODELS):
        for prec in ("fp32", "int8"):
            for th in (1, 4):
                for mhz in (1500, 2400):
                    key = f"{m}|{prec}|{th}|{mhz}"
                    rows.append({"config_key": key, "base_model": m, "variant": m if prec == "fp32" else f"{m}_int8",
                                 "precision": prec, "imgsz": 640, "batch": 1, "threads": th, "cpu_mhz": mhz,
                                 "rig": "nov2026" if mi < 6 else "oct2026", "n_cells": 3, "energy_cv": 0.02,
                                 "latency_cv": 0.03, "energy_mJ": 1000.0 * (mi + 1) * (1 + 0.1 * th) * 1500 / mhz,
                                 "latency_ms": 100.0 * (mi + 1) / th ** 0.3, "mean_P_W": 6.0 + mi * 0.2 + th * 0.1,
                                 "cell_ids": ";".join(f"{key}#{r}" for r in range(3))})
    return pd.DataFrame(rows)


def _pred(configs, split, target, seed, rng, cands=None) -> pd.DataFrame:
    folds = splits.make_splits(configs, seed)[split]
    by = configs.set_index("config_key")
    cands = cands or (CANDS if target == "energy_mJ" else [c for c in CANDS if c != "M3Factored"])
    rows = []
    for f in folds:
        for cand in [*cands, "AUTO"]:
            y = by.loc[f.test_keys, target].to_numpy(float)
            yh = y * np.exp(rng.normal(0.02, NOISE[cand], len(y)))
            nov = rng.uniform(0, 2, len(y))
            learned = cand.startswith("M") or cand == "AUTO"
            rows.append(pd.DataFrame({
                "split": split, "fold": f.fold, "similarity_class": f.similarity_class, "config_key": f.test_keys,
                "base_model": by.loc[f.test_keys, "base_model"].to_numpy(), "candidate": cand, "target": target, "seed": seed,
                "y_true": y, "y_pred": yh, "lo90": yh * 0.7 if learned else np.nan, "hi90": yh * 1.4 if learned else np.nan,
                "novelty": nov, "flagged": nov > 1.5, "threshold": 1.5,
                "params": json.dumps({"candidate": "M1Ridge", "params": {"alpha": 1.0}} if cand == "AUTO" else {})}))
    return pd.concat(rows, ignore_index=True)


def _log(path, rec):
    rec = {"ts": "2026-11-23T10:00:00+00:00", "commit": "c" * 40, "dirty": False, "data_hash": "d" * 64, "machine": "nuc", **rec}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def make_inputs(root, seeds=(MASTER, MASTER + 1, MASTER + 2)) -> ri.Paths:
    """Write a complete, consistent set of inputs under `root`; returns Paths (outputs under root/out)."""
    from pathlib import Path
    root = Path(root)
    p = ri.Paths(root / "runs.jsonl", root / "mode_a", root / "mode_l", root / "extras", root / "mode_a_configs.csv",
                 root / "qc_cells.csv", root / "deploy_validation.csv", root / "out", root / "out" / "fig")
    for d in (p.mode_a_dir, p.mode_l_dir, p.extras_dir):
        d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    configs = make_configs()
    configs.to_csv(p.configs_csv, index=False)
    cells = [c for ids in configs["cell_ids"] for c in ids.split(";")]
    pd.DataFrame({"cell_id": cells, "include": True, "primary": True, "rules": ["Q1flag" if i % 7 == 0 else "" for i in range(len(cells))],
                  "slip_scored_frac": 0.002, "ctx_read_frac_scored": 0.01, "max_residual_ms": 0.2, "throttle_current": False,
                  "max_temp_C": 61.0}).to_csv(p.qc_csv, index=False)

    def mode_a_run(split, target, seed):
        rid = uuid.uuid4().hex
        _pred(configs, split, target, seed, rng).to_parquet(p.mode_a_dir / f"{rid}.parquet", index=False)
        _log(p.runs_log, {"run_id": rid, "command": "mode-a", "split": split, "target": target, "seed": seed,
                          "split_sha256": "a" * 64, "table_sha256": "b" * 64})
        return rid
    for split in ri.SPLITS:
        for target in ri.TARGETS:
            mode_a_run(split, target, MASTER)
    for s in seeds:
        if s != MASTER:
            mode_a_run("S-LOMO", "energy_mJ", s)
    # extras
    for t in ri.TARGETS:
        _log(p.runs_log, {"run_id": uuid.uuid4().hex, "command": "mode-a-extras", "part": "fit_final", "seed": MASTER, "target": t,
                          "candidate": "M1Ridge", "params": {"alpha": 1.0}, "path": f"/x/{t}.pkl", "sha256": "e" * 64,
                          "auto_counts": {"M1Ridge": 5}})
    lc = pd.DataFrame([{"heldout": h, "k": k, "subset": "a;b", "mdape": 0.3 / k + 0.02 * i}
                       for i, h in enumerate(splits.YOLO) for k in (2, 3, 4, 5)])
    rid = uuid.uuid4().hex
    lc.to_parquet(p.extras_dir / f"{rid}.parquet", index=False)
    _log(p.runs_log, {"run_id": rid, "command": "mode-a-extras", "part": "learning_curve", "seed": MASTER, "target": "energy_mJ",
                      "candidate": "M1Ridge", "params": {"alpha": 1.0}})
    yolo_cfg = configs[configs["base_model"].isin(splits.YOLO)]
    for training in ("pooled", "v2only"):
        t = configs if training == "pooled" else yolo_cfg
        res = pd.concat([_pred(t, "S-LOMO", "energy_mJ", MASTER, rng).query("fold in @splits.YOLO")], ignore_index=True)
        rid = uuid.uuid4().hex
        res.to_parquet(p.extras_dir / f"{rid}.parquet", index=False)
        _log(p.runs_log, {"run_id": rid, "command": "mode-a-extras", "part": "v2_vs_pooled", "training": training, "split": "S-LOMO",
                          "target": "energy_mJ", "seed": MASTER})
    # deployment validation
    dep = pd.DataFrame([{"case": c, "cell_id": f"cell{i}", "phase": ph, "measured_J": 100.0, "predicted_J": 100.0 * (1 + 0.05 * i),
                         "ape": 0.05 * i, "source": "lomo", "notes": "n", "predicted_J_no_soak": 100.0, "lo90_J": 80.0, "hi90_J": 130.0,
                         "flagged": False, "saturated": False}
                        for i, (c, ph) in enumerate([("serve-default", "serve"), ("serve-fps", "serve"), ("ondemand", "forward"),
                                                     ("soak", "forward"), ("fan", "serve")])])
    dep.to_csv(p.deploy_csv, index=False)
    _log(p.runs_log, {"run_id": uuid.uuid4().hex, "command": "deploy-validate", "n_rows": len(dep)})
    # Mode L
    lr = uuid.uuid4().hex
    rows = []
    for f in YOLO_L:
        for c in range(3):
            for ri_, r in enumerate(RUNGS):
                et = 100.0 + 10 * c
                ape = abs(rng.normal(0.08 - 0.005 * ri_, 0.03))
                row = {"fold": f, "cell_id": f"{f}_{c}", "rung": r, "params": "{}", "exploratory": r == "L1-cell", "E_true_J": et,
                       "E_pred_J": et * (1 + ape), "ape": ape, "bias_pct": ape, "E_true_meas_J": et * 0.8,
                       "E_pred_meas_J": et * 0.8 * (1 + ape), "ape_meas": ape, "mae_window_W": 0.4, "ape_1s": 0.1, "ape_10s": 0.07,
                       "l1_fallback_frac": 0.05 if r.startswith("L1") else np.nan}
                row.update({f"ape_phase_{ph}": ape * 1.5 for ph in ("idle", "load_weights", "warmup", "forward", "serve")})
                rows.append(row)
    pd.DataFrame(rows).to_parquet(p.mode_l_dir / f"{lr}.parquet", index=False)
    _log(p.runs_log, {"run_id": lr, "command": "mode-l", "part": "lomo", "seed": MASTER})
    pm = pd.DataFrame([{"rung": r, "exploratory": r == "L1-cell", "cell_id": f"c{i}", "model_key": f"w{i % 6}",
                        "family": "v1" if i % 2 else "A", "E_true_J": 100.0, "E_pred_J": 100.0 * (1 + 0.1 * (i % 4)), "ape": 0.1 * (i % 4),
                        "bias_pct": 0.05, "mae_window_W": 0.5, "n_windows": 100, "slope": 0.9, "intercept": 1.0, "l1_fallback_frac": 0.02}
                       for r in ("L1", "L1-cell") for i in range(24)])
    pm.to_parquet(p.mode_l_dir / f"{lr}_pmic_v1.parquet", index=False)
    _log(p.runs_log, {"run_id": lr, "command": "mode-l", "part": "pmic_v1", "seed": MASTER})
    cfg_l = [k for k in configs["config_key"] if k.startswith(("yolov8s", "yolo11n"))][:6]
    pr = []
    for k in cfg_l:
        for j, kk in ((0, 1), (1, 0)):
            for r in RUNGS:
                t = 1000.0
                pe = t * np.exp(rng.normal(0, 0.1 + 0.01 * RUNGS.index(r)))
                a = t * np.exp(rng.normal(0, 0.2))
                pr.append({"config_key": k, "j": f"{k}#{j}", "k": f"{k}#{kk}", "rung": r, "exploratory": r == "L1-cell", "probe_mJ": pe,
                           "true_mJ": t, "ape": abs(pe / t - 1), "mode_a_mJ": a, "mode_a_ape": abs(a / t - 1),
                           "mode_a_run_id": "ma" * 16, "mode_a_seed_matched": True})
    pd.DataFrame(pr).to_parquet(p.mode_l_dir / f"{lr}_probe.parquet", index=False)
    _log(p.runs_log, {"run_id": lr, "command": "mode-l", "part": "probe", "seed": MASTER})
    return p


def family_map() -> pd.Series:
    return pd.Series({f"{f}_{c}": "A" if c < 2 else "soak" for f in YOLO_L for c in range(3)})


def overhead_frame() -> pd.DataFrame:
    return pd.DataFrame([{"base_model": "yolo11n", "imgsz": 640, "threads": 4, "cpu_mhz": 2400, "P_off_W": 9.91, "P_on_W": 9.59,
                          "n_off": 3, "n_on": 17, "delta_W": 0.32, "delta_pct": 3.34}])
