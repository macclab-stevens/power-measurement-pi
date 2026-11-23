"""Mode L (spec 7): live board-power model from causal 100 ms windows, scored on accumulated energy of
YOLO models it never saw (leave one base model out), the PMIC-only check across all workloads, and probe
mode (does a short unmetered run of a new model predict its energy per image better than Mode A?).

Nothing from a test cell reaches a fit: the outer fold fits on the other base models, the grid point is
chosen by an inner leave-one-model-out over those models only, and window models only ever see the rows
passed to `fit`. Energies are J (E_mJ / 1000), power W, probe energy per image mJ."""
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import deploy
import evaluate
import mode_l_models as mm
import runlog
import tables

PHASES = ("idle", "load_weights", "warmup", "forward", "serve")
IDLE_PHASES = ("idle", "baseline_A", "baseline_B")        # baseline windows are the idle phase
HORIZONS = {"ape_1s": 10, "ape_10s": 100}                 # windows per block (100 ms windows)
METRIC_KEYS = ("mdape", "mape", "mae", "rmse", "bias_pct")


# ------------------------------------------------------------------ scoring helpers

def block_ape(p_hat, dur_s, e_mJ, size: int) -> float:
    """Median over consecutive non-overlapping blocks of `size` windows (a trailing partial block is
    dropped) of |sum(p_hat * dur_s) - sum(E_mJ) / 1000| / (sum(E_mJ) / 1000). NaN if no full block."""
    n = (len(p_hat) // size) * size
    if n == 0:
        return float("nan")
    pred = (np.asarray(p_hat, float) * np.asarray(dur_s, float))[:n].reshape(-1, size).sum(1)
    true = np.asarray(e_mJ, float)[:n].reshape(-1, size).sum(1) / 1000.0
    return float(np.median(np.abs(pred - true) / true))


def _ape(pred, true) -> float:
    return abs(pred - true) / true if true > 0 else float("nan")


def _cell_apes(w: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """Whole-cell energy APE per cell of `w` (order of first appearance)."""
    codes, _ = pd.factorize(w["cell_id"])
    n = codes.max() + 1
    pred = np.bincount(codes, weights=p * w["dur_s"].to_numpy(float), minlength=n)
    true = np.bincount(codes, weights=w["E_mJ"].to_numpy(float), minlength=n) / 1000.0
    return np.abs(pred - true) / true


def _score_cell(c: pd.DataFrame, p: np.ndarray) -> dict:
    """Metrics of one cell's windows `c` (sorted by w_idx) with window power predictions `p`."""
    dur, e, ph = c["dur_s"].to_numpy(float), c["E_mJ"].to_numpy(float), c["phase"].to_numpy()
    e_true, e_pred = e.sum() / 1000.0, float((p * dur).sum())
    meas = ph == tables.MODE
    out = {"E_true_J": e_true, "E_pred_J": e_pred, "ape": _ape(e_pred, e_true),
           "bias_pct": (e_pred - e_true) / e_true if e_true > 0 else float("nan"),
           "E_true_meas_J": e[meas].sum() / 1000.0 if meas.any() else float("nan"),
           "E_pred_meas_J": float((p * dur)[meas].sum()) if meas.any() else float("nan"),
           "mean_P_pred_meas_W": float(p[meas].mean()) if meas.any() else float("nan"),
           "mae_window_W": float(np.mean(np.abs(p - c["P_W"].to_numpy(float)))),
           "n_windows": len(c), "n_windows_meas": int(meas.sum())}
    out["ape_meas"] = _ape(out["E_pred_meas_J"], out["E_true_meas_J"]) if meas.any() else float("nan")
    for name, size in HORIZONS.items():
        out[name] = block_ape(p, dur, e, size)
    for phase in PHASES:
        m = np.isin(ph, IDLE_PHASES if phase == "idle" else (phase,))
        out[f"ape_phase_{phase}"] = (_ape(float((p * dur)[m].sum()), e[m].sum() / 1000.0)
                                     if m.any() else float("nan"))
    return out


# ------------------------------------------------------------------ outer / inner folds

def _prep(w: pd.DataFrame, cat: pd.DataFrame, qc: pd.DataFrame | None) -> pd.DataFrame:
    """Windows of QC-primary catalogue cells with `base_model`, sorted by (cell_id, w_idx)."""
    c = cat if qc is None else cat[cat["cell_id"].isin(qc.loc[qc["primary"].astype(bool), "cell_id"])]
    w = w[w["cell_id"].isin(c["cell_id"])].copy()
    w["base_model"] = w["cell_id"].map(dict(zip(cat["cell_id"], cat["base_model"])))
    return w.sort_values(["cell_id", "w_idx"], kind="stable").reset_index(drop=True)


def _fit(rung: mm.Rung, seed: int, params: dict, train: pd.DataFrame):
    return rung.make(seed).fit(train, params, mm.cell_weights(train))


def _select(rung: mm.Rung, seed: int, train: pd.DataFrame) -> dict:
    """Grid point with the lowest median whole-cell energy APE over the inner test cells of a
    leave-one-base-model-out inside `train` (ties: first point)."""
    if len(rung.grid) == 1:
        return rung.grid[0]
    apes = [[] for _ in rung.grid]
    for m in sorted(train["base_model"].unique()):
        tr, te = train[train["base_model"] != m], train[train["base_model"] == m]
        for gi, params in enumerate(rung.grid):
            apes[gi].extend(_cell_apes(te, _fit(rung, seed, params, tr).predict(te)))
    med = [np.nanmedian(a) if np.isfinite(a).any() else np.inf for a in apes]
    return rung.grid[int(np.argmin(med))]


def _run_fold(held: str, w: pd.DataFrame, rungs: dict, seed: int) -> list:
    train, test = w[w["base_model"] != held], w[w["base_model"] == held]
    rows = []
    for name, rung in rungs.items():
        params = _select(rung, seed, train)
        model = _fit(rung, seed, params, train)
        p = model.predict(test)
        fb = getattr(model, "fallback_frac", None)
        for cid, idx in test.groupby("cell_id", sort=True).indices.items():
            c = test.iloc[idx]
            rows.append({"fold": held, "cell_id": cid, "rung": name,
                         "params": json.dumps(params, sort_keys=True),
                         **_score_cell(c, p[idx]),
                         "l1_fallback_frac": fb(c) if fb else float("nan")})
    return rows


def run_lomo(w: pd.DataFrame, cat: pd.DataFrame, seed: int, *, qc: pd.DataFrame | None = None,
             rungs: dict | None = None, folds=None, n_jobs: int = 1) -> pd.DataFrame:
    """Leave one YOLO base model out over the QC-primary counter cells of `w`; one row per
    (fold, test cell, rung). `seed` is the tree random_state. `rungs` defaults to
    `mode_l_models.default_rungs()`; `folds` restricts the held-out base models; results are identical
    for any `n_jobs`."""
    rungs = rungs or mm.default_rungs()
    w = _prep(w, cat, qc)
    held = sorted(w["base_model"].unique()) if folds is None else list(folds)
    res = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(h, w, rungs, seed) for h in held)
    return pd.DataFrame([r for fold in res for r in fold])


# ------------------------------------------------------------------ PMIC-only, leave one workload out

def run_pmic_v1(pw: pd.DataFrame, cat: pd.DataFrame, qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """L1 (P = a + b * pmic_W) scored leave-one-workload-out (by `model_key`) over the QC-primary cells
    of `pw` (v1 and v2), whole-cell energy APE per cell with weights 1 / windows per cell."""
    c = cat if qc is None else cat[cat["cell_id"].isin(qc.loc[qc["primary"].astype(bool), "cell_id"])]
    info = c.set_index("cell_id")[["model_key", "family"]]
    pw = pw[pw["cell_id"].isin(info.index)].copy()
    pw["model_key"] = pw["cell_id"].map(info["model_key"])
    pw = pw.sort_values(["cell_id", "t_start_ns"] if "t_start_ns" in pw else ["cell_id"],
                        kind="stable").reset_index(drop=True)
    rows = []
    for key in sorted(pw["model_key"].unique()):
        tr, te = pw[pw["model_key"] != key], pw[pw["model_key"] == key]
        model = mm.L1Pmic().fit(tr, {}, mm.cell_weights(tr))
        p = model.predict(te)
        for cid, idx in te.groupby("cell_id", sort=True).indices.items():
            cc = te.iloc[idx]
            s = _score_cell(cc, p[idx])
            rows.append({"cell_id": cid, "model_key": key, "family": info.loc[cid, "family"],
                         "E_true_J": s["E_true_J"], "E_pred_J": s["E_pred_J"], "ape": s["ape"],
                         "bias_pct": s["bias_pct"], "mae_window_W": s["mae_window_W"],
                         "n_windows": s["n_windows"], "slope": model.b, "intercept": model.a,
                         "l1_fallback_frac": model.fallback_frac(cc)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ probe mode

def probe(w: pd.DataFrame, der: pd.DataFrame, cat: pd.DataFrame, mode_a_pred: pd.DataFrame | None,
          seed: int, *, lomo: pd.DataFrame | None = None, qc: pd.DataFrame | None = None,
          rungs: dict | None = None, n_jobs: int = 1) -> pd.DataFrame:
    """For every Mode A configuration with >= 2 QC-primary, Mode A eligible counter cells and every
    ordered pair of its cells (j, k), j != k: probe_mJ = mean predicted power over cell j's measured-phase
    windows (W; the outer-fold model that never saw j's base model) x latency_ms_per_image of j;
    true_mJ = energy_mJ_per_image of k. `mode_a_pred` holds the held-out Mode A predictions
    (`config_key`, `e_img_mJ`, e.g. from deploy.preds_from_mode_a); `lomo` is a run_lomo result (computed
    here when None). Columns: config_key, j, k, rung, probe_mJ, true_mJ, ape, mode_a_mJ, mode_a_ape."""
    if lomo is None:
        lomo = run_lomo(w, cat, seed, qc=qc, rungs=rungs, n_jobs=n_jobs)
    d = der[der["mode"] == tables.MODE].set_index("cell_id")
    ok = set(lomo["cell_id"])
    c = cat[cat["cell_id"].isin(ok) & cat["mode_a_eligible"].astype(bool)].copy()
    c["config_key"] = [deploy.mode_a_key_for_cell(r) for _, r in c.iterrows()]
    mode_a = ({} if mode_a_pred is None or len(mode_a_pred) == 0
              else mode_a_pred.drop_duplicates("config_key").set_index("config_key")["e_img_mJ"].to_dict())
    pbar = lomo.set_index(["cell_id", "rung"])["mean_P_pred_meas_W"]
    rows = []
    for key, g in sorted(c.groupby("config_key"), key=lambda kv: kv[0]):
        cells = sorted(g["cell_id"])
        if len(cells) < 2:
            continue
        a = float(mode_a.get(key, float("nan")))
        for j in cells:
            for k in cells:
                if j == k:
                    continue
                true = float(d.loc[k, "energy_mJ_per_image"]) if k in d.index else float("nan")
                lat = float(d.loc[j, "latency_ms_per_image"]) if j in d.index else float("nan")
                for rung in lomo["rung"].unique():
                    pr = float(pbar.get((j, rung), float("nan"))) * lat
                    rows.append({"config_key": key, "j": j, "k": k, "rung": rung, "probe_mJ": pr,
                                 "true_mJ": true, "ape": _ape(pr, true), "mode_a_mJ": a,
                                 "mode_a_ape": _ape(a, true)})
    return pd.DataFrame(rows, columns=["config_key", "j", "k", "rung", "probe_mJ", "true_mJ", "ape",
                                       "mode_a_mJ", "mode_a_ape"])


# ------------------------------------------------------------------ the run

def _pooled(df: pd.DataFrame, true: str, pred: str) -> dict:
    return {k: v for k, v in evaluate.metrics(df[true], df[pred]).items()}


def run_all(w, pw, cat, der, qc, *, seed, n_jobs, mode_a_pred, out_dir, rungs=None, folds=None,
            runlog_path=None) -> dict:
    """Run the ladder, the PMIC-only check and (when Mode A predictions are given) probe mode; write
    <run_id>.parquet, <run_id>_pmic_v1.parquet and <run_id>_probe.parquet to `out_dir` and one runlog
    line per part. Returns run_id, the three frames and `probe_skipped`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    lomo = run_lomo(w, cat, seed, qc=qc, rungs=rungs, folds=folds, n_jobs=n_jobs)
    wall = time.perf_counter() - t0
    lomo.to_parquet(out / f"{run_id}.parquet", index=False)
    per = {r: {**_pooled(g, "E_true_J", "E_pred_J"),
               "median_ape_1s": float(g["ape_1s"].median()), "median_ape_10s": float(g["ape_10s"].median()),
               "mdape_meas": _pooled(g, "E_true_meas_J", "E_pred_meas_J")["mdape"]}
           for r, g in lomo.groupby("rung", sort=False)}
    runlog.append({"run_id": run_id, "command": "mode-l", "part": "lomo", "seed": seed, "n_jobs": n_jobs,
                   "folds": sorted(lomo["fold"].unique()), "n_cells": int(lomo["cell_id"].nunique()),
                   "metrics": per, "wall_s": round(wall, 1)}, runlog_path)
    t0 = time.perf_counter()
    pm = run_pmic_v1(pw, cat, qc)
    wall_p = time.perf_counter() - t0
    pm.to_parquet(out / f"{run_id}_pmic_v1.parquet", index=False)
    runlog.append({"run_id": run_id, "command": "mode-l", "part": "pmic_v1", "seed": seed, "n_cells": len(pm),
                   "median_ape": float(pm["ape"].median()),
                   "median_ape_by_family": {f: float(g["ape"].median()) for f, g in pm.groupby("family")},
                   "wall_s": round(wall_p, 1)}, runlog_path)
    skipped = mode_a_pred is None
    pr = None
    if not skipped:
        t0 = time.perf_counter()
        pr = probe(w, der, cat, mode_a_pred, seed, lomo=lomo)
        pr.to_parquet(out / f"{run_id}_probe.parquet", index=False)
        runlog.append({"run_id": run_id, "command": "mode-l", "part": "probe", "seed": seed, "n_pairs": len(pr),
                       "median_ape": {r: float(g["ape"].median()) for r, g in pr.groupby("rung", sort=False)},
                       "median_mode_a_ape": float(pr["mode_a_ape"].median()) if len(pr) else float("nan"),
                       "wall_s": round(time.perf_counter() - t0, 1)}, runlog_path)
    return {"run_id": run_id, "lomo": lomo, "pmic_v1": pm, "probe": pr, "probe_skipped": skipped,
            "metrics": per, "wall_s": wall}
