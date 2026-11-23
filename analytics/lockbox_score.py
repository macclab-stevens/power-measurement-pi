"""Lockbox, part 2: score the frozen models on the measured lockbox cells. Nothing is fitted here.

Truth comes from the lockbox cells only (QC, derive, windows). Mode A predictions are read from the
committed lockbox_predictions.csv (its sha256 is recorded); Mode L uses the frozen window-model pickles
(sha256-checked against runs.jsonl). The calculator takes the frozen Mode A AUTO-final predictions, the
measured serve duration and request count, and the idle power of the TRAINING cells."""
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import data
import deploy
import evaluate
import lockbox as lb
import mode_l
import mode_l_models as mlm
import report_inputs as ri
import runlog
import settings
import tables
import windows

DESIGN = {"precision": "fp32", "batch": 1, "threads": 4, "cpu_mhz": 2400, "governor": "pinned", "fan_mode": "full",
          "rig": "nov2026", "task": "image"}
NEW_ONLY = "new models"
KEY = ["base_model", "imgsz"]
TARGET_COL = {"energy_mJ": "energy_mJ", "latency_ms": "latency_ms", "mean_P_W": "mean_P_W"}


def check_design(cat_l: pd.DataFrame) -> None:
    """Every lockbox cell is fp32, batch 1, 4 threads, 2400 MHz pinned, fan full, nov2026, image task, with no
    serve sweep, soak or concurrency (the default serve phase of every cell is not a sweep). Raises ValueError."""
    for _, r in cat_l.iterrows():
        bad = [f"{k}={r[k]!r}" for k, v in DESIGN.items() if r[k] != v]
        for k in ("serve_fps", "soak_minutes"):
            if not pd.isna(r[k]):
                bad.append(f"{k}={r[k]!r}")
        if r["concurrent_mode"] is not None and not pd.isna(r["concurrent_mode"]):
            bad.append(f"concurrent_mode={r['concurrent_mode']!r}")
        if bad:
            raise ValueError(f"lockbox cell {r['cell_id']} is outside the lockbox design: {', '.join(bad)}")


# ------------------------------------------------------------------ Mode A

def truth_table(cat_l: pd.DataFrame, der: pd.DataFrame, qc_l: pd.DataFrame) -> pd.DataFrame:
    """One row per (base_model, imgsz) over QC-primary lockbox cells with a finite forward energy: n_cells,
    cell_ids and the mean energy_mJ, latency_ms, mean_P_W per image. No `mode_a_eligible` filter: the
    lockbox models are outside data.VISION_MODELS."""
    t = deploy.cell_table(cat_l, der, qc_l)
    t = t[np.isfinite(t["energy_mJ_per_image"].astype(float))]
    rows = []
    for (m, s), g in t.sort_values("cell_id").groupby(KEY, sort=True):
        rows.append({"base_model": m, "imgsz": int(s), "config_key": tables.config_key(g.iloc[0]), "n_cells": len(g),
                     "cell_ids": ";".join(g["cell_id"]),
                     **{k: float(g[col].mean()) for k, col in tables.MEASURES.items() if k in TARGET_COL}})
    return pd.DataFrame(rows, columns=["base_model", "imgsz", "config_key", "n_cells", "cell_ids", *TARGET_COL])


def score_mode_a(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per configuration x candidate x target: truth, APE, inside the 90 % interval, flagged. A configuration with
    no QC-primary cell has NaN truth and APE (counted, not dropped)."""
    p = pred.rename(columns={"model": "base_model"})
    tr = truth.set_index(KEY)
    rows = []
    for r in p.itertuples():
        k = (r.base_model, int(r.imgsz))
        y = float(tr.loc[k, r.target]) if k in tr.index else float("nan")
        rows.append({"config_key": r.config_key, "model": r.base_model, "imgsz": int(r.imgsz), "candidate": r.candidate,
                     "role": r.role, "target": r.target, "pred": r.pred, "lo90": r.lo90, "hi90": r.hi90, "truth": y,
                     "ape": abs(r.pred - y) / y if y > 0 else float("nan"),
                     "inside": bool(r.lo90 <= y <= r.hi90) if np.isfinite(y) else False, "flagged": bool(r.flagged),
                     "novelty": r.novelty, "n_cells": int(tr.loc[k, "n_cells"]) if k in tr.index else 0,
                     "new_model": r.novelty > 0})
    return pd.DataFrame(rows)


def summarise_mode_a(s: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per target, role and candidate over all configurations and over the new models only."""
    rows = []
    for (target, role, cand), g in s.groupby(["target", "role", "candidate"], sort=False):
        for label, sub in (("all", g), (NEW_ONLY, g[g["new_model"]])):
            ok = sub[np.isfinite(sub["truth"])]
            m = evaluate.metrics(ok["truth"], ok["pred"]) if len(ok) else {"n": 0, "mdape": np.nan, "mape": np.nan,
                                                                          "bias_pct": np.nan}
            rows.append({"target": target, "role": role, "candidate": cand, "group": label, "n": m["n"],
                         "mdape": m["mdape"], "mape": m["mape"], "bias_pct": m["bias_pct"],
                         "inside_90": float(ok["inside"].mean()) if len(ok) else np.nan,
                         "n_flagged": int(ok["flagged"].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calculator

def calculator(cat_l: pd.DataFrame, der: pd.DataFrame, qc_l: pd.DataFrame, pred: pd.DataFrame,
               idle: pd.DataFrame) -> pd.DataFrame:
    """Serve-phase energy per QC-primary lockbox cell: the calculator fed with the frozen AUTO-final energy and
    latency predictions, the measured request count and duration; `idle` is the training cells' idle table.
    The 90 % band uses the energy interval. Cells with no serve phase or no idle row are counted in attrs."""
    prim = pred[pred["role"] == "primary"]
    e = prim[prim["target"] == "energy_mJ"].set_index(["model", "imgsz"])
    t = prim[prim["target"] == "latency_ms"].set_index(["model", "imgsz"])
    cells = deploy.cell_table(cat_l, der, qc_l)
    rows, skipped = [], {"no_serve": 0, "no_idle": 0, "no_prediction": 0}
    for _, r in cells.iterrows():
        k = (r["base_model"], int(r["imgsz"]))
        n = float(r["serve_n_requests"]) * float(r["batch"]) if pd.notna(r["serve_n_requests"]) else 0.0
        if not n > 0 or not np.isfinite(r["serve_dur_ms"]):
            skipped["no_serve"] += 1
            continue
        if k not in e.index:
            skipped["no_prediction"] += 1
            continue
        scen = {"cpu_key": deploy.cpu_key(r), "fan_mode": r["fan_mode"], "rig": r["rig"], "n_images": n,
                "duration_s": r["serve_dur_ms"] / 1000.0}
        tm = float(t.loc[k, "pred"])
        try:
            p, lo, hi = (deploy.predict(scen, float(e.loc[k, c]), tm, idle, None, None)["E_J"]
                         for c in ("pred", "lo90", "hi90"))
        except KeyError:
            skipped["no_idle"] += 1
            continue
        meas = float(r["serve_E_mJ"]) / 1000.0
        rows.append({"cell_id": r["cell_id"], "model": k[0], "imgsz": k[1], "n_images": n, "duration_s": scen["duration_s"],
                     "measured_J": meas, "predicted_J": p, "lo90_J": lo, "hi90_J": hi, "ape": abs(p - meas) / meas,
                     "inside": bool(lo <= meas <= hi), "flagged": bool(e.loc[k, "flagged"]), "new_model": bool(e.loc[k, "novelty"] > 0)})
    out = pd.DataFrame(rows, columns=["cell_id", "model", "imgsz", "n_images", "duration_s", "measured_J", "predicted_J",
                                      "lo90_J", "hi90_J", "ape", "inside", "flagged", "new_model"])
    out.attrs["skipped"] = skipped
    return out


# ------------------------------------------------------------------ Mode L

def mode_l_cells(w: pd.DataFrame, cat_l: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Whole-cell, measured-segment and 1 s / 10 s block APE per lockbox cell and frozen rung (the metrics
    of `mode_l._score_cell`, plus the cell's model and image size). `w` holds QC-primary lockbox windows."""
    w = w.sort_values(["cell_id", "w_idx"], kind="stable").reset_index(drop=True)
    info = cat_l.set_index("cell_id")
    rungs = mlm.default_rungs()
    rows = []
    for name, model in models.items():
        p = np.asarray(model.predict(w), float)
        for cid, idx in w.groupby("cell_id", sort=True).indices.items():
            rows.append({"cell_id": cid, "rung": name, "exploratory": rungs[name].exploratory if name in rungs else False,
                         "model": info.loc[cid, "base_model"], "imgsz": int(info.loc[cid, "imgsz"]),
                         **mode_l._score_cell(w.iloc[idx], p[idx])})
    return pd.DataFrame(rows)


def summarise_mode_l(c: pd.DataFrame) -> pd.DataFrame:
    """Median APE per rung over YOLO-family lockbox models and over non-YOLO ones (resnet18, mbv3) separately."""
    c = c.assign(family=np.where(c["model"].str.startswith("yolo"), "YOLO family", "non-YOLO"))
    cols = ["ape", "ape_meas", "ape_1s", "ape_10s"]
    g = c.groupby(["family", "rung"], sort=False)
    out = g[cols].median().add_prefix("median_").join(g.size().rename("n_cells")).reset_index()
    out["exploratory"] = out["rung"].map(c.drop_duplicates("rung").set_index("rung")["exploratory"])
    return out


def probe_scores(scores: pd.DataFrame, der: pd.DataFrame, cat_l: pd.DataFrame, pred: pd.DataFrame,
                 energy_run_id: str | None) -> pd.DataFrame:
    """Probe mode on the lockbox: `mode_l.probe` over configurations with >= 2 primary counter cells, fed with the
    frozen Mode A primary energy prediction (the lockbox models are outside data.VISION_MODELS, so every lockbox
    cell is marked Mode A eligible here)."""
    prim = pred[(pred["role"] == "primary") & (pred["target"] == "energy_mJ")].set_index(["model", "imgsz"])["pred"]
    cat_p = cat_l.assign(mode_a_eligible=True)
    keys = {}
    for _, r in cat_p.iterrows():
        keys[deploy.mode_a_key_for_cell(r)] = float(prim.get((r["base_model"], int(r["imgsz"])), np.nan))
    ma = pd.DataFrame({"config_key": list(keys), "e_img_mJ": list(keys.values())})
    ma.attrs["run_ids"] = {"energy_mJ": energy_run_id}
    return mode_l.probe(None, der, cat_p, ma, 0, lomo=scores)


def summarise_probe(p: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return pd.DataFrame(columns=["rung", "n_pairs", "median_probe_ape", "median_mode_a_ape"])
    g = p.groupby("rung", sort=False)
    return g.agg(n_pairs=("ape", "size"), median_probe_ape=("ape", "median"),
                 median_mode_a_ape=("mode_a_ape", "median")).reset_index()


# ------------------------------------------------------------------ report

def _fmt(c, v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:.1%}" if ("ape" in c or "mdape" in c or "mape" in c or "bias" in c or "inside" in c) else f"{v:.4g}"
    return str(v).replace("|", "\\|")


def md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_No data._\n"
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(_fmt(c, v) for c, v in zip(df.columns, row)) + " |\n"
                          for row in df.itertuples(index=False))


def render(*, truth, mode_a, summary_a, calc, calc_summary, l_cells, l_summary, probe_summary, prov: dict) -> str:
    """lockbox.md: provenance, then Mode A, calculator, Mode L and probe tables."""
    out = ["# Lockbox results\n",
           "Scored once; any later change is exploratory. Nothing was fitted on lockbox data: Mode A predictions "
           "are the committed `lockbox_predictions.csv`, Mode L uses the frozen window models.\n",
           "## Provenance\n",
           f"- predictions file `{prov['predictions_csv']}` sha256 `{prov['predictions_sha256']}` "
           f"(written by `lockbox-predict` run `{prov.get('predict_run_id')}`)",
           f"- scoring run `{prov['run_id']}`; earlier scoring runs: {prov['n_earlier']}",
           f"- lockbox cells {prov['n_cells']} ({prov['n_primary']} QC-primary): {', '.join(prov['cell_ids'])}",
           "- frozen Mode A models (candidate / target: sha256, runlog run):"]
    out += [f"  - {k}: `{v[0]}` ({v[1]})" for k, v in prov["mode_a_models"].items()]
    out.append("- frozen Mode L models (rung: sha256, runlog run):")
    out += [f"  - {k}: `{v[0]}` ({v[1]})" for k, v in prov["mode_l_models"].items()]
    out += ["", "## Measured configurations\n", md(truth[["base_model", "imgsz", "n_cells", "cell_ids", *TARGET_COL]])]
    out += ["## Mode A: pooled\n", "`new models` are those with novelty above 0 (outside the 9 training models); "
            "resnet18 and mbv3 are training models, so their Mode A rows are in-sample.\n", md(summary_a),
            "## Mode A: per configuration\n",
            md(mode_a[["model", "imgsz", "target", "role", "candidate", "pred", "lo90", "hi90", "truth", "ape", "inside",
                       "flagged", "novelty"]])]
    out += ["## Deployment calculator (serve phase)\n",
            f"Frozen Mode A AUTO-final predictions with the measured serve duration and request count; idle power "
            f"from the training cells. Skipped: {calc.attrs.get('skipped')}.\n", md(calc_summary), md(calc)]
    out += ["## Mode L\n", "Whole-cell APE (`median_ape`), measured-segment APE, 1 s and 10 s blocks, medians per frozen "
            "rung. Rungs were trained on YOLO v8/11 cells only; resnet18 and mbv3 are non-YOLO extrapolations and "
            "are reported separately. Exploratory rungs are marked.\n", md(l_summary),
            "### Per cell\n", md(l_cells[["cell_id", "rung", "model", "imgsz", "E_true_J", "E_pred_J", "ape", "ape_meas",
                                          "ape_1s", "ape_10s"]])]
    out += ["## Probe\n", "Mean predicted power over a measured-phase run of cell j times j's latency, against the "
            "energy per image of cell k (ordered pairs of one configuration), per rung, against the frozen Mode A "
            "primary prediction.\n", md(probe_summary)]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ the command

def lockbox_windows(cat_l: pd.DataFrame, qc_l: pd.DataFrame) -> pd.DataFrame:
    """100 ms counter windows of the QC-primary lockbox cells that have counters (`windows.cell_windows`)."""
    ok = set(qc_l.loc[qc_l["primary"].astype(bool), "cell_id"])
    parts = []
    for _, r in cat_l[cat_l["cell_id"].isin(ok) & cat_l["has_counters"].astype(bool)].iterrows():
        d = windows.cell_windows(Path(r["path"]), data.read_meta(r["path"]))
        if d is not None and len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def require_lockbox(cat: pd.DataFrame) -> pd.DataFrame:
    """The lockbox rows of the catalogue; LockboxError if there are none."""
    cat_l = cat[cat["source"] == "lockbox"]
    if cat_l.empty:
        raise lb.LockboxError(f"no lockbox cells: {settings.data_root() / 'lockbox' / 'cells'} is missing or empty; "
                              f"lockbox-score runs after the lockbox cells are measured and copied there")
    return cat_l


def run_score(paths: ri.Paths, seed: int, *, cat: pd.DataFrame, der: pd.DataFrame, qc_train: pd.DataFrame,
              qc_l: pd.DataFrame, w: pd.DataFrame) -> dict:
    """Score everything once and write lockbox.md, the parquets under derived/lockbox/ and one runlog line.
    Fits nothing: Mode A predictions come from the committed predictions CSV (its model sha256 must equal the
    frozen pickles' runlog lines, and each pickle is checked), Mode L from the frozen rung pickles."""
    cat_l = require_lockbox(cat)
    cat_t = cat[cat["source"] != "lockbox"]
    check_design(cat_l)
    recs = ri.read_records(paths.runs_log)
    pred_path = Path(paths.out_dir) / lb.PREDICTIONS_CSV
    if not pred_path.is_file():
        raise lb.LockboxError(f"{pred_path} is missing; it must be written and committed before the lockbox is measured "
                              f"(`uv run python cli.py lockbox-predict ...`)")
    pred, pred_sha = pd.read_csv(pred_path), runlog.file_hash(pred_path)
    ma_models = {}
    for target, cand, _, rec in lb.mode_a_refs(recs, seed):
        lb.load_frozen(rec)                                           # sha256 check against runs.jsonl
        want = set(pred.loc[(pred["candidate"] == cand) & (pred["target"] == target), "model_sha256"])
        if want != {rec["sha256"]}:
            raise lb.LockboxError(f"{pred_path.name} was written with {cand}/{target} sha256 {sorted(want)}, but the "
                                  f"frozen model of runs.jsonl is {rec['sha256']}")
        ma_models[f"{cand} / {target}"] = (rec["sha256"], rec["run_id"])
    l_models, l_ids = {}, {}
    for name in mlm.default_rungs():
        rec = ri.latest_record(recs, command="final-mode-l", part="fit_final", rung=name, seed=seed)
        if rec is None:
            raise lb.LockboxError(f"no frozen Mode L model for rung {name} with seed {seed}; run "
                                  f"`uv run python cli.py final-mode-l --seed {seed}`")
        l_models[name], l_ids[name] = lb.load_frozen(rec), (rec["sha256"], rec["run_id"])
    run_id = uuid.uuid4().hex
    w = w[w["cell_id"].isin(qc_l.loc[qc_l["primary"].astype(bool), "cell_id"])] if len(w) else w
    truth = truth_table(cat_l, der, qc_l)
    scores = score_mode_a(pred, truth)
    sum_a = summarise_mode_a(scores)
    calc = calculator(cat_l, der, qc_l, pred, deploy.idle_table(cat_t, der, qc_train))
    calc_sum = (calc.groupby("new_model").agg(n=("ape", "size"), median_ape=("ape", "median"), inside_90=("inside", "mean"))
                .reset_index() if len(calc) else pd.DataFrame())
    l_cells = mode_l_cells(w, cat_l, l_models) if len(w) else pd.DataFrame(columns=["cell_id", "rung", "model", "ape"])
    l_sum = summarise_mode_l(l_cells) if len(l_cells) else pd.DataFrame()
    energy_ids = {c: r[1] for c, r in ma_models.items()}
    probe = (probe_scores(l_cells, der, cat_l, pred, next((v for k, v in energy_ids.items() if k.endswith("/ energy_mJ")), None))
             if len(l_cells) else pd.DataFrame())
    out_dir = settings.derived_dir() / "lockbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("truth", truth), ("mode_a_scores", scores), ("mode_a_summary", sum_a), ("calculator", calc),
                     ("mode_l_cells", l_cells), ("probe", probe)):
        df.to_parquet(out_dir / f"{run_id}_{name}.parquet", index=False)
    prim = qc_l["primary"].astype(bool)
    earlier = sum(1 for r in recs if r.get("command") == "lockbox-score")
    predict_rec = ri.latest_record(recs, command="lockbox-predict", predictions_sha256=pred_sha) or {}
    prov = {"predictions_csv": f"analytics/reports/{lb.PREDICTIONS_CSV}", "predictions_sha256": pred_sha,
            "predict_run_id": predict_rec.get("run_id"), "run_id": run_id, "n_earlier": earlier,
            "n_cells": len(cat_l), "n_primary": int(prim.sum()), "cell_ids": sorted(cat_l["cell_id"]),
            "mode_a_models": ma_models, "mode_l_models": l_ids}
    text = render(truth=truth, mode_a=scores, summary_a=sum_a, calc=calc, calc_summary=calc_sum, l_cells=l_cells,
                  l_summary=l_sum, probe_summary=summarise_probe(probe), prov=prov)
    md_path = Path(paths.out_dir) / "lockbox.md"
    md_path.write_text(text)
    rec = runlog.append({"run_id": run_id, "command": "lockbox-score", "seed": seed, "predictions_sha256": pred_sha,
                         "predictions_found_in_runlog": bool(predict_rec), "n_cells": len(cat_l), "n_primary": int(prim.sum()),
                         "n_earlier_scorings": earlier, "mode_a_models": ma_models, "mode_l_models": l_ids,
                         "mode_a_mdape": {f"{r.candidate}/{r.target}/{r.group}": r.mdape for r in sum_a.itertuples()},
                         "mode_l_median_ape": ({f"{r.family}/{r.rung}": r.median_ape for r in l_sum.itertuples()}
                                               if len(l_sum) else {}),
                         "report": str(md_path), "outputs": str(out_dir)}, paths.runs_log)
    return {"run_id": run_id, "truth": truth, "mode_a": scores, "summary_a": sum_a, "calculator": calc,
            "mode_l": l_cells, "mode_l_summary": l_sum, "probe": probe, "report": md_path, "rec": rec}
