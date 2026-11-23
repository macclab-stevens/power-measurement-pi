"""Lockbox, part 1: frozen models and the a priori predictions written before any lockbox cell exists.

The lockbox (spec 9) is a few brand-new models measured AFTER every model is frozen. This module
  * finds the frozen Mode A models in runs.jsonl (AUTO-final from `mode-a-extras`, secondary from
    `final-mode-a-secondary`) and refuses a pickle whose sha256 differs from its runlog line;
  * builds one Mode A feature row per `model:imgsz` from the layer-table CSV that `tools/static_features.py`
    writes (same derived columns as `tables.mode_a_table`);
  * predicts energy, latency and power with every frozen candidate, with a 90 % interval from that
    candidate's S-LOMO held-out log residuals and an applicability flag from the S-LOMO novelty threshold.
Nothing here fits anything. Scoring is in `lockbox_score`."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import deploy
import mode_a
import mode_a_extras
import mode_a_models as mm
import mode_l_models as mlm
import report_inputs as ri
import runlog
import static
import tables

DEFAULT_CONFIGS = ["yolo12n:640", "yolo12n:320", "yolo11n-seg:640", "resnet18:224", "mbv3:224"]
FIXED = {"precision": "fp32", "batch": 1, "threads": 4, "cpu_mhz": 2400, "rig": "nov2026"}   # every lockbox cell
SECONDARY = mode_a_extras.SECONDARY
PRED_COLS = ["config_key", "model", "imgsz", "candidate", "role", "target", "pred", "lo90", "hi90", "novelty",
             "threshold", "flagged", "model_sha256", "static_csv_sha256"]
SERVE_REQUESTS = 10
SERVE_GAP_S = 0.2
PREDICTIONS_CSV = "lockbox_predictions.csv"
SERVE_CSV = "lockbox_serve_predictions.csv"
MANIFEST = "lockbox_frozen_models.json"
FINAL_CMD = "uv run python cli.py mode-a-extras"
SECONDARY_CMD = "uv run python cli.py final-mode-a-secondary"
LOMO_CMD = "uv run python cli.py mode-l"
FINAL_L_CMD = "uv run python cli.py final-mode-l"


class LockboxError(RuntimeError):
    """A lockbox input is missing or inconsistent; the message says what to do."""


# ------------------------------------------------------------------ configurations and static features

def parse_config(spec: str) -> tuple:
    """"yolo12n:640" -> ("yolo12n", 640)."""
    model, sep, size = str(spec).rpartition(":")
    if not sep or not model or not size.isdigit():
        raise LockboxError(f"bad --config {spec!r}; expected <model>:<imgsz>, e.g. yolo12n:640")
    return model, int(size)


def static_csv_path(static_dir, model: str, imgsz: int) -> Path:
    return Path(static_dir) / f"{model}_{imgsz}.csv"


def feature_rows(configs: list, static_dir) -> tuple:
    """(Mode A feature rows, {config: sha256 of its static CSV}) for `configs` ((model, imgsz) pairs).
    A missing CSV raises LockboxError naming the file and the tool command that makes it."""
    rows, shas = [], {}
    for model, imgsz in configs:
        p = static_csv_path(static_dir, model, imgsz)
        if not p.is_file():
            raise LockboxError(f"static features file {p} is missing; make it with: cd tools && uv run python "
                               f"static_features.py --model <weights or torchvision:name of {model}> --imgsz {imgsz} --out {p}")
        feats = static.features_from_layer_table(pd.read_csv(p))
        rows.append({"base_model": model, "variant": model, "imgsz": imgsz, "quant": "fp32", "n_quantized_modules": 0,
                     **FIXED, **feats})
        shas[(model, imgsz)] = runlog.file_hash(p)
    t = tables.add_derived_columns(pd.DataFrame(rows))
    t.insert(0, "config_key", [tables.config_key(r) for _, r in t.iterrows()])
    return t, shas


# ------------------------------------------------------------------ frozen models

def frozen_path(rec: dict) -> Path:
    """The pickle a runlog line names (falls back to the same file name under derived/models/)."""
    import settings
    p = Path(rec["path"])
    return p if p.is_file() else settings.derived_dir() / "models" / p.name


def load_frozen(rec: dict):
    """Unpickle the model of a runlog line after checking the file's sha256 against the line."""
    p = frozen_path(rec)
    if not p.is_file():
        raise LockboxError(f"frozen model {p} is missing")
    got = runlog.file_hash(p)
    if got != rec["sha256"]:
        raise LockboxError(f"frozen model {p} has sha256 {got}, but runs.jsonl run {rec.get('run_id')} recorded "
                           f"{rec['sha256']}; it was changed after it was frozen")
    return pickle.loads(p.read_bytes())


def mode_a_refs(recs: list, seed: int) -> list:
    """[(target, candidate, role, runlog record)]: the AUTO-final candidate of each target (latest
    `mode-a-extras` fit of `seed`) first, then the secondary candidates (latest `final-mode-a-secondary`
    fit of each); a secondary equal to the primary of its target is listed once."""
    out = []
    for target in ri.TARGETS:
        prim = ri.latest_record(recs, command="mode-a-extras", part="fit_final", target=target, seed=seed)
        if prim is None:
            raise LockboxError(f"no final Mode A fit for target {target} with seed {seed}; run `{FINAL_CMD} --seed {seed}`")
        out.append((target, prim["candidate"], "primary", prim))
        for cand in SECONDARY[target]:
            if cand == prim["candidate"]:
                continue
            rec = ri.latest_record(recs, command="final-mode-a-secondary", part="fit_final", target=target,
                                   candidate=cand, seed=seed)
            if rec is None:
                raise LockboxError(f"no frozen {cand} for target {target} with seed {seed}; run `{SECONDARY_CMD} --seed {seed}`")
            out.append((target, cand, "secondary", rec))
    return out


def mode_l_refs(recs: list, seed: int) -> dict:
    """{rung: latest `final-mode-l` runlog record of `seed`} for all eight rungs; a missing rung raises
    LockboxError naming the command."""
    out = {}
    for name in mlm.default_rungs():
        rec = ri.latest_record(recs, command="final-mode-l", part="fit_final", rung=name, seed=seed)
        if rec is None:
            raise LockboxError(f"no frozen Mode L model for rung {name} with seed {seed}; run `{FINAL_L_CMD} --seed {seed}`")
        out[name] = rec
    return out


def check_sha(rec: dict) -> str:
    """The pickle's sha256 if it equals the runlog line's, else LockboxError (the file is not unpickled)."""
    p = frozen_path(rec)
    if not p.is_file():
        raise LockboxError(f"frozen model {p} is missing")
    got = runlog.file_hash(p)
    if got != rec["sha256"]:
        raise LockboxError(f"frozen model {p} has sha256 {got}, but runs.jsonl run {rec.get('run_id')} recorded "
                           f"{rec['sha256']}; it was changed after it was frozen")
    return got


def mode_l_frozen(l_refs: dict) -> dict:
    return {r: {"sha256": rec["sha256"], "run_id": rec["run_id"]} for r, rec in l_refs.items()}


def manifest(seed: int, pred_sha: str, refs: list, l_refs: dict) -> dict:
    """Content of reports/lockbox_frozen_models.json: every frozen model with its sha256 and the predictions CSV's."""
    return {"seed": seed, "predictions_csv": PREDICTIONS_CSV, "predictions_sha256": pred_sha,
            "mode_a": {f"{c}/{t}": {"role": role, "sha256": r["sha256"], "run_id": r["run_id"]} for t, c, role, r in refs},
            "mode_l": mode_l_frozen(l_refs)}


def refuse_if_lockbox_cells(what: str) -> None:
    """`what` fits or freezes a model; it must not run once lockbox cells exist under ANALYTICS_DATA."""
    if lockbox_cells_exist():
        raise LockboxError(f"lockbox cells already exist; `{what}` would not be a pre-lockbox freeze and is refused "
                           f"(models are frozen before the lockbox is measured)")


# ------------------------------------------------------------------ intervals, novelty, threshold

def interval_quantiles(slomo: pd.DataFrame, candidate: str) -> tuple:
    """(q05, q95) of the candidate's S-LOMO held-out log residuals log(y_true) - log(y_pred), all rows."""
    r = slomo[slomo["candidate"] == candidate]
    resid = np.log(r["y_true"].to_numpy(float)) - np.log(r["y_pred"].to_numpy(float))
    resid = resid[np.isfinite(resid)]
    if resid.size < 2:
        raise LockboxError(f"too few S-LOMO residuals for candidate {candidate}")
    q5, q95 = np.quantile(resid, [0.05, 0.95])
    return float(q5), float(q95)


def threshold_from_slomo(slomo: pd.DataFrame) -> float:
    """Largest S-LOMO outer-fold novelty whose AUTO fold MdAPE is below 15 %; 0.0 if there is none."""
    a = slomo[slomo["candidate"] == mode_a.AUTO]
    ok = []
    for _, g in a.groupby("fold"):
        m = float(np.median(np.abs(g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)) / g["y_true"].to_numpy(float)))
        if m < mode_a.APPLICABLE_MDAPE:
            ok.append(float(g["novelty"].iloc[0]))
    return max(ok) if ok else 0.0


def descriptor(desc_train: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """`desc_train` plus the descriptor of every model of `feats` that is not in it (static.model_descriptor)."""
    new = feats[~feats["base_model"].isin(desc_train.index)]
    if new.empty:
        return desc_train
    st = new.drop_duplicates(["base_model", "imgsz"]).set_index(["base_model", "imgsz"])
    return pd.concat([desc_train, static.model_descriptor(st)[desc_train.columns]])


def novelty_by_model(desc: pd.DataFrame, models, train_models) -> dict:
    """Novelty of each model against the training models (0 for a training model)."""
    tm = sorted(set(train_models))
    return {m: 0.0 if m in tm else mm.novelty_distance(desc, m, tm) for m in sorted(set(models))}


# ------------------------------------------------------------------ the predictions

def predict_table(feats: pd.DataFrame, shas: dict, refs: list, slomo: dict, desc_train: pd.DataFrame,
                  train_models, models: dict | None = None) -> pd.DataFrame:
    """One row per configuration x candidate x target (columns PRED_COLS). `refs` is `mode_a_refs`,
    `slomo` {target: S-LOMO prediction frame}; `models` {(target, candidate): model} overrides loading the
    pickles (the sha256 of the file is still the one recorded)."""
    nov = novelty_by_model(descriptor(desc_train, feats), feats["base_model"], train_models)
    rows = []
    for target, cand, role, rec in refs:
        model = (models or {}).get((target, cand))
        if model is None:
            model = load_frozen(rec)
        pred = np.asarray(model.predict(feats), float)
        q5, q95 = interval_quantiles(slomo[target], cand)
        thr = threshold_from_slomo(slomo[target])
        for i, r in enumerate(feats.itertuples()):
            n = nov[r.base_model]
            rows.append({"config_key": r.config_key, "model": r.base_model, "imgsz": int(r.imgsz), "candidate": cand,
                         "role": role, "target": target, "pred": float(pred[i]), "lo90": float(pred[i] * np.exp(q5)),
                         "hi90": float(pred[i] * np.exp(q95)), "novelty": n, "threshold": thr, "flagged": bool(n > thr),
                         "model_sha256": rec["sha256"], "static_csv_sha256": shas[(r.base_model, int(r.imgsz))]})
    return pd.DataFrame(rows, columns=PRED_COLS)


def serve_table(pred: pd.DataFrame, idle: pd.DataFrame) -> pd.DataFrame:
    """Calculator prediction of the default serve phase (10 requests, 200 ms gap, so duration =
    10 x (t_img + 0.2 s)) per configuration from the primary energy and latency predictions. For
    documentation only: scoring uses the measured duration."""
    p = pred[pred["role"] == "primary"]
    e, t = p[p["target"] == "energy_mJ"].set_index("config_key"), p[p["target"] == "latency_ms"].set_index("config_key")
    rows = []
    for key in e.index:
        t_s = float(t.loc[key, "pred"]) / 1000.0
        dur = SERVE_REQUESTS * (t_s + SERVE_GAP_S)
        scen = {"cpu_key": str(FIXED["cpu_mhz"]), "fan_mode": "full", "rig": FIXED["rig"], "n_images": SERVE_REQUESTS,
                "duration_s": dur}
        out = {k: deploy.predict(scen, float(e.loc[key, c]), float(t.loc[key, "pred"]), idle, None, None)["E_J"]
               for k, c in (("E_J", "pred"), ("E_J_lo90", "lo90"), ("E_J_hi90", "hi90"))}
        rows.append({"config_key": key, "model": e.loc[key, "model"], "imgsz": int(e.loc[key, "imgsz"]),
                     "e_img_mJ": float(e.loc[key, "pred"]), "t_img_ms": float(t.loc[key, "pred"]),
                     "n_requests": SERVE_REQUESTS, "duration_s": dur, **out, "flagged": bool(e.loc[key, "flagged"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the command

def slomo_runs(paths: ri.Paths, recs: list, seed: int) -> dict:
    """{target: RunRef} of the seed-`seed` S-LOMO runs (strict: no other seed stands in)."""
    out = {}
    for target in ri.TARGETS:
        rec = ri.latest(recs, paths.mode_a_dir, command="mode-a", split="S-LOMO", target=target, seed=seed)
        if rec is None:
            raise LockboxError(f"no S-LOMO Mode A run with seed {seed} for target {target}; run "
                               f"`{ri.MODE_A_CMD} --split S-LOMO --target {target} --seed {seed}`")
        out[target] = ri.ref_of(rec, paths.mode_a_dir)
    return out


def lockbox_cells_exist() -> bool:
    import settings
    d = settings.data_root() / "lockbox" / "cells"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def run_predict(paths: ri.Paths, configs: list, static_dir, seed: int, *, t: pd.DataFrame, desc_train: pd.DataFrame,
                idle: pd.DataFrame, force: bool = False) -> dict:
    """Write reports/lockbox_predictions.csv, reports/lockbox_frozen_models.json (every frozen Mode A and Mode L
    model with its sha256, plus the predictions' sha256), the documentation-only serve prediction and one runlog
    line (which also records `mode_l_frozen`). All eight Mode L rungs must be frozen (`final-mode-l`).
    `t` is the training configuration table (its base models are the training models), `desc_train` the descriptor of those models, `idle` the idle table of the training cells. Refuses to run when any
    lockbox cell already exists (the predictions must precede the data) or when the file exists (unless
    `force`)."""
    out = Path(paths.out_dir) / PREDICTIONS_CSV
    if lockbox_cells_exist():
        raise LockboxError("lockbox cells already exist; the predictions must be written (and committed) before "
                           "any lockbox cell, and the committed lockbox_predictions.csv is the record")
    if out.exists() and not force:
        raise LockboxError(f"{out} already exists; it is the committed record of the predictions (--force to overwrite)")
    pairs = [parse_config(c) for c in configs]
    if len(set(pairs)) != len(pairs):
        raise LockboxError("duplicate --config")
    recs = ri.read_records(paths.runs_log)
    refs = mode_a_refs(recs, seed)
    l_refs = mode_l_refs(recs, seed)
    for rec in l_refs.values():
        check_sha(rec)
    slomo = slomo_runs(paths, recs, seed)
    feats, shas = feature_rows(pairs, static_dir)
    pred = predict_table(feats, shas, refs, {k: v.df for k, v in slomo.items()}, desc_train, set(t["base_model"]))
    serve = serve_table(pred, idle)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    serve_out = out.with_name(SERVE_CSV)
    serve.to_csv(serve_out, index=False)
    pred_sha = runlog.file_hash(out)
    man_out = out.with_name(MANIFEST)
    man_out.write_text(json.dumps(manifest(seed, pred_sha, refs, l_refs), indent=2, sort_keys=True) + "\n")
    rec = runlog.append({"command": "lockbox-predict", "seed": seed, "configs": configs, "n_rows": len(pred),
                         "predictions_csv": str(out), "predictions_sha256": pred_sha,
                         "mode_l_frozen": mode_l_frozen(l_refs), "manifest_sha256": runlog.file_hash(man_out),
                         "serve_csv_sha256": runlog.file_hash(serve_out),
                         "frozen": {f"{c}/{tg}": r["sha256"] for tg, c, _, r in refs},
                         "frozen_run_ids": {f"{c}/{tg}": r["run_id"] for tg, c, _, r in refs},
                         "slomo_run_ids": {k: v.run_id for k, v in slomo.items()},
                         "static_csv_sha256": {f"{m}:{s}": h for (m, s), h in shas.items()},
                         "n_flagged_primary": int(pred[pred["role"] == "primary"].groupby("config_key")["flagged"].any().sum())},
                        paths.runs_log)
    return {"pred": pred, "serve": serve, "path": out, "manifest": man_out, "rec": rec}
