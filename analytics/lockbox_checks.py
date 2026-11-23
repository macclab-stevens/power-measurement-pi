"""Lockbox integrity checks that need no model: the scoring code is the code of `lockbox-predict`, the measured models
are the models that were predicted, and the idle table is the one the serve predictions used. Nothing is fitted."""
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import runlog
import settings
import static


class LockboxError(RuntimeError):
    """A lockbox input is missing or inconsistent; the message says what to do."""


EXACT = ("macs", "params")          # compared exactly; every other static feature to a relative 1e-9
REL_TOL = 1e-9
ALLOW_FLAG = "--allow-code-change"


# ------------------------------------------------------------------ code pin

def config_sha256() -> str:
    return runlog.file_hash(settings.HERE / "config.toml")


def code_pin() -> dict:
    """What `lockbox-predict` records in the frozen-models manifest: `git rev-parse HEAD`, the sha256 of every tracked
    analytics/*.py file, the sha256 of config.toml and the combined `code_sha256` (also on every runlog line)."""
    try:
        commit = runlog._git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    files = runlog.code_files()
    return {"commit": commit, "files": files, "config_toml_sha256": config_sha256(), "code_sha256": runlog.code_sha256(files)}


def commit_relation(old: str) -> str:
    """How HEAD relates to the commit recorded at `lockbox-predict`: same commit, descendant, not an ancestor, or unknown."""
    try:
        head = runlog._git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if old in (None, "", "unknown"):
        return "unknown"
    if head == old:
        return "same commit"
    r = subprocess.run(["git", "merge-base", "--is-ancestor", old, "HEAD"], cwd=settings.HERE, capture_output=True)
    return {0: "descendant of the recorded commit", 1: "not an ancestor of HEAD"}.get(r.returncode, "unknown")


def check_code(pin: dict | None, allow: bool) -> dict:
    """Compare the manifest's code pin with the code now. Returns {"changed_files": [...], "config_changed": bool, "commit": str,
    "relation": str}; a mismatch (a module changed, added or removed, config.toml changed, or the recorded commit is not an
    ancestor of HEAD) raises LockboxError unless `allow`, in which case the caller labels its outputs exploratory."""
    if not pin or "files" not in pin:
        raise LockboxError("the frozen-models manifest has no code pin (`code`), so the scoring code cannot be verified; it is written by "
                           f"`lockbox-predict`. {ALLOW_FLAG} scores anyway and labels the result exploratory")
    now = runlog.code_files()
    old = pin["files"]
    changed = sorted(k for k in set(now) | set(old) if now.get(k) != old.get(k))
    cfg = config_sha256() != pin.get("config_toml_sha256")
    rel = commit_relation(pin.get("commit"))
    out = {"changed_files": changed + (["analytics/config.toml"] if cfg else []), "config_changed": cfg, "commit": pin.get("commit"), "relation": rel}
    bad = bool(changed) or cfg or rel == "not an ancestor of HEAD"
    if bad and not allow:
        why = []
        if out["changed_files"]:
            why.append("changed since `lockbox-predict`: " + ", ".join(out["changed_files"]))
        if rel == "not an ancestor of HEAD":
            why.append(f"the commit recorded at `lockbox-predict` ({pin.get('commit')}) is not an ancestor of HEAD")
        raise LockboxError("the scoring code is not the code of `lockbox-predict`; " + "; ".join(why) + f". {ALLOW_FLAG} scores anyway; "
                           "lockbox.md and the runlog line are then labelled exploratory")
    return out


# ------------------------------------------------------------------ idle table

def idle_sha256(idle: pd.DataFrame) -> str:
    """sha256 of the idle table (rows sorted by rig, cpu_key, fan_mode; floats to 12 significant digits)."""
    t = idle.sort_values(["rig", "cpu_key", "fan_mode"]).reset_index(drop=True)
    return hashlib.sha256(t.to_csv(index=False, float_format="%.12g").encode()).hexdigest()


# ------------------------------------------------------------------ measured model against the predicted one

def feature_mismatches(want: dict, got: dict) -> list:
    """[(feature, static value, measured value)] where they differ: macs and params exactly, the rest to a relative 1e-9 (NaN equals NaN)."""
    out = []
    for k in sorted(set(want) | set(got)):
        a, b = want.get(k, np.nan), got.get(k, np.nan)
        same = (a == b) if k in EXACT else bool(np.isclose(a, b, rtol=REL_TOL, atol=0.0, equal_nan=True))
        if not same:
            out.append((k, a, b))
    return out


def check_static(cat_l: pd.DataFrame, qc_l: pd.DataFrame, pred: pd.DataFrame, static_dir) -> dict:
    """For every QC-primary lockbox cell whose (model, imgsz) was predicted: the static CSV in `static_dir` must be the file whose
    sha256 the predictions recorded, and the cell's own `layer_table.csv` (fp32, batch 1: `check_design`) must give the same
    `static.features_from_layer_table` as that CSV. Raises LockboxError naming the cell and the features that differ.
    Returns {"checked": n cells, "configs": n configurations, "unpredicted": [cell ids of configurations nobody predicted]}."""
    rec = pred.drop_duplicates(["model", "imgsz"]).set_index(["model", "imgsz"])["static_csv_sha256"]
    ok = set(qc_l.loc[qc_l["primary"].astype(bool), "cell_id"])
    cells = cat_l[cat_l["cell_id"].isin(ok)]
    feats, bad, checked, unpredicted = {}, [], 0, []
    for _, r in cells.sort_values("cell_id").iterrows():
        key = (r["base_model"], int(r["imgsz"]))
        if key not in rec.index:
            unpredicted.append(r["cell_id"])
            continue
        if key not in feats:
            csv = Path(static_dir) / f"{key[0]}_{key[1]}.csv"
            if not csv.is_file():
                raise LockboxError(f"static features file {csv} is missing; the predictions recorded its sha256 {rec[key]}")
            got = runlog.file_hash(csv)
            if got != rec[key]:
                raise LockboxError(f"static features file {csv.name} has sha256 {got}, but the predictions were made with {rec[key]}")
            feats[key] = static.features_from_layer_table(pd.read_csv(csv))
        lt = Path(r["path"]) / "layer_table.csv"
        if not lt.is_file():
            raise LockboxError(f"lockbox cell {r['cell_id']} has no layer_table.csv ({lt}), so the measured model cannot be compared with "
                               f"the one that was predicted")
        diff = feature_mismatches(feats[key], static.features_from_layer_table(pd.read_csv(lt)))
        if diff:
            bad.append(f"lockbox cell {r['cell_id']} ({key[0]}:{key[1]}): " + ", ".join(f"{k} (predicted from {a:.10g}, measured {b:.10g})" for k, a, b in diff))
        checked += 1
    if bad:
        raise LockboxError("the measured model is not the one that was predicted; " + "; ".join(bad))
    return {"checked": checked, "configs": len(feats), "unpredicted": unpredicted}
