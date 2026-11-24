"""The frozen models of the model cards, read from the run log (`mode-a-extras` final fits, `final-mode-a-secondary`,
`final-mode-l`): one row per model with its sha256, parameters, run id and whether the pickle on this machine still
hashes to that sha256. Read-only: nothing is fitted or written here."""
import json
from pathlib import Path

import pandas as pd

import lockbox
import mode_a_extras
import mode_l_models as mlm
import report_inputs as ri
import runlog

L_SECONDARY = ("L0", "L1", "L3-trees", "L4-trees")      # pre-declared secondary rungs (PROTOCOL.md, freeze amendment)


def file_status(rec: dict) -> str:
    """`matches` if the pickle of a runlog line hashes to the line's sha256, `DIFFERS` if not, `not on this machine` if absent."""
    p = lockbox.frozen_path(rec)
    if not Path(p).is_file():
        return "not on this machine"
    return "matches" if runlog.file_hash(p) == rec["sha256"] else "DIFFERS"


def _params(rec: dict) -> str:
    return json.dumps(rec.get("params"), sort_keys=True)


def mode_a_frozen(recs: list, seed: int) -> tuple:
    """(table, [what is missing]) of the final Mode A models of `seed`: the AUTO-final candidate of each target (role
    primary) and the secondary candidates (B3 for every target, M3 for energy), the latest line of each."""
    rows, missing = [], []
    for target in ri.TARGETS:
        prim = ri.latest_record(recs, command="mode-a-extras", part="fit_final", target=target, seed=seed)
        if prim is None:
            missing.append(f"primary / {target}")
        else:
            rows.append((target, prim["candidate"], "primary", prim))
        for cand in mode_a_extras.SECONDARY[target]:
            if prim is not None and prim["candidate"] == cand:
                continue
            rec = ri.latest_record(recs, command="final-mode-a-secondary", part="fit_final", target=target, candidate=cand, seed=seed)
            if rec is None:
                missing.append(f"{cand} / {target}")
            else:
                rows.append((target, cand, "secondary", rec))
    t = pd.DataFrame([{"target": tg, "candidate": c, "role": role, "params": _params(r), "sha256": r["sha256"], "run_id": r["run_id"],
                       "command": r["command"], "trained on": f"{r.get('n_configs')} configurations",
                       "AUTO chose (folds)": json.dumps(r.get("auto_counts"), sort_keys=True) if role == "primary" else "",
                       "S-LOMO run": r.get("source_run_id", ""), "file": file_status(r)} for tg, c, role, r in rows])
    return t, missing


def mode_l_frozen(recs: list, seed: int) -> tuple:
    """(table, [missing rungs]) of the `final-mode-l` fits of `seed`, the latest line of each of the eight R1 rungs
    (the R2 rungs are not part of the frozen set)."""
    rows, missing = [], []
    for name, rung in ((n, r) for n, r in mlm.default_rungs().items() if n in mlm.R1_RUNGS):
        rec = ri.latest_record(recs, command="final-mode-l", part="fit_final", rung=name, seed=seed)
        if rec is None:
            missing.append(name)
            continue
        role = "exploratory" if rung.exploratory else ("secondary (pre-declared)" if name in L_SECONDARY else "reported")
        rows.append({"rung": name, "role": role, "params": _params(rec), "sha256": rec["sha256"], "run_id": rec["run_id"],
                     "cells": rec.get("n_cells"), "windows": rec.get("n_windows"), "file": file_status(rec)})
    return pd.DataFrame(rows), missing


def missing_text(missing: list) -> str:
    return "" if not missing else "not frozen: " + ", ".join(missing) + ". "
