"""Synthetic frozen models, S-LOMO runs, static CSVs and runs.jsonl for the lockbox tests, all under a tmp root.
`build(root)` sets ANALYTICS_DATA to root/data while it fits, so every pickle lands under root/data/derived."""
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import mode_a
import mode_a_extras as ex
import mode_l
import mode_l_models as mlm
import report_inputs as ri
import runlog
from test_mode_a import SMALL_GRIDS, lomo, make_dataset
from test_mode_l import make_windows

MASTER = 20261122
TRAIN = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "resnet18", "mbv3"]
CONFIGS = ["yolo12n:640", "yolo12n:320", "yolo11n-seg:640", "resnet18:224", "mbv3:224"]
SCALE = {"yolo12n_640": 1.0, "yolo12n_320": 0.25, "yolo11n-seg_640": 1.3, "resnet18_224": 0.6, "mbv3_224": 0.3}
ROWS = [("Conv2d", 0.8e4, 8e7), ("BatchNorm2d", 1e3, 1e6), ("SiLU", 0, None), ("Linear", 2e3, 4e6), ("MaxPool2d", 0, None)]


def layer_csv(path, scale: float) -> None:
    rows = [{"path": f"m.l{i}", "class": c, "leaf": True, "params": int(p * scale), "macs": None if m is None else int(m * scale),
             "bytes_moved": int(4 * p * scale + 3e5 * scale), "count": 1} for i, (c, p, m) in enumerate(ROWS)]
    pd.DataFrame(rows).to_csv(path, index=False)


def static_dir(root) -> Path:
    d = Path(root) / "static"
    d.mkdir(parents=True, exist_ok=True)
    for k, s in SCALE.items():
        layer_csv(d / f"{k}.csv", s)
    return d


def build(root) -> SimpleNamespace:
    root = Path(root)
    old = os.environ.get("ANALYTICS_DATA")
    os.environ["ANALYTICS_DATA"] = str(root / "data")
    try:
        t, desc = make_dataset(TRAIN, clocks=(1500, 2400), threads=(4,))
        t["rig"] = "nov2026"
        paths = ri.Paths(root / "runs.jsonl", root / "mode_a", root / "mode_l", root / "extras", root / "t.csv", root / "q.csv",
                         root / "d.csv", root / "out", root / "out" / "fig")
        paths.mode_a_dir.mkdir(parents=True)
        paths.out_dir.mkdir(parents=True)
        for target in ri.TARGETS:
            res = mode_a.run_split(t, lomo(t), target, MASTER, desc, grids=SMALL_GRIDS)
            res.to_parquet(paths.mode_a_dir / f"{target}.parquet")
            runlog.append({"run_id": target, "command": "mode-a", "split": "S-LOMO", "target": target, "seed": MASTER}, paths.runs_log)
            _, info = mode_a.fit_final(t, "M1Ridge", target, MASTER, desc, SMALL_GRIDS)
            runlog.append({"command": "mode-a-extras", "part": "fit_final", "seed": MASTER, "target": target, "candidate": "M1Ridge",
                           "params": info["params"], "path": info["path"], "sha256": info["sha256"]}, paths.runs_log)
        ex.run_secondary(t, desc, MASTER, paths, grids=SMALL_GRIDS)
        # Mode L: first grid point of every rung on three YOLO-named models
        w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
        qc = pd.DataFrame({"cell_id": cat["cell_id"], "primary": True})
        rungs = mlm.first_point(mlm.default_rungs())
        for name in rungs:
            _, info = mode_l.fit_final(w, cat.assign(source="v2"), qc, name, MASTER, rungs)
            runlog.append({"command": "final-mode-l", "part": "fit_final", "seed": MASTER, "rung": name, **info}, paths.runs_log)
        return SimpleNamespace(paths=paths, t=t, desc=desc, static_dir=static_dir(root), data=root / "data")
    finally:
        if old is None:
            os.environ.pop("ANALYTICS_DATA", None)
        else:
            os.environ["ANALYTICS_DATA"] = old
