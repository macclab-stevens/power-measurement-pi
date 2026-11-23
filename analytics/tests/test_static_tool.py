"""tools/static_features.py (tools env, run through subprocess) against the Pi's own layer tables.

The tool writes a CSV in the layer_table.csv format; `static.features_from_layer_table` reads both and
every feature key must agree (exact for counts, relative 1e-9 for floats). This is what lets the
lockbox score a brand-new model from its NUC-built layer table. Skipped when the tools env, the
weights or the real reference cells are absent."""
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import static

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
CELLS = Path.home() / "pi-power-data"
CASES = [
    pytest.param("../yolo11n.pt", 640, CELLS / "v2/cells/v1ref_yolo11n_2400_r00/layer_table.csv",
                 REPO / "yolo11n.pt", id="yolo11n-640"),
    pytest.param("torchvision:resnet18", 224, CELLS / "v1/cells/resnet18_1500_r00/layer_table.csv",
                 None, id="resnet18-224"),
]
INT_KEYS = {"depth", "n_leaf"} | {f"n_{g}" for g in static.GROUPS}
LAST = {}


def _both(a: dict, b: dict) -> pd.DataFrame:
    rows = []
    for k in a:
        x, y = a[k], b[k]
        if k in INT_KEYS:
            ok = x == y
        else:
            ok = (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-9, abs=0.0)
        rows.append({"feature": k, "tool": x, "pi_cell": y, "equal": bool(ok)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model,imgsz,pi_csv,weights", CASES)
def test_tool_features_equal_pi_layer_table(model, imgsz, pi_csv, weights, tmp_path, capsys):
    if shutil.which("uv") is None or not (TOOLS / ".venv").is_dir():
        pytest.skip("tools/.venv or uv is absent")
    if not pi_csv.is_file() or (weights is not None and not weights.is_file()):
        pytest.skip(f"real reference cell or weights absent: {pi_csv}")
    out = tmp_path / "tool_layer_table.csv"
    proc = subprocess.run(["uv", "run", "python", "static_features.py", "--model", model, "--imgsz", str(imgsz),
                           "--out", str(out)], cwd=TOOLS, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = static.features_from_layer_table(pd.read_csv(out))
    want = static.features_from_layer_table(pd.read_csv(pi_csv))
    assert list(got) == list(want)
    cmp = _both(got, want)
    with capsys.disabled():
        print(f"\n{model} @ {imgsz}: tool CSV vs {pi_csv.parent.name}\n{cmp.to_string(index=False)}")
    assert cmp["equal"].all(), cmp[~cmp["equal"]].to_string(index=False)
