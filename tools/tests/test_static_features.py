"""tools/static_features.py: the NUC-built layer table equals the Pi's for the same model."""
import csv
from pathlib import Path

import pytest

import static_features as sf

REPO = Path(__file__).resolve().parents[2]
REAL = Path.home() / "pi-power-data/v2/cells/v1ref_yolo11n_2400_r00/layer_table.csv"
COLUMNS = ["path", "config_id", "class", "config_json", "input_shapes", "input_dtypes", "out_shapes",
           "callsite", "mode", "leaf", "params", "macs", "bytes_moved", "arith_intensity", "t_eff",
           "seq_len", "count", "lat_mean_ms", "lat_median_ms", "lat_std_ms"]


def _leaf_sums(path):
    """(sum of macs x multiplicity-free leaf rows, params over unique leaf paths)."""
    rows = [r for r in csv.DictReader(open(path, newline="")) if r["leaf"] == "True"]
    pos = [int(r["count"]) for r in rows if int(r["count"]) > 0]
    mn = min(pos)
    macs = sum(int(float(r["macs"] or 0)) * round(int(r["count"]) / mn) for r in rows)
    params, seen = 0, set()
    for r in rows:
        if r["path"] not in seen:
            seen.add(r["path"])
            params += int(r["params"])
    return macs, params


def test_columns_match_layer_table_format():
    assert sf.COLUMNS == COLUMNS


@pytest.mark.filterwarnings("ignore")
def test_resnet18_tiny_runs(tmp_path):
    out = tmp_path / "r18.csv"
    sf.main(["--model", "torchvision:resnet18", "--imgsz", "64", "--out", str(out)])
    rows = list(csv.DictReader(open(out, newline="")))
    assert list(rows[0]) == COLUMNS
    assert any(r["class"] == "Conv2d" and r["leaf"] == "True" for r in rows)
    assert {int(r["count"]) for r in rows if r["leaf"] == "True"} <= {2, 4, 6}


@pytest.mark.filterwarnings("ignore")
def test_yolo11n_640_equals_pi_cell(tmp_path):
    pt = REPO / "yolo11n.pt"
    if not pt.is_file() or not REAL.is_file():
        pytest.skip("yolo11n.pt or the real v1ref_yolo11n cell is absent")
    out = tmp_path / "yolo11n.csv"
    sf.main(["--model", str(pt), "--imgsz", "640", "--out", str(out)])
    assert _leaf_sums(out) == _leaf_sums(REAL)
