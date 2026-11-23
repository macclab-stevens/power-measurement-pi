"""Mode A configuration table: synthetic merge/drop rules and the real data."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data
import derive
import qc
import static
import tables

FEATS = {"macs": 1e9, "params": 2e6, "weight_bytes": 8e6, "act_bytes": 5e7, "depth": 10, "n_leaf": 50,
         **{f"n_{g}": 1 for g in static.GROUPS}, **{f"frac_macs_{g}": 1 / 6 for g in static.GROUPS}}


def _cat_row(cid, **kw):
    r = {"cell_id": cid, "base_model": "yolo11n", "variant": "yolo11n", "precision": "fp32",
         "quant": "fp32", "n_quantized_modules": 0, "imgsz": 640, "batch": 1, "threads": 4,
         "cpu_mhz": 2400, "rig": "nov2026", "family": "A", "mode_a_eligible": True}
    r.update(kw)
    return r


def _der(cid, e, lat, mode="forward"):
    return {"cell_id": cid, "mode": mode, "energy_mJ_per_image": e, "latency_ms_per_image": lat,
            "mean_P_W": 5.0, "dyn_energy_mJ_per_image": e * 0.6}


def _qc(cid, primary=True):
    return {"cell_id": cid, "include": True, "primary": primary, "rules": ""}


def _st(models=(("yolo11n", 640),)):
    idx = pd.MultiIndex.from_tuples(list(models), names=["base_model", "imgsz"])
    return pd.DataFrame([{"cell_id": "x", **FEATS}] * len(models), index=idx)


def _build(cat_rows, der_rows, qc_rows, st=None):
    return tables.mode_a_table(pd.DataFrame(cat_rows), pd.DataFrame(der_rows), pd.DataFrame(qc_rows),
                               st if st is not None else _st())


def test_config_key_and_cols():
    assert tables.CONFIG_COLS == ["base_model", "variant", "precision", "imgsz", "batch", "threads",
                                  "cpu_mhz", "rig"]
    row = _cat_row("a")
    assert tables.config_key(row) == "yolo11n|yolo11n|fp32|640|1|4|2400|nov2026"


def test_merge_by_input_source_and_stats():
    cat = [_cat_row("a", input_source="repeat"), _cat_row("b", input_source="coco", family="A-input"),
           _cat_row("c", imgsz=320, cpu_mhz=1500)]
    der = [_der("a", 100.0, 50.0), _der("b", 110.0, 52.0), _der("c", 40.0, 20.0)]
    t = _build(cat, der, [_qc(c) for c in "abc"], _st([("yolo11n", 640), ("yolo11n", 320)]))
    assert len(t) == 2 and list(t["config_key"]) == sorted(t["config_key"])
    m = t[t["imgsz"] == 640].iloc[0]
    assert m["n_cells"] == 2 and m["cell_ids"] == "a;b" and m["family_set"] == "A;A-input"
    assert m["energy_mJ"] == pytest.approx(105.0) and m["latency_ms"] == pytest.approx(51.0)
    assert m["energy_cv"] == pytest.approx(np.std([100, 110], ddof=1) / 105)
    assert m["dyn_energy_mJ"] == pytest.approx(63.0) and m["mean_P_W"] == pytest.approx(5.0)
    assert m["log_macs"] == pytest.approx(math.log(1e9)) and m["cpu_ghz"] == pytest.approx(2.4)
    assert m["is_int8"] == 0 and m["is_fp16"] == 0
    for k in FEATS:
        assert m[k] == FEATS[k]
    assert "cell_id" not in t.columns
    one = t[t["imgsz"] == 320].iloc[0]
    assert one["n_cells"] == 1 and math.isnan(one["energy_cv"]) and math.isnan(one["latency_cv"])
    assert one["cpu_ghz"] == pytest.approx(1.5)


def test_noise_floor_ignores_single_cell_configs():
    cat = [_cat_row("a"), _cat_row("b"), _cat_row("c", imgsz=320)]
    der = [_der("a", 100.0, 50.0), _der("b", 102.0, 50.0), _der("c", 1.0, 1.0)]
    t = _build(cat, der, [_qc(c) for c in "abc"], _st([("yolo11n", 640), ("yolo11n", 320)]))
    expect = np.std([100, 102], ddof=1) / 101
    assert tables.noise_floor(t) == pytest.approx(expect)
    assert math.isnan(tables.noise_floor(t[t["n_cells"] == 1]))


def test_qc_excluded_concurrency_and_other_modes_dropped():
    cat = [_cat_row("a"), _cat_row("bad"), _cat_row("conc", mode_a_eligible=False,
                                                    concurrent_mode="train"),
           _cat_row("lm", base_model="yolov8n", variant="yolov8n")]
    der = [_der("a", 100.0, 50.0), _der("bad", 500.0, 50.0), _der("conc", 900.0, 90.0),
           _der("lm", 1.0, 1.0, mode="decode")]
    q = [_qc("a"), _qc("bad", primary=False), _qc("conc"), _qc("lm")]
    t = _build(cat, der, q)
    assert len(t) == 1 and t.iloc[0]["cell_ids"] == "a"
    assert not t["cell_ids"].str.contains("conc").any()


def test_int8_fp16_flags():
    cat = [_cat_row("a", precision="int8", variant="yolo11n_int8s", quant="int8s", n_quantized_modules=7),
           _cat_row("b", precision="fp16", variant="yolo11n_fp16")]
    t = _build(cat, [_der("a", 1.0, 1.0), _der("b", 1.0, 1.0)], [_qc("a"), _qc("b")])
    i8 = t[t["precision"] == "int8"].iloc[0]
    f16 = t[t["precision"] == "fp16"].iloc[0]
    assert (i8["is_int8"], i8["is_fp16"], i8["n_quantized_modules"], i8["quant"]) == (1, 0, 7, "int8s")
    assert (f16["is_int8"], f16["is_fp16"]) == (0, 1)


def test_missing_static_features_raise_naming_config():
    cat = [_cat_row("a", imgsz=416)]
    with pytest.raises(ValueError, match=r"yolo11n.*416"):
        _build(cat, [_der("a", 1.0, 1.0)], [_qc("a")])


def test_nan_energy_cell_is_dropped():
    cat = [_cat_row("a"), _cat_row("b")]
    t = _build(cat, [_der("a", 100.0, 50.0), _der("b", float("nan"), float("nan"))], [_qc("a"), _qc("b")])
    assert t.iloc[0]["n_cells"] == 1 and t.iloc[0]["cell_ids"] == "a"


def test_real_data(real_data):
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    q = pd.read_csv(Path(qc.__file__).parent / "reports" / "qc_cells.csv")
    st = static.static_table(cat)
    t = tables.mode_a_table(cat, der, q, st)
    assert len(t) == 58
    src = cat.set_index("cell_id")["source"]
    by_src = t["cell_ids"].map(lambda ids: "+".join(sorted({src[i] for i in ids.split(";")})))
    assert by_src.value_counts().to_dict() == {"v2": 43, "v1": 15}
    assert not t["macs"].isna().any()
    assert set(t["base_model"]) <= set(data.VISION_MODELS) and len(data.VISION_MODELS) == 9
    assert t["config_key"].is_unique
    nf = tables.noise_floor(t)
    assert 0.0001 < nf < 0.03
