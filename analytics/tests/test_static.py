"""Static model features: synthetic layer tables and the real Mode A cells."""
import math

import pandas as pd
import pytest

import data
import static


def _row(path, cls, leaf, params, macs, bm, count):
    return {"path": path, "class": cls, "leaf": leaf, "params": params, "macs": macs,
            "bytes_moved": bm, "count": count}


def _lt():
    return pd.DataFrame([
        _row("0", "Conv", False, 100, None, 9999, 2),           # non-leaf container: ignored
        _row("0.conv", "Conv2d", True, 10, 1000, 4 * 10 + 500, 2),
        _row("0.bn", "BatchNorm2d", True, 4, 80, 4 * 4 + 100, 2),
        _row("0.act", "SiLU", True, 0, 50, 200, 2),
        _row("1.pool", "MaxPool2d", True, 0, 30, 90, 6),         # called 3x per frame
        _row("2.up", "Upsample", True, 0, None, 40, 2),          # NaN macs, class in "other"
        _row("3.fc", "Linear", "True", 20, 400, 4 * 20 + 60, 2), # leaf as a string
        _row("3.fc", "Linear", "True", 20, 800, 4 * 20 + 120, 2),  # 2nd signature, same path
    ])


def test_groups_cover_spec():
    assert static.GROUPS == ["conv", "linear", "norm", "act", "pool", "other"]
    assert static.CLASS_GROUPS["ConvTranspose2d"] == "conv"
    assert static.CLASS_GROUPS["GroupNorm"] == "norm"
    assert static.CLASS_GROUPS["Hardswish"] == "act"
    assert static.CLASS_GROUPS["AdaptiveAvgPool2d"] == "pool"
    assert "Upsample" not in static.CLASS_GROUPS


def test_multiplicity_nan_and_group_fractions():
    f = static.features_from_layer_table(_lt())
    # multiplicity: MaxPool counts 6 vs min 2 -> x3
    assert f["macs"] == 1000 + 80 + 50 + 30 * 3 + 0 + 400 + 800
    assert f["params"] == 10 + 4 + 20          # unique leaf paths, 3.fc once
    assert f["weight_bytes"] == 4 * 34
    # act_bytes: bytes_moved - 4*params_row, x mult
    assert f["act_bytes"] == 500 + 100 + 200 + 90 * 3 + 40 + 60 + 120
    assert f["depth"] == 2
    assert f["n_leaf"] == 6
    assert (f["n_conv"], f["n_norm"], f["n_act"], f["n_pool"], f["n_linear"], f["n_other"]) \
        == (1, 1, 1, 1, 1, 1)
    assert f["frac_macs_conv"] == pytest.approx(1000 / f["macs"])
    assert f["frac_macs_linear"] == pytest.approx(1200 / f["macs"])
    assert f["frac_macs_other"] == 0.0
    assert sum(f[f"frac_macs_{g}"] for g in static.GROUPS) == pytest.approx(1.0)


def test_batch_divides_macs_and_act_bytes_only():
    f1 = static.features_from_layer_table(_lt())
    f2 = static.features_from_layer_table(_lt(), batch=2)
    assert f2["macs"] == f1["macs"] / 2
    assert f2["act_bytes"] == f1["act_bytes"] / 2
    assert f2["params"] == f1["params"]
    assert f2["frac_macs_conv"] == pytest.approx(f1["frac_macs_conv"])


def test_act_bytes_clipped_at_zero():
    lt = pd.DataFrame([_row("a", "Conv2d", True, 100, 10, 50, 1)])   # bytes < 4*params
    assert static.features_from_layer_table(lt)["act_bytes"] == 0


def test_leaf_false_string_is_not_leaf():
    lt = pd.DataFrame([_row("a", "Conv", "False", 5, None, 10, 1),
                       _row("a.b", "Conv2d", "True", 5, 7, 30, 1)])
    f = static.features_from_layer_table(lt)
    assert f["n_leaf"] == 1 and f["macs"] == 7


def _fake_layer_table(scale=1.0):
    return _lt().assign(macs=lambda d: d["macs"] * scale)


def _cat(rows):
    return pd.DataFrame(rows)


def _write(tmp_path, cid, scale=1.0):
    d = tmp_path / cid
    d.mkdir()
    _fake_layer_table(scale).to_csv(d / "layer_table.csv", index=False)
    return str(d)


def _crow(cid, path, model="yolo11n", imgsz=640, **kw):
    r = {"cell_id": cid, "path": path, "base_model": model, "imgsz": imgsz, "batch": 1,
         "precision": "fp32", "mode_a_eligible": True}
    r.update(kw)
    return r


def test_static_table_selection_and_first_by_cell_id(tmp_path):
    cat = _cat([
        _crow("b", _write(tmp_path, "b", 1.0005)),
        _crow("a", _write(tmp_path, "a")),
        _crow("c8", _write(tmp_path, "c8", 5.0), precision="int8"),       # ignored
        _crow("cb", _write(tmp_path, "cb", 5.0), batch=2),                # ignored
        _crow("ce", _write(tmp_path, "ce", 5.0), mode_a_eligible=False),  # ignored
        _crow("m", _write(tmp_path, "m"), model="effb0", imgsz=224),
    ])
    st = static.static_table(cat)
    assert list(st.index) == [("effb0", 224), ("yolo11n", 640)]
    assert st.index.names == ["base_model", "imgsz"]
    assert st.loc[("yolo11n", 640), "macs"] == 1000 + 80 + 50 + 90 + 400 + 800
    assert st.loc[("yolo11n", 640), "cell_id"] == "a"


def test_static_table_disagreement_raises(tmp_path):
    cat = _cat([_crow("a", _write(tmp_path, "a")), _crow("b", _write(tmp_path, "b", 1.5))])
    with pytest.raises(ValueError, match="disagree"):
        static.static_table(cat)


def test_static_table_skips_cells_without_layer_table(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    cat = _cat([_crow("x", str(d)), _crow("a", _write(tmp_path, "a"))])
    assert list(static.static_table(cat)["cell_id"]) == ["a"]


def test_model_descriptor_prefers_640(tmp_path):
    cat = _cat([_crow("a", _write(tmp_path, "a"), imgsz=320),
                _crow("b", _write(tmp_path, "b", 2.0), imgsz=640),
                _crow("c", _write(tmp_path, "c"), model="effb0", imgsz=224)])
    md = static.model_descriptor(static.static_table(cat))
    assert list(md.index) == ["effb0", "yolo11n"]
    assert md.index.name == "base_model"
    st = static.static_table(cat)
    assert md.loc["yolo11n", "log_macs_per_px"] == pytest.approx(
        math.log(st.loc[("yolo11n", 640), "macs"] / 640 ** 2))
    assert md.loc["effb0", "log_macs_per_px"] == pytest.approx(
        math.log(st.loc[("effb0", 224), "macs"] / 224 ** 2))
    assert md.loc["yolo11n", "log_params"] == pytest.approx(math.log(34))
    assert {"depth", *[f"frac_macs_{g}" for g in static.GROUPS]} <= set(md.columns)


def test_real_static_table_covers_mode_a(real_data):
    cat = data.catalogue(real_data)
    st = static.static_table(cat)
    want = [(m, s) for m in ("yolo11n", "yolo11s", "yolo11m") for s in (320, 480, 640, 960)]
    want += [(m, 640) for m in ("yolov8n", "yolov8s", "yolov8m")]
    want += [(m, 224) for m in ("effb0", "mbv3", "resnet18")]
    missing = [w for w in want if w not in st.index]
    assert not missing
    assert 2.5e9 < st.loc[("yolo11n", 640), "macs"] < 4.5e9
    md = static.model_descriptor(st)
    assert set(data.VISION_MODELS[:9]) <= set(md.index)      # the v1/v2 models; v3 ones appear once v3 is present


def test_n_module_calls_counts_containers_and_leaves_with_multiplicity():
    # counts per row: container 2 (x1), conv 2, bn 2, act 2, pool 6 (x3), up 2, fc 2, fc 2  -> 1+1+1+1+3+1+1+1
    f = static.features_from_layer_table(_lt())
    assert f["n_module_calls"] == 10
    assert static.features_from_layer_table(_lt(), batch=2)["n_module_calls"] == 5


def test_n_module_calls_unit_is_the_smallest_positive_leaf_count():
    lt = pd.DataFrame([_row("0", "Seq", False, 0, None, 0, 8),         # container: 2 calls a frame
                       _row("0.a", "Conv2d", True, 1, 5, 10, 4),         # leaf unit = 4 (smallest positive leaf count)
                       _row("0.b", "ReLU", True, 0, 1, 10, 8),
                       _row("0.c", "ReLU", True, 0, 1, 10, 0)])         # never called: not counted
    assert static.n_module_calls(lt) == 2 + 1 + 2 + 0
    assert static.n_module_calls(lt.iloc[0:0]) == 0.0


def test_n_module_calls_reads_the_lockbox_static_csv_through_the_same_path():
    import pathlib
    p = pathlib.Path(static.__file__).parent / "lockbox_static" / "mbv3_224.csv"
    f = static.features_from_layer_table(pd.read_csv(p))
    assert f["n_module_calls"] == 208      # 208 hooked modules, each called once per image


def test_static_layer_tables_reads_model_imgsz_csvs(tmp_path):
    import pandas as pd
    pd.DataFrame({"path": ["a"], "leaf": [True], "params": [1], "macs": [2], "bytes_moved": [8], "class": ["Conv2d"],
                  "count": [1]}).to_csv(tmp_path / "swint_224.csv", index=False)
    out = static.static_layer_tables(tmp_path)
    assert list(out) == [("swint", 224)]
    assert static.static_layer_tables(tmp_path / "missing") == {}
