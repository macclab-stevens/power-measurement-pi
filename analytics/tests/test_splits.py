"""Split battery: group separation, coverage, determinism, inner folds, and the real table."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import data
import splits
import tables

REAL_CSV = Path(__file__).resolve().parents[1] / "reports" / "mode_a_configs.csv"


def _table():
    rows = []
    for bm in data.VISION_MODELS:
        for clk in (1500, 2000, 2400):
            rows.append((bm, bm, "fp32", clk))
    rows += [("yolo11n", "yolo11n_int8s", "int8", 2400), ("yolo11n", "yolo11n_int8s", "int8", 1500),
             ("yolo11s", "yolo11s_int8s", "int8", 2400), ("yolo11n", "yolo11n_fp16", "fp16", 2400)]
    df = pd.DataFrame(rows, columns=["base_model", "variant", "precision", "cpu_mhz"])
    df["imgsz"], df["batch"], df["threads"], df["rig"] = 640, 1, 4, "r"
    df["config_key"] = [tables.config_key(r) for _, r in df.iterrows()]
    return df


def _by_key(t):
    return t.set_index("config_key")


def test_constants():
    assert splits.VISION == data.VISION_MODELS
    assert splits.YOLO == data.VISION_MODELS[:6]


def test_fold_names_and_classes():
    sp = splits.make_splits(_table(), 1)
    assert set(sp) == {"S-LOMO", "S-LSO", "S-LVO", "S-LCO", "RANDOM"}
    assert [f.fold for f in sp["S-LOMO"]] == data.VISION_MODELS
    cls = {f.fold: f.similarity_class for f in sp["S-LOMO"]}
    assert cls["yolov8n"] == "sibling-seen" and cls["resnet18"] == "new-architecture"
    assert {f.fold: f.similarity_class for f in sp["S-LSO"]} == {
        "n": "extrapolation", "s": "interpolation", "m": "extrapolation"}
    assert {f.fold: f.similarity_class for f in sp["S-LVO"]} == {
        "yolo11n_int8s": "variant-seen-on-other-model", "yolo11s_int8s": "variant-seen-on-other-model",
        "yolo11n_fp16": "unsupported"}
    assert {f.fold: f.similarity_class for f in sp["S-LCO"]} == {
        "1500": "clock-extrapolation", "2000": "clock-interpolation", "2400": "clock-extrapolation",
        "1500+2000->2400": "clock-extrapolation"}
    assert [f.fold for f in sp["RANDOM"]] == [f"k{i}" for i in range(5)]
    assert {f.similarity_class for f in sp["RANDOM"]} == {"sanity"}


def test_disjoint_and_nonempty_everywhere():
    t = _table()
    allk = set(t["config_key"])
    for name, folds in splits.make_splits(t, 1).items():
        for f in folds:
            assert f.train_keys and f.test_keys, (name, f.fold)
            assert not set(f.train_keys) & set(f.test_keys), (name, f.fold)
            assert set(f.train_keys) | set(f.test_keys) <= allk


def test_group_separation():
    t = _table()
    bk = _by_key(t)
    sp = splits.make_splits(t, 1)
    for name in ("S-LOMO", "S-LSO"):
        for f in sp[name]:
            tr = {bk.loc[k, "base_model"] for k in f.train_keys}
            te = {bk.loc[k, "base_model"] for k in f.test_keys}
            assert not tr & te, (name, f.fold)
    for f in sp["S-LVO"]:
        assert {bk.loc[k, "variant"] for k in f.test_keys} == {f.fold}
        assert f.fold not in {bk.loc[k, "variant"] for k in f.train_keys}
        assert len(f.train_keys) + len(f.test_keys) == len(t)
    for f in sp["S-LCO"]:
        trc = {bk.loc[k, "cpu_mhz"] for k in f.train_keys}
        tec = {bk.loc[k, "cpu_mhz"] for k in f.test_keys}
        assert not trc & tec
    ex = {f.fold: f for f in sp["S-LCO"]}["1500+2000->2400"]
    assert {bk.loc[k, "cpu_mhz"] for k in ex.train_keys} == {1500, 2000}
    assert {bk.loc[k, "cpu_mhz"] for k in ex.test_keys} == {2400}


def test_each_config_test_once_lomo_random():
    t = _table()
    sp = splits.make_splits(t, 1)
    for name in ("S-LOMO", "RANDOM"):
        keys = [k for f in sp[name] for k in f.test_keys]
        assert sorted(keys) == sorted(t["config_key"]), name


def test_lso_s_fold_is_both_s_models():
    t = _table()
    f = {f.fold: f for f in splits.make_splits(t, 1)["S-LSO"]}["s"]
    want = set(t.loc[t["base_model"].isin(["yolov8s", "yolo11s"]), "config_key"])
    assert set(f.test_keys) == want
    assert set(f.train_keys) == set(t["config_key"]) - want


def test_absent_groups_yield_no_fold():
    t = _table()
    t = t[t["variant"] != "yolo11s_int8s"]
    sp = splits.make_splits(t, 1)
    assert {f.fold for f in sp["S-LVO"]} == {"yolo11n_int8s", "yolo11n_fp16"}
    t2 = t[~t["base_model"].isin(["yolov8n", "yolo11n"])]
    assert {f.fold for f in splits.make_splits(t2, 1)["S-LSO"]} == {"s", "m"}
    t3 = t[t["cpu_mhz"] == 2400]
    assert splits.make_splits(t3, 1)["S-LCO"] == []   # a single clock leaves nothing to train on
    t4 = t[t["cpu_mhz"] != 2400]
    assert {f.fold for f in splits.make_splits(t4, 1)["S-LCO"]} == {"1500", "2000"}


def test_inner_folds_stay_inside_training_table():
    t = _table()
    f = splits.make_splits(t, 1)["S-LOMO"][0]
    train_t = t[t["config_key"].isin(f.train_keys)]
    inner = splits.inner_folds(train_t)
    assert len(inner) == train_t["base_model"].nunique()
    ok = set(train_t["config_key"])
    for tr, va in inner:
        assert set(tr) | set(va) <= ok
        assert not set(tr) & set(va) and tr and va
        assert {train_t.set_index("config_key").loc[k, "base_model"] for k in va}.isdisjoint(
            {train_t.set_index("config_key").loc[k, "base_model"] for k in tr})
    assert sorted(k for _, va in inner for k in va) == sorted(ok)


def test_inner_folds_few_models_is_leave_one_config_out():
    t = _table()
    sub = t[t["base_model"].isin(["effb0", "mbv3"]) & (t["variant"] == t["base_model"])]
    inner = splits.inner_folds(sub)
    assert len(inner) == len(sub) == 6
    assert all(len(va) == 1 and len(tr) == 5 for tr, va in inner)


def test_same_seed_identical_files_different_seed_differs(tmp_path):
    t = _table()
    h1 = splits.write_splits(splits.make_splits(t, 7), tmp_path / "a")
    h2 = splits.write_splits(splits.make_splits(t, 7), tmp_path / "b")
    h3 = splits.write_splits(splits.make_splits(t.sample(frac=1, random_state=0), 7), tmp_path / "c")
    h4 = splits.write_splits(splits.make_splits(t, 8), tmp_path / "d")
    assert h1 == h2 == h3
    assert h1["RANDOM.csv"] != h4["RANDOM.csv"]
    for name, h in h1.items():
        assert hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest() == h
    df = pd.read_csv(tmp_path / "a" / "S-LOMO.csv")
    assert list(df.columns) == ["split", "fold", "similarity_class", "config_key", "role"]
    assert set(df["role"]) == {"train", "test"}


@pytest.mark.skipif(not REAL_CSV.is_file(), reason="reports/mode_a_configs.csv absent")
def test_real_table_fold_counts():
    t = pd.read_csv(REAL_CSV)
    sp = splits.make_splits(t, 20261122)
    assert {k: len(v) for k, v in sp.items()} == {"S-LOMO": 15, "S-LSO": 3, "S-LVO": 3, "S-LCO": 4, "RANDOM": 5}
    assert {f.fold: f.similarity_class for f in sp["S-LVO"]}["yolo11n_fp16"] == "unsupported"
    assert [f.fold for f in sp["S-LCO"]] == ["1500", "2000", "2400", "1500+2000->2400"]


# ---- R2: inner folds matched to the regime ---------------------------------------------------------------
def _check_partition(train_t, inner):
    ok = sorted(train_t["config_key"])
    assert sorted(k for _, va in inner for k in va) == ok            # every key validates exactly once
    for tr, va in inner:
        assert tr and va and not set(tr) & set(va) and set(tr) | set(va) == set(ok)


def _fold_frames(t, split, fold):
    f = next(x for x in splits.make_splits(t, 1)[split] if x.fold == fold)
    return t[t["config_key"].isin(f.train_keys)].reset_index(drop=True), t[t["config_key"].isin(f.test_keys)]


def test_regime_is_decided_from_the_data_not_the_split_name():
    t = _table()
    for split, fold, want in (("S-LOMO", "yolo11n", "new-model"), ("S-LSO", "n", "new-model"),
                              ("S-LVO", "yolo11n_fp16", "seen-model"), ("S-LCO", "2400", "seen-model"),
                              ("S-LCO", "1500+2000->2400", "seen-model")):
        train, test = _fold_frames(t, split, fold)
        assert splits.regime_of(train, test) == want, (split, fold)
    train, test = _fold_frames(t, "S-LOMO", "yolo11n")
    assert splits.regime_of(train, train.iloc[:3]) == "seen-model"      # any frame whose models are all in training
    assert splits.regime_of(train, pd.concat([train.iloc[:2], test])) == "new-model"   # one absent model is enough


def test_new_model_regime_is_leave_one_base_model_out_and_seen_model_is_random_five_fold():
    t = _table()
    train, _ = _fold_frames(t, "S-LOMO", "yolo11n")
    assert splits.inner_folds_for(train, "new-model", 1) == splits.inner_folds(train)
    inner = splits.inner_folds_for(train, "seen-model", 7, 2)
    assert len(inner) == 5
    _check_partition(train, inner)
    with pytest.raises(ValueError, match="regime"):
        splits.inner_folds_for(train, "S-LOMO", 1)


def test_inner_random_is_deterministic_in_seed_and_index_order_independent_and_loco_when_tiny():
    t = _table()
    a = splits.inner_folds_for(t, "seen-model", 7, 2)
    assert a == splits.inner_folds_for(t.sample(frac=1, random_state=0), "seen-model", 7, 2)
    assert a != splits.inner_folds_for(t, "seen-model", 7, 3) and a != splits.inner_folds_for(t, "seen-model", 8, 2)
    inner = splits.inner_folds_for(t.iloc[:4], "seen-model", 7)
    assert len(inner) == 4 and all(len(va) == 1 and len(tr) == 3 for tr, va in inner)
