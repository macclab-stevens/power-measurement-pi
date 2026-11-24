"""Train/test split battery. Every prediction claim is tested on these folds, so group separation
is the point: a fold is built only when both sides are non-empty, and `train_keys` / `test_keys`
are lists of `config_key` strings from the Mode A configuration table."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import data

VISION = data.VISION_MODELS
YOLO = VISION[:6]
SIZE_PAIRS = {"n": ("yolov8n", "yolo11n"), "s": ("yolov8s", "yolo11s"), "m": ("yolov8m", "yolo11m")}
SIZE_CLASS = {"n": "extrapolation", "s": "interpolation", "m": "extrapolation"}
EXTRA_CLOCK_FOLD = "1500+2000->2400"
N_RANDOM = 5
SPLIT_NAMES = ["S-LOMO", "S-LSO", "S-LVO", "S-LCO", "RANDOM"]


@dataclass
class Fold:
    split: str
    fold: str
    similarity_class: str
    train_keys: list
    test_keys: list


def _keys(t: pd.DataFrame, mask) -> list:
    return sorted(t.loc[mask, "config_key"].astype(str))


def _fold(out: list, split: str, name: str, cls: str, t: pd.DataFrame, test_mask, train_mask) -> None:
    train, test = _keys(t, train_mask), _keys(t, test_mask)
    if train and test:
        out.append(Fold(split, name, cls, train, test))


def _lomo(t: pd.DataFrame) -> list:
    present = set(t["base_model"])
    models = [m for m in VISION if m in present] + sorted(present - set(VISION))
    out: list = []
    for m in models:
        cls = "sibling-seen" if m in YOLO else "new-architecture"
        _fold(out, "S-LOMO", m, cls, t, t["base_model"] == m, t["base_model"] != m)
    return out


def _lso(t: pd.DataFrame) -> list:
    out: list = []
    for name, pair in SIZE_PAIRS.items():
        test = t["base_model"].isin(pair)
        _fold(out, "S-LSO", name, SIZE_CLASS[name], t, test, ~test)
    return out


def _lvo(t: pd.DataFrame) -> list:
    out: list = []
    for v in sorted(set(t.loc[t["precision"] != "fp32", "variant"])):
        prec = set(t.loc[t["variant"] == v, "precision"])
        cls = "variant-seen-on-other-model" if prec == {"int8"} else "unsupported"
        _fold(out, "S-LVO", v, cls, t, t["variant"] == v, t["variant"] != v)
    return out


def _lco(t: pd.DataFrame) -> list:
    clk = t["cpu_mhz"].astype(int)
    out: list = []
    for c in sorted(set(clk)):
        cls = "clock-interpolation" if c == 2000 else "clock-extrapolation"
        _fold(out, "S-LCO", str(c), cls, t, clk == c, clk != c)
    _fold(out, "S-LCO", EXTRA_CLOCK_FOLD, "clock-extrapolation", t, clk == 2400, clk.isin([1500, 2000]))
    return out


def _random(t: pd.DataFrame, seed: int) -> list:
    keys = sorted(t["config_key"].astype(str))   # order-independent: same seed, same folds
    if len(keys) < N_RANDOM:
        return []
    out = []
    for i, (tr, te) in enumerate(KFold(N_RANDOM, shuffle=True, random_state=seed).split(keys)):
        out.append(Fold("RANDOM", f"k{i}", "sanity", sorted(keys[j] for j in tr), sorted(keys[j] for j in te)))
    return out


def make_splits(t: pd.DataFrame, seed: int) -> dict:
    """All five splits over the configuration table `t` (needs `config_key`, `base_model`,
    `variant`, `precision`, `cpu_mhz`). Only folds non-empty on both sides are returned."""
    return {"S-LOMO": _lomo(t), "S-LSO": _lso(t), "S-LVO": _lvo(t), "S-LCO": _lco(t),
            "RANDOM": _random(t, seed)}


def write_splits(sp: dict, outdir) -> dict:
    """Write one CSV per split (`<split>.csv`; columns split, fold, similarity_class, config_key,
    role; rows sorted by split, fold, role, config_key). Returns {file name: sha256 of its bytes}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, folds in sp.items():
        rows = [(f.split, f.fold, f.similarity_class, k, role)
                for f in folds for role, keys in (("train", f.train_keys), ("test", f.test_keys)) for k in keys]
        df = pd.DataFrame(rows, columns=["split", "fold", "similarity_class", "config_key", "role"])
        df = df.sort_values(["split", "fold", "role", "config_key"], kind="stable")
        path = outdir / f"{name}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def inner_folds(train_t: pd.DataFrame) -> list:
    """Inner folds over a training table only: leave-one-base-model-out as (train_keys, val_keys);
    with fewer than 3 base models, leave-one-config-out."""
    keys = train_t["config_key"].astype(str)
    models = sorted(set(train_t["base_model"]))
    out = []
    if len(models) >= 3:
        for m in models:
            hit = (train_t["base_model"] == m).to_numpy()
            tr, va = sorted(keys[~hit]), sorted(keys[hit])
            if tr and va:
                out.append((tr, va))
    else:
        allk = sorted(keys)
        for k in allk:
            tr = [x for x in allk if x != k]
            if tr:
                out.append((tr, [k]))
    return out


# ---- inner folds that match the kind of generalisation being tested ---------------------------------------
N_INNER_RANDOM = 5
REGIMES = ("new-model", "seen-model")


def random_inner_folds(train_t: pd.DataFrame, seed: int, index: int = 0) -> list:
    """Random 5-fold over the configurations of `train_t` as (train_keys, val_keys), deterministic in
    (`seed`, `index`) through numpy.random.default_rng([seed, index]) over the sorted keys; leave-one-config-out
    with fewer than 5 configurations."""
    keys = sorted(train_t["config_key"].astype(str))
    if len(keys) < N_INNER_RANDOM:
        return [([x for x in keys if x != k], [k]) for k in keys if len(keys) > 1]
    perm = np.random.default_rng([int(seed), int(index)]).permutation(len(keys))
    out = []
    for chunk in np.array_split(perm, N_INNER_RANDOM):
        va = sorted(keys[j] for j in chunk)
        out.append(([k for k in keys if k not in set(va)], va))
    return out


def regime_of(train_t: pd.DataFrame, test_t: pd.DataFrame) -> str:
    """"seen-model" when every test base model is in the training frame, else "new-model". Decided from the
    data, not from the split name."""
    return "seen-model" if set(test_t["base_model"]) <= set(train_t["base_model"]) else "new-model"


def inner_folds_for(train_t: pd.DataFrame, regime: str, seed: int, index: int = 0) -> list:
    """Inner folds (train_keys, val_keys) for model selection, from the training table only. The inner rehearsal
    must match the kind of generalisation being tested: a "new-model" regime (a test base model is absent from
    training) is extrapolation over models, rehearsed by leave-one-base-model-out (`inner_folds`); a "seen-model"
    regime (every test base model is in training; a seen model at a new variant, clock or configuration) is
    interpolation over configurations, rehearsed by random 5-fold over configurations (`random_inner_folds`,
    seeded by `seed` and the outer fold `index`)."""
    if regime == "new-model":
        return inner_folds(train_t)
    if regime == "seen-model":
        return random_inner_folds(train_t, seed, index)
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
