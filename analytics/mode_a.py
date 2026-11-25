"""Mode A nested evaluation: every candidate is scored on unseen configurations of each outer fold.

Leakage rule: everything that is fitted, selected or thresholded for an outer fold sees only that
fold's training rows. `run_split` hands each outer fold a training frame and a test frame; the search
(`_Search`) is built from the training frame alone, so test rows cannot reach a fit, a grid choice,
an interval quantile or the applicability threshold. Inner folds match the kind of
generalisation the outer fold tests (`splits.inner_folds_for`, R2), decided from the data: when some test base
model is absent from the training frame (new-model regime: extrapolation over models) they are leave-one-base-
model-out; when every test base model is in the training frame (seen-model regime: a seen model at a new variant,
clock or configuration is interpolation over configurations) they are random 5-fold over configurations. The
applicability threshold always comes from leave-one-base-model-out inner folds (novelty is a per-model notion).
Prediction-interval quantiles weight every inner fold equally.

Per outer fold and target (R2): for each candidate the grid point with the lowest inner BALANCED error
is chosen (ties: first grid point). The balanced error is the mean over inner folds of the fold's median
|log(pred / y)|; with leave-one-base-model-out folds every held-out base model counts once
whatever its number of rows (the R1 rule pooled all inner rows, which let the many-row YOLO models decide).
A non-finite fold error makes the point +inf (never selected). M3's latency and power parts are chosen the
same way among the M1 and M2 grid points on `latency_ms` and `mean_P_W`; M4's parts among its own grid
(beta for latency, alpha for power) on the same targets. AUTO is the candidate with the lowest balanced
error among `AUTO_AMONG`. The pooled MdAPE (`_mdape`) is kept for the applicability threshold only. Inner
predictions are cached per outer fold, keyed by candidate, grid point (sorted JSON) and target, so M3 and
M4 reuse the inner predictions of their parts' targets.
"""
import hashlib
import itertools
import json
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import evaluate
import mode_a_models as mm
import settings
import splits

AUTO = "AUTO"
APPLICABLE_MDAPE = 0.15
BASELINES = ["B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear"]
LEARNED = ["M1Ridge", "M2Trees"]
PARTS = {"latency": "latency_ms", "power": "mean_P_W"}
FACTORED = ("M3Factored", "M4Physics")   # energy = latency part x power part
CANDIDATES_BY_TARGET = {"energy_mJ": BASELINES + LEARNED + ["M3Factored", "M4Physics"],
                        "latency_ms": BASELINES + LEARNED + ["M4Physics"],
                        "mean_P_W": BASELINES + LEARNED + ["M4Physics"]}
AUTO_AMONG = {"energy_mJ": LEARNED + ["M3Factored", "M4Physics"], "latency_ms": LEARNED + ["M4Physics"],
              "mean_P_W": LEARNED + ["M4Physics"]}
OUT_COLS = ["split", "fold", "similarity_class", "config_key", "base_model", "candidate", "target", "seed",
            "y_true", "y_pred", "lo90", "hi90", "novelty", "flagged", "threshold", "params", "n_resid"]


def _json(p) -> str:
    return json.dumps(p, sort_keys=True)


def _make(name: str, desc: pd.DataFrame, seed: int | None = None):
    """A fresh candidate; `seed` reaches every HistGradientBoostingRegressor (M2, M3's M2 parts)."""
    cls = mm.CANDIDATES[name]   # M4Physics is deterministic and takes no seed
    if name == "B2NearestModel":
        return cls(desc)
    return cls(seed) if name in ("M2Trees", "M3Factored") else cls()


def _mdape(y, p) -> float:
    """Median absolute percentage error; +inf when empty or not finite (never selected)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.size == 0:
        return float("inf")
    v = float(np.median(np.abs(p - y) / y))
    return v if np.isfinite(v) else float("inf")


def _balanced(y_parts, p_parts) -> float:
    """Model-balanced selection error: the mean over inner folds of the fold's median |log(p / y)|. Each
    inner fold holds out one base model, so each model counts once whatever its row count. +inf when there
    is no fold or any fold error is not finite (a prediction <= 0 or NaN): such a point is never selected."""
    errs = []
    for y, p in zip(y_parts, p_parts):
        y, p = np.asarray(y, float), np.asarray(p, float)
        if y.size == 0:
            continue
        with np.errstate(all="ignore"):
            e = float(np.median(np.abs(np.log(p / y)))) if np.all(p > 0) and np.all(y > 0) else float("nan")
        if not np.isfinite(e):
            return float("inf")
        errs.append(e)
    return float(np.mean(errs)) if errs else float("inf")


def _pooled(parts) -> np.ndarray:
    return np.concatenate([np.asarray(x, float) for x in parts])


def _check_table(t: pd.DataFrame) -> None:
    if t["config_key"].duplicated().any():
        raise ValueError("config_key must be unique in the configuration table")


class _Search:
    """Inner-fold model selection over one training frame (the only data it can see)."""

    def __init__(self, train_t: pd.DataFrame, desc: pd.DataFrame, grids: dict, seed: int | None = None,
                 folds: list | None = None):
        """`folds`: inner folds as (train_keys, val_keys); default leave-one-base-model-out."""
        _check_table(train_t)
        self.desc, self.grids, self.seed = desc, grids, seed
        by = train_t.set_index("config_key", drop=False)
        self.folds = [(by.loc[tr].reset_index(drop=True), by.loc[va].reset_index(drop=True))
                      for tr, va in (splits.inner_folds(train_t) if folds is None else folds)]
        self.cache: dict = {}

    def truth(self, target: str) -> list:
        return [va[target].to_numpy(float) for _, va in self.folds]

    def inner_pred(self, name: str, params: dict, target: str) -> list:
        """Per inner fold predictions of one grid point (cached by candidate, point, target)."""
        key = (name, _json(params), target)
        if key not in self.cache:
            self.cache[key] = [np.asarray(_make(name, self.desc, self.seed).fit(tr, target, params).predict(va), float)
                               for tr, va in self.folds]
        return self.cache[key]

    def preds_of(self, name: str, params: dict, target: str) -> list:
        """Inner predictions per fold of a chosen candidate with its `select` params (a factored candidate's
        energy predictions are the product of its parts')."""
        if name in FACTORED and target == "energy_mJ":
            parts = [self.inner_pred(params[k]["name"], params[k]["params"], tgt) if name == "M3Factored"
                     else self.inner_pred(name, params[k], tgt) for k, tgt in PARTS.items()]
            return [lat * pw for lat, pw in zip(*parts)]
        return self.inner_pred(name, params, target)

    def _best(self, names, target: str):
        """(name, params, inner preds, balanced error) of the best grid point among `names`."""
        y = self.truth(target)
        best = None
        for name in names:
            for params in mm.grid_for(self.grids, name, target):
                preds = self.inner_pred(name, params, target)
                m = _balanced(y, preds)
                if best is None or m < best[3]:   # strict: ties keep the first grid point
                    best = (name, params, preds, m)
        return best

    def select(self, name: str, target: str) -> dict:
        """Chosen grid point of `name` with its inner predictions per fold, its balanced inner error
        (`score`, used to select) and its pooled inner MdAPE (`mdape`, informational)."""
        y = self.truth(target)
        if name not in FACTORED:
            n, params, preds, m = self._best([name], target)
            return {"params": params, "preds": preds, "score": m, "mdape": _mdape(_pooled(y), _pooled(preds))}
        if target != "energy_mJ":
            if name == "M3Factored":
                raise ValueError("M3Factored is only valid for target energy_mJ")
            n, params, preds, m = self._best([name], target)
            return {"params": params, "preds": preds, "score": m, "mdape": _mdape(_pooled(y), _pooled(preds))}
        chosen, part_preds = {}, []
        for key, tgt in PARTS.items():
            n, params, preds, _ = self._best(LEARNED if name == "M3Factored" else [name], tgt)
            chosen[key] = {"name": n, "params": params} if name == "M3Factored" else params
            part_preds.append(preds)
        preds = [lat * pw for lat, pw in zip(*part_preds)]
        return {"params": chosen, "preds": preds, "score": _balanced(y, preds),
                "mdape": _mdape(_pooled(y), _pooled(preds))}


def wquantile(values, weights, qs) -> np.ndarray:
    """Weighted quantiles by linear interpolation of the weighted plotting positions
    p_i = (cumulative weight up to and including i - w_i / 2) / total weight over the sorted values
    (clamped to the extreme values outside [p_first, p_last]); equal weights give the positions (i + 1/2) / n."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    pos = (np.cumsum(w) - w / 2) / w.sum()
    return np.interp(np.atleast_1d(np.asarray(qs, float)), pos, v)


def _interval(yhat: np.ndarray, resid: np.ndarray, weights: np.ndarray | None = None):
    if resid.size < 2:
        nan = np.full(len(yhat), np.nan)
        return nan, nan.copy()
    w = np.ones(resid.size) if weights is None else weights
    q5, q95 = wquantile(resid, w, [0.05, 0.95])
    return yhat * np.exp(q5), yhat * np.exp(q95)


def _residuals(y_parts, p_parts) -> tuple:
    """(finite log residuals log y - log p over the inner folds, weights): a fold's residuals each weigh
    1 / (the number of its finite residuals), so every inner fold counts equally in the interval quantiles."""
    res, wts = [], []
    for y, p in zip(y_parts, p_parts):
        with np.errstate(all="ignore"):
            r = np.log(np.asarray(y, float)) - np.log(np.asarray(p, float))
        r = r[np.isfinite(r)]
        if r.size:
            res.append(r)
            wts.append(np.full(r.size, 1.0 / r.size))
    if not res:
        return np.empty(0), np.empty(0)
    return np.concatenate(res), np.concatenate(wts)


def _threshold(novelties, mdapes) -> float:
    """Largest inner-fold novelty whose inner AUTO MdAPE is below 15 %; 0.0 if there is none."""
    ok = [n for n, m in zip(novelties, mdapes) if m < APPLICABLE_MDAPE]
    return float(max(ok)) if ok else 0.0


def _model_novelty(desc, model: str, train_models) -> float:
    tm = sorted(set(train_models))
    return 0.0 if model in tm else mm.novelty_distance(desc, model, tm)


def _run_fold(train_t, test_t, fold_info: tuple, target: str, seed: int, desc, grids: dict,
              index: int = 0) -> pd.DataFrame:
    """One outer fold from its training and test frames (the search never sees `test_t`). `index` is the
    position of the fold in its split (it seeds the random inner folds)."""
    split, fold, cls = fold_info
    regime = splits.regime_of(train_t, test_t)
    search = _Search(train_t, desc, grids, seed, splits.inner_folds_for(train_t, regime, seed, index))
    names = CANDIDATES_BY_TARGET[target]
    sel = {n: search.select(n, target) for n in names}
    auto = min(AUTO_AMONG[target], key=lambda n: sel[n]["score"])   # min keeps the first on ties
    truth = search.truth(target)
    # applicability threshold: novelty is a per-model notion, so always from leave-one-base-model-out inner
    # folds (AUTO's chosen point; the selection folds themselves when they already are those)
    lomo = search if regime == "new-model" else _Search(train_t, desc, grids, seed)
    lomo_preds = sel[auto]["preds"] if lomo is search else lomo.preds_of(auto, sel[auto]["params"], target)
    novs = [max(_model_novelty(desc, m, tr["base_model"]) for m in set(va["base_model"]))
            for tr, va in lomo.folds]
    mdapes = [_mdape(y, p) for y, p in zip(lomo.truth(target), lomo_preds)]
    threshold = _threshold(novs, mdapes)
    train_models = sorted(set(train_t["base_model"]))
    nov = {m: _model_novelty(desc, m, train_models) for m in sorted(set(test_t["base_model"]))}
    novelty = test_t["base_model"].map(nov).to_numpy(float)

    final = {n: np.asarray(_make(n, desc, seed).fit(train_t, target, sel[n]["params"]).predict(test_t), float)
             for n in names}   # one refit per candidate; AUTO reuses the chosen one

    def block(label, name, params):
        s, yhat = sel[name], final[name]
        resid, wts = _residuals(truth, s["preds"])
        lo, hi = _interval(yhat, resid, wts)
        shown = params if label == name else {"candidate": name, "params": params}
        return pd.DataFrame({
            "split": split, "fold": fold, "similarity_class": cls, "config_key": test_t["config_key"].to_numpy(),
            "base_model": test_t["base_model"].to_numpy(), "candidate": label, "target": target, "seed": seed,
            "y_true": test_t[target].to_numpy(float), "y_pred": yhat, "lo90": lo, "hi90": hi,
            "novelty": novelty, "flagged": novelty > threshold, "threshold": threshold, "params": _json(shown),
            "n_resid": int(resid.size)})

    parts = [block(n, n, sel[n]["params"]) for n in names]
    parts.append(block(AUTO, auto, sel[auto]["params"]))
    return pd.concat(parts, ignore_index=True)[OUT_COLS]


def run_split(t: pd.DataFrame, folds: list, target: str, seed: int, desc: pd.DataFrame,
              n_jobs: int = 1, grids: dict | None = None) -> pd.DataFrame:
    """Score every candidate (plus AUTO) on each outer fold: one row per (fold, test config, candidate).
    Results are identical for any `n_jobs`. `grids` defaults to `mode_a_models.GRIDS`."""
    if target not in CANDIDATES_BY_TARGET:
        raise ValueError(f"unknown target {target!r}")
    _check_table(t)
    grids = grids if grids is not None else mm.GRIDS
    by = t.set_index("config_key", drop=False)
    jobs = []
    for f in folds:
        if set(f.train_keys) & set(f.test_keys):
            raise ValueError(f"fold {f.split}/{f.fold}: train and test keys overlap")
        missing = (set(f.train_keys) | set(f.test_keys)) - set(by.index)
        if missing:
            raise ValueError(f"fold {f.split}/{f.fold}: keys not in the table: {sorted(missing)[:3]}")
        jobs.append((by.loc[list(f.train_keys)].reset_index(drop=True), by.loc[list(f.test_keys)].reset_index(drop=True),
                     (f.split, f.fold, f.similarity_class)))
    res = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(tr, te, info, target, seed, desc, grids, i)
                                  for i, (tr, te, info) in enumerate(jobs))
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=OUT_COLS)


REGIMES = splits.REGIMES


def fit_final(t: pd.DataFrame, candidate: str, target: str, seed: int, desc: pd.DataFrame, grids: dict | None = None,
              regime: str = "new-model"):
    """Grid chosen over all of `t` by the balanced inner error, refit on every row, pickled under
    derived/models/mode_a_{target}_{candidate}_{regime}.pkl. `regime` "new-model" selects with leave-one-base-
    model-out inner folds (the model to predict is new), "seen-model" with random 5-fold folds over the
    configurations (its base model is in training). Returns (model, {"path", "sha256", "params", "regime"}).
    `seed` is the random_state of every HistGradientBoostingRegressor (selection and final fit) and seeds the
    random inner folds."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    grids = grids if grids is not None else mm.GRIDS
    if candidate not in mm.CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    folds = splits.inner_folds_for(t, regime, seed)
    params = _Search(t, desc, grids, seed, folds).select(candidate, target)["params"]
    model = _make(candidate, desc, seed).fit(t.reset_index(drop=True), target, params)
    out = settings.derived_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mode_a_{target}_{candidate}_{regime}.pkl"
    path.write_bytes(pickle.dumps(model))
    return model, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "params": params,
                   "regime": regime}


def fit_candidates(t: pd.DataFrame, target: str, seed: int, desc: pd.DataFrame, regime: str,
                   grids: dict | None = None) -> dict:
    """Post hoc scoring helper: the search of `fit_final` run once over `t` for every candidate of `target` plus AUTO,
    with no pickles written. Inner folds are those of `regime` (`splits.inner_folds_for`, index 0, as in `fit_final`).
    Returns {candidate or AUTO: {"model", "params", "score", "q5", "q95", "n_resid", "chosen"}}: the model refitted on
    every row of `t` with the chosen grid point, the balanced inner error, the fold-weighted 5 % and 95 % quantiles of the
    inner log residuals log(y / pred) (the 90 % interval of a prediction p is [p exp(q5), p exp(q95)], as in `_run_fold`;
    NaN when there are fewer than two residuals) and, for AUTO, the candidate it picked (the lowest balanced error among
    `AUTO_AMONG`, ties to the first). Only `t` is read."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    if target not in CANDIDATES_BY_TARGET:
        raise ValueError(f"unknown target {target!r}")
    grids = grids if grids is not None else mm.GRIDS
    search = _Search(t, desc, grids, seed, splits.inner_folds_for(t, regime, seed))
    truth = search.truth(target)
    sel = {n: search.select(n, target) for n in CANDIDATES_BY_TARGET[target]}
    t0 = t.reset_index(drop=True)
    out = {}
    for n, s in sel.items():
        resid, wts = _residuals(truth, s["preds"])
        q5, q95 = wquantile(resid, wts, [0.05, 0.95]) if resid.size >= 2 else (np.nan, np.nan)
        out[n] = {"model": _make(n, desc, seed).fit(t0, target, s["params"]), "params": s["params"], "score": s["score"],
                  "q5": float(q5), "q95": float(q95), "n_resid": int(resid.size), "chosen": n}
    auto = min(AUTO_AMONG[target], key=lambda n: sel[n]["score"])
    out[AUTO] = {**out[auto], "chosen": auto}
    return out


def learning_curve(t: pd.DataFrame, target: str, seed: int, candidate: str, params: dict | None,
                   desc: pd.DataFrame, max_subsets: int = 10) -> pd.DataFrame:
    """Fixed candidate and params (no inner search). Each YOLO base model is held out in turn; train on
    k = 2..5 of the other YOLO models plus every non-YOLO configuration. All k-subsets if there are at
    most `max_subsets`, else `max_subsets` drawn with numpy.random.default_rng(seed)."""
    _check_table(t)
    rng = np.random.default_rng(seed)
    yolo = [m for m in splits.YOLO if m in set(t["base_model"])]
    non_yolo = t[~t["base_model"].isin(splits.YOLO)]
    rows = []
    for held in yolo:
        test = t[t["base_model"] == held]
        others = [m for m in yolo if m != held]
        for k in range(2, 6):
            if k > len(others):
                continue
            combos = list(itertools.combinations(others, k))
            if len(combos) > max_subsets:
                combos = [combos[i] for i in sorted(rng.choice(len(combos), size=max_subsets, replace=False))]
            for sub in combos:
                train = pd.concat([t[t["base_model"].isin(sub)], non_yolo], ignore_index=True)
                yhat = _make(candidate, desc, seed).fit(train, target, params or {}).predict(test)
                rows.append({"heldout": held, "k": k, "subset": ";".join(sub),
                             "mdape": evaluate.metrics(test[target].to_numpy(float), yhat)["mdape"]})
    return pd.DataFrame(rows, columns=["heldout", "k", "subset", "mdape"])
