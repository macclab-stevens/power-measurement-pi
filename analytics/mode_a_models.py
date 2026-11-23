"""Mode A candidates (baselines B0-B3, learned M1-M3) and the novelty distance.

Every candidate has `fit(train, target, params) -> self` and `predict(test) -> np.ndarray` (positive
values in the target's unit). `fit` only sees the rows it is given and never mutates them; `predict`
before `fit` raises ValueError. Learned candidates fit log(target) and return exp(prediction).
`rig_nov` is derived from the `rig` column (1 if "nov2026") wherever FEATURES are assembled.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import settings

FEATURES = ["log_macs", "log_params", "log_weight_bytes", "log_act_bytes", "depth", "frac_macs_conv",
            "frac_macs_linear", "frac_macs_norm", "frac_macs_act", "frac_macs_pool", "frac_macs_other",
            "imgsz", "batch", "threads", "cpu_ghz", "is_int8", "is_fp16", "n_quantized_modules"]
RIG_FEATURE = "rig_nov"
B3_FEATURES = ["log_macs", "log_params", "cpu_ghz", "threads", "batch", "is_int8", "is_fp16"]
TARGETS = ("energy_mJ", "latency_ms", "mean_P_W")
SD_FLOOR = 1e-6


def _need_fit(model, attr: str = "_fitted"):
    if not getattr(model, attr, False):
        raise ValueError(f"{type(model).__name__}.predict called before fit")


def _check_target(target: str):
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")


def _log_target(train: pd.DataFrame, target: str) -> np.ndarray:
    y = train[target].to_numpy(float)
    if len(y) == 0 or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError(f"target {target} must be finite and positive on a non-empty training set")
    return np.log(y)


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Design matrix for `cols`; the pseudo-column `rig_nov` is derived from `rig`."""
    out = {}
    for c in cols:
        if c == RIG_FEATURE:
            out[c] = (df["rig"] == "nov2026").astype(float).to_numpy()
        else:
            out[c] = df[c].to_numpy(float)
    return np.column_stack([out[c] for c in cols]) if len(df) else np.empty((0, len(cols)))


def _nearest_clock(available, mhz: float) -> float:
    """Clock in `available` with the smallest |difference| to `mhz`; ties go to the lower clock."""
    return min((abs(float(a) - float(mhz)), float(a)) for a in set(available))[1]


# ---- novelty ----------------------------------------------------------------------------------
def _standardised(desc: pd.DataFrame, train_models: list[str]) -> pd.DataFrame:
    """desc standardised with mean/SD (population, floor 1e-6) over the training models only."""
    tm = list(dict.fromkeys(train_models))
    if not tm:
        raise ValueError("novelty distance needs at least one training model")
    missing = [m for m in tm if m not in desc.index]
    if missing:
        raise ValueError(f"no descriptor for training models {missing}")
    base = desc.loc[tm].to_numpy(float)
    mu, sd = base.mean(axis=0), np.maximum(base.std(axis=0), SD_FLOOR)
    return pd.DataFrame((desc.to_numpy(float) - mu) / sd, index=desc.index, columns=desc.columns)


def novelty_distances(desc: pd.DataFrame, test_model: str, train_models: list[str]) -> pd.Series:
    """Euclidean distance (training-standardised descriptor space) from test_model to each training model."""
    if test_model not in desc.index:
        raise ValueError(f"no descriptor for test model {test_model!r}")
    z = _standardised(desc, train_models)
    tm = list(dict.fromkeys(train_models))
    d = np.sqrt(((z.loc[tm].to_numpy() - z.loc[test_model].to_numpy()) ** 2).sum(axis=1))
    return pd.Series(d, index=tm)


def novelty_distance(desc: pd.DataFrame, test_model: str, train_models: list[str]) -> float:
    """Minimum distance from test_model to any training model (0 if it is one of them)."""
    if test_model in set(train_models):
        _standardised(desc, train_models)
        return 0.0
    return float(novelty_distances(desc, test_model, train_models).min())


# ---- baselines --------------------------------------------------------------------------------
class B0Median:
    """Median of the training target."""

    def fit(self, train, target, params=None):
        _check_target(target)
        _log_target(train, target)
        self.value_ = float(np.median(train[target].to_numpy(float)))
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        return np.full(len(test), self.value_)


class B1MacsProportional:
    """c(clock) * macs, c = median(target / macs) at the clock (nearest clock if absent). For
    mean_P_W: the per-clock median power, no MAC scaling."""

    def fit(self, train, target, params=None):
        _check_target(target)
        _log_target(train, target)
        self.power_ = target == "mean_P_W"
        y = train[target].to_numpy(float)
        v = y if self.power_ else y / train["macs"].to_numpy(float)
        self.c_ = {float(k): float(np.median(g)) for k, g in pd.Series(v).groupby(train["cpu_mhz"].to_numpy(float))}
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        mhz = test["cpu_mhz"].to_numpy(float)
        c = np.array([self.c_[_nearest_clock(self.c_, m)] for m in mhz])
        return c if self.power_ else c * test["macs"].to_numpy(float)


class B2NearestModel:
    """Target of the most similar training configuration, scaled by macs ratio (energy, latency)."""
    LEVELS = [["imgsz", "batch", "threads", "cpu_mhz", "precision"], ["imgsz", "batch", "cpu_mhz", "precision"],
              ["imgsz", "cpu_mhz", "precision"], ["imgsz", "cpu_mhz"]]
    COLS = ["base_model", "imgsz", "batch", "threads", "cpu_mhz", "precision", "macs"]

    def __init__(self, desc: pd.DataFrame):
        self.desc = desc

    def fit(self, train, target, params=None):
        _check_target(target)
        _log_target(train, target)
        self.target_ = target
        self.train_ = train[self.COLS + [target]].reset_index(drop=True).copy()
        self.models_ = list(dict.fromkeys(self.train_["base_model"]))
        self.z_ = _standardised(self.desc, self.models_)   # standardised over the training models only
        self._fitted = True
        return self

    def _candidates(self, row) -> pd.DataFrame:
        t = self.train_
        for cols in self.LEVELS:
            m = np.ones(len(t), bool)
            for c in cols:
                m &= (t[c] == row[c]).to_numpy()
            if m.any():
                return t[m]
        same = t[t["imgsz"] == row["imgsz"]]
        pool = same if len(same) else t   # last resort: no training row at this imgsz
        return pool[pool["cpu_mhz"] == _nearest_clock(pool["cpu_mhz"], row["cpu_mhz"])]

    def predict(self, test):
        _need_fit(self)
        out = []
        for _, row in test.iterrows():
            cand = self._candidates(row)
            if row["base_model"] not in self.z_.index:
                raise ValueError(f"no descriptor for test model {row['base_model']!r}")
            names = list(dict.fromkeys(cand["base_model"]))
            dist = np.sqrt(((self.z_.loc[names].to_numpy() - self.z_.loc[row["base_model"]].to_numpy()) ** 2).sum(axis=1))
            best = names[int(np.argmin(dist))]   # ties: first in training order
            nn = cand[cand["base_model"] == best]
            v = nn[self.target_].to_numpy(float)
            if self.target_ != "mean_P_W":
                v = v * float(row["macs"]) / nn["macs"].to_numpy(float)
            out.append(float(np.median(v)))   # several rows of the nearest model: median
        return np.array(out)


class B3LogLinear:
    """OLS of log(target) on B3_FEATURES."""

    def fit(self, train, target, params=None):
        _check_target(target)
        y = _log_target(train, target)
        self.model_ = LinearRegression().fit(feature_matrix(train, B3_FEATURES), y)
        self.coef_, self.intercept_ = self.model_.coef_, float(self.model_.intercept_)
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        return np.exp(self.model_.predict(feature_matrix(test, B3_FEATURES)))


# ---- learned models ---------------------------------------------------------------------------
ALL_FEATURES = FEATURES + [RIG_FEATURE]


class M1Ridge:
    """StandardScaler + Ridge on FEATURES + rig_nov, fitted to log(target); params: alpha (default 1)."""

    def fit(self, train, target, params=None):
        _check_target(target)
        y = _log_target(train, target)
        alpha = float((params or {}).get("alpha", 1.0))
        self.model_ = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        self.model_.fit(feature_matrix(train, ALL_FEATURES), y)
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        return np.exp(self.model_.predict(feature_matrix(test, ALL_FEATURES)))


class M2Trees:
    """HistGradientBoostingRegressor on FEATURES + rig_nov, fitted to log(target). `seed` becomes the
    estimator's random_state (default: the config.toml seed); a `random_state` in params wins."""

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def fit(self, train, target, params=None):
        _check_target(target)
        y = _log_target(train, target)
        p = {"random_state": settings.seed() if self.seed is None else self.seed, **(params or {})}
        self.model_ = HistGradientBoostingRegressor(**p).fit(feature_matrix(train, ALL_FEATURES), y)
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        return np.exp(self.model_.predict(feature_matrix(test, ALL_FEATURES)))


class M3Factored:
    """latency_hat * power_hat (ms x W = mJ). params = {"latency": {"name", "params"}, "power": {...}},
    each part M1Ridge or M2Trees, fitted on latency_ms and mean_P_W of the same training rows."""
    PARTS = ("M1Ridge", "M2Trees")

    def __init__(self, seed: int | None = None):
        self.seed = seed   # handed to an M2Trees part

    def fit(self, train, target, params=None):
        if target != "energy_mJ":
            raise ValueError("M3Factored is only valid for target energy_mJ")
        params = params or {}
        parts = {}
        for key, tgt in (("latency", "latency_ms"), ("power", "mean_P_W")):
            spec = params.get(key)
            if not spec or spec.get("name") not in self.PARTS:
                raise ValueError(f"M3Factored params[{key!r}] must name one of {self.PARTS}")
            part = M2Trees(self.seed) if spec["name"] == "M2Trees" else CANDIDATES[spec["name"]]()
            parts[key] = part.fit(train, tgt, spec.get("params") or {})
        self.latency_, self.power_ = parts["latency"], parts["power"]
        self._fitted = True
        return self

    def predict(self, test):
        _need_fit(self)
        return self.latency_.predict(test) * self.power_.predict(test)


CANDIDATES: dict[str, type] = {"B0Median": B0Median, "B1MacsProportional": B1MacsProportional,
                               "B2NearestModel": B2NearestModel, "B3LogLinear": B3LogLinear,
                               "M1Ridge": M1Ridge, "M2Trees": M2Trees, "M3Factored": M3Factored}

GRIDS: dict[str, list[dict] | None] = {
    "B0Median": [{}], "B1MacsProportional": [{}], "B2NearestModel": [{}], "B3LogLinear": [{}],
    "M1Ridge": [{"alpha": float(a)} for a in np.logspace(-2, 4, 7)],
    "M2Trees": [{"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": leaf}
                for d, lr, it, leaf in itertools.product([2, 3, 4], [0.05, 0.1], [100, 300], [3, 5])],
    "M3Factored": None,   # the runner chooses the latency and power parts
}
