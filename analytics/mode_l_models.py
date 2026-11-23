"""Window power models of the Mode L ladder (spec 7) and their grids.

Every model has `fit(train_w, params, sample_weight) -> self` and `predict(w) -> np.ndarray` (W). A
model only ever sees the rows it is given; imputation medians, scalers and lookups are fitted inside
`fit`. `sample_weight` is 1 / (windows in the row's cell), so each cell counts equally (it sums to the
number of training cells). Ridge penalties act on that weighted sum, so alpha is relative to a total
weight equal to the number of cells, not the number of windows."""
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

COUNTERS = ["busy", "cycles_rate", "instr_rate", "ipc", "cache_miss_rate", "branch_mpki", "stall_frac",
            "llc_mpki", "ctx_sw_rate", "pf_rate", "counted", "busy_l1", "busy_l2", "busy_l3",
            "cycles_rate_l1", "cycles_rate_l2", "cycles_rate_l3", "ipc_l1", "ipc_l2", "ipc_l3",
            "freq_mhz", "temp_C", "fan_state"]
FAN_MODES = ("full", "auto", "off")
GOVERNORS = ("pinned", "ondemand")
ONEHOT = [f"fan_{f}" for f in FAN_MODES] + [f"gov_{g}" for g in GOVERNORS]
BUSY_HI = 0.5


def cell_weights(w: pd.DataFrame) -> np.ndarray:
    """1 / (number of rows of the row's cell): sums to the number of cells."""
    return (1.0 / w.groupby("cell_id")["cell_id"].transform("size")).to_numpy(dtype=float)


def _col(w, name):
    return w[name].to_numpy(dtype=float) if name in w.columns else np.full(len(w), np.nan)


def _wmean(v, sw):
    return float(np.sum(v * sw) / np.sum(sw))


class L0Lookup:
    """Weighted mean P by (cpu_key, fan_mode, busy > 0.5), falling back to cpu_key, then the global
    weighted mean. Without a `busy` column every window counts as not busy."""

    def _keys(self, w):
        busy = _col(w, "busy") > BUSY_HI
        return pd.DataFrame({"c": w["cpu_key"].astype(str).to_numpy(), "f": w["fan_mode"].astype(str).to_numpy(),
                             "b": busy})

    def fit(self, train_w, params, sample_weight):
        k = self._keys(train_w)
        k["wy"] = train_w["P_W"].to_numpy(float) * sample_weight
        k["w"] = sample_weight
        full = k.groupby(["c", "f", "b"])[["wy", "w"]].sum()
        cpu = k.groupby("c")[["wy", "w"]].sum()
        self.full = (full["wy"] / full["w"]).to_dict()
        self.cpu = (cpu["wy"] / cpu["w"]).to_dict()
        self.glob = float(k["wy"].sum() / k["w"].sum())
        return self

    def predict(self, w):
        k = self._keys(w)
        out = np.empty(len(w))
        for i, (c, f, b) in enumerate(zip(k["c"], k["f"], k["b"])):
            out[i] = self.full.get((c, f, b), self.cpu.get(c, self.glob))
        return out


class L1Pmic:
    """Weighted OLS P = a + b * pmic_W over training windows with a PMIC read. Windows without one are
    predicted by an L0 lookup fitted on the same training rows (`fallback_frac` reports their share)."""

    def fit(self, train_w, params, sample_weight):
        self.l0 = L0Lookup().fit(train_w, {}, sample_weight)
        x, y = _col(train_w, "pmic_W"), train_w["P_W"].to_numpy(float)
        ok = np.isfinite(x)
        x, y, sw = x[ok], y[ok], sample_weight[ok]
        if len(x) == 0 or np.ptp(x) == 0:
            self.a, self.b = (_wmean(y, sw) if len(x) else self.l0.glob), 0.0
            return self
        xm, ym = _wmean(x, sw), _wmean(y, sw)
        self.b = float(np.sum(sw * (x - xm) * (y - ym)) / np.sum(sw * (x - xm) ** 2))
        self.a = float(ym - self.b * xm)
        return self

    def predict(self, w):
        x = _col(w, "pmic_W")
        ok = np.isfinite(x)
        out = np.empty(len(w))
        out[ok] = self.a + self.b * x[ok]
        if (~ok).any():
            out[~ok] = self.l0.predict(w[~ok])
        return out

    @staticmethod
    def fallback_frac(w) -> float:
        return float(np.mean(~np.isfinite(_col(w, "pmic_W")))) if len(w) else float("nan")


class _RidgeMixin:
    """Median-impute inside the fit, standardise with the weighted mean and sd, weighted Ridge."""

    def _ridge_fit(self, X, y, sw, alpha):
        self.med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in X.T])
        X = np.where(np.isfinite(X), X, self.med)
        self.mu = (X * sw[:, None]).sum(0) / sw.sum()
        sd = np.sqrt((((X - self.mu) ** 2) * sw[:, None]).sum(0) / sw.sum())
        self.sd = np.where(sd > 0, sd, 1.0)
        self.ridge = Ridge(alpha=float(alpha)).fit((X - self.mu) / self.sd, y, sample_weight=sw)

    def _ridge_predict(self, X):
        X = np.where(np.isfinite(X), X, self.med)
        return self.ridge.predict((X - self.mu) / self.sd)


class L2Busy(_RidgeMixin):
    """Ridge on busy, busy * freq_mhz and cycles_rate."""

    def fit(self, train_w, params, sample_weight):
        self._ridge_fit(self._X(train_w), train_w["P_W"].to_numpy(float), sample_weight, params["alpha"])
        return self

    @staticmethod
    def _X(w):
        b = _col(w, "busy")
        return np.column_stack([b, b * _col(w, "freq_mhz"), _col(w, "cycles_rate")])

    def predict(self, w):
        return self._ridge_predict(self._X(w))


class L3Counters(_RidgeMixin):
    """All counter features, lags, clock, temperature, fan state and one-hot fan_mode / governor.
    variant "ridge" (params: alpha) or "trees" (HistGradientBoostingRegressor; params max_depth,
    learning_rate, max_iter, min_samples_leaf; random_state = `seed`, no early stopping so max_iter is
    the real number of iterations)."""
    extra: list = []

    def __init__(self, variant="ridge", seed=0):
        if variant not in ("ridge", "trees"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant, self.seed = variant, seed

    def feature_names(self):
        return COUNTERS + ONEHOT + self.extra

    def _X(self, w):
        cols = [_col(w, c) for c in COUNTERS]
        fm, gv = w["fan_mode"].astype(str).to_numpy(), w["governor"].astype(str).to_numpy()
        cols += [(fm == f).astype(float) for f in FAN_MODES] + [(gv == g).astype(float) for g in GOVERNORS]
        cols += [_col(w, c) for c in self.extra]
        X = np.column_stack(cols)
        if self.variant == "ridge" and self.extra:
            X = np.column_stack([X, (~np.isfinite(_col(w, "pmic_W"))).astype(float)])
        return X

    def fit(self, train_w, params, sample_weight):
        X, y = self._X(train_w), train_w["P_W"].to_numpy(float)
        if self.variant == "ridge":
            self._ridge_fit(X, y, sample_weight, params["alpha"])
        else:
            self.hgb = HistGradientBoostingRegressor(
                max_depth=params["max_depth"], learning_rate=params["learning_rate"],
                max_iter=params["max_iter"], min_samples_leaf=params["min_samples_leaf"],
                early_stopping=False, random_state=self.seed).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, w):
        X = self._X(w)
        return self._ridge_predict(X) if self.variant == "ridge" else self.hgb.predict(X)


class L4CountersPmic(L3Counters):
    """L3 plus pmic_W and pmic_age_s (NaN kept for trees; Ridge median-imputes and adds a
    `pmic_missing` indicator)."""
    extra = ["pmic_W", "pmic_age_s"]

    def feature_names(self):
        return super().feature_names() + (["pmic_missing"] if self.variant == "ridge" else [])


# ------------------------------------------------------------------ rungs and grids

class Rung(NamedTuple):
    name: str
    make: Callable      # make(seed) -> a fresh window model
    grid: list          # list of params dicts; one point means nothing to select


ALPHAS = np.logspace(-2, 4, 7)
TREE_GRID = [{"max_depth": d, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": leaf}
             for d in (3, 5) for leaf in (50, 200)]


def default_rungs() -> dict:
    ridge = [{"alpha": float(a)} for a in ALPHAS]
    return {
        "L0": Rung("L0", lambda seed: L0Lookup(), [{}]),
        "L1": Rung("L1", lambda seed: L1Pmic(), [{}]),
        "L2": Rung("L2", lambda seed: L2Busy(), ridge),
        "L3-ridge": Rung("L3-ridge", lambda seed: L3Counters("ridge", seed), ridge),
        "L3-trees": Rung("L3-trees", lambda seed: L3Counters("trees", seed), TREE_GRID),
        "L4-ridge": Rung("L4-ridge", lambda seed: L4CountersPmic("ridge", seed), ridge),
        "L4-trees": Rung("L4-trees", lambda seed: L4CountersPmic("trees", seed), TREE_GRID),
    }


def first_point(rungs: dict) -> dict:
    """The same rungs with only their first grid point (smoke runs)."""
    return {k: Rung(v.name, v.make, v.grid[:1]) for k, v in rungs.items()}
