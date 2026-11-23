"""Mode A tables of the results report. Pure functions of prediction frames (the `mode_a` parquet
columns) and the configuration table; no I/O. Percent quantities arrive as fractions and are shown as
percent. Candidate order is fixed by the protocol: B0, B1, B2, B3, M1, M2, M3, AUTO."""
import numpy as np
import pandas as pd

import eda_write as ew
import evaluate
import mode_a

CAND_ORDER = mode_a.BASELINES + mode_a.LEARNED + ["M3Factored", mode_a.AUTO]
SHORT = {"B0Median": "B0 median", "B1MacsProportional": "B1 MACs proportional", "B2NearestModel": "B2 nearest model",
         "B3LogLinear": "B3 log-linear", "M1Ridge": "M1 ridge", "M2Trees": "M2 trees", "M3Factored": "M3 factored",
         "AUTO": "AUTO"}
UNITS = {"energy_mJ": "mJ", "latency_ms": "ms", "mean_P_W": "W"}
NOUN = {"energy_mJ": "energy per image", "latency_ms": "latency per image", "mean_P_W": "mean power"}
FLOOR_COL = {"energy_mJ": "energy_cv", "latency_ms": "latency_cv"}
P_SIG = 0.05
FACTORS = ["precision", "imgsz", "threads", "batch", "cpu_mhz", "rig"]


def group_col(split: str) -> str:
    """Bootstrap clusters: base models for the group splits, configurations for RANDOM."""
    return "config_key" if split == "RANDOM" else "base_model"


def ordered(df: pd.DataFrame, col: str = "candidate") -> pd.DataFrame:
    rank = {c: i for i, c in enumerate(CAND_ORDER)}
    return df.sort_values(col, key=lambda s: s.map(lambda c: rank.get(c, 99)), kind="stable").reset_index(drop=True)


def noise_floor(configs: pd.DataFrame, target: str) -> float:
    """Median repeat CV over configurations with at least two cells (NaN where none is recorded)."""
    col = FLOOR_COL.get(target)
    if col is None or col not in configs:
        return float("nan")
    sub = configs.loc[configs["n_cells"] >= 2, col]
    return float(sub.median()) if len(sub) else float("nan")


def pct(x: float, d: int = 1) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def pct_ci(p: float, lo: float, hi: float) -> str:
    return "" if not np.isfinite(p) else f"{pct(p)} [{pct(lo)}, {pct(hi)}]"


def _num(x: float) -> str:
    """Fixed notation with about four significant digits (no exponent for the magnitudes seen here)."""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.1f}" if abs(x) >= 100 else f"{x:.2f}" if abs(x) >= 1 else f"{x:.3f}"


def num_ci(p: float, lo: float, hi: float) -> str:
    return "" if not np.isfinite(p) else f"{_num(p)} [{_num(lo)}, {_num(hi)}]"


def floor_text(nf: float, target: str) -> str:
    if np.isfinite(nf):
        return f"noise floor (median repeat CV, configurations with at least 2 cells) {pct(nf)} %"
    return ("noise floor not defined for this target (no repeat CV is recorded); no indistinguishability rule applied"
            if target == "mean_P_W" else "noise floor not defined (no configuration with 2 or more cells)")


def version_text(configs: pd.DataFrame, keys) -> str:
    """Configurations, cells and dataset versions of the configurations `keys` (ew.prov wording)."""
    sub = configs[configs["config_key"].isin(set(keys))]
    return ew.prov(sub)


def floor_reused(nf: float) -> str:
    """Floor text for Mode L and probe tables, which reuse Mode A's energy noise floor."""
    return f"Mode A's energy {floor_text(nf, 'energy_mJ')}, reused here as the indistinguishability margin"


def caption(*, split: str, target: str, prov: str, run_ids, floor: str | None, extra: str = "") -> str:
    """One-line provenance above every table: split, target, n with dataset version(s) (`prov`, e.g. from
    `version_text`), run_id and the noise floor (`floor`, from `floor_text` / `floor_reused`; pass None only
    for tables that are not error tables). `floor` has no default so that no caption can omit it silently."""
    ids = f"`{run_ids}`" if isinstance(run_ids, str) else ", ".join(f"`{r}`" for r in run_ids)
    return (f"*Split: {split}. Target: {target}. {prov}. run_id: {ids}." + (f" {floor[0].upper()}{floor[1:]}." if floor else "") + "*"
            + (f" {extra}" if extra else "") + "\n")


# ------------------------------------------------------------------ pooled metrics with intervals

def _boot(d: pd.DataFrame, group: str, fn, n_boot: int, seed: int):
    return evaluate.cluster_bootstrap(d, group, fn, n_boot, seed=seed)


def pooled_table(df: pd.DataFrame, split: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Per candidate (protocol order): n, MdAPE, MAPE, MAE, bias with cluster-bootstrap 95 % intervals
    (columns `<m>`, `<m>_lo`, `<m>_hi`) and 90 % coverage."""
    g = group_col(split)
    rows = []
    for cand, c in df.groupby("candidate", sort=False):
        d = c[[g, "y_true", "y_pred"]].reset_index(drop=True)
        row = {"candidate": cand, **{k: v for k, v in evaluate.metrics(c["y_true"], c["y_pred"]).items()
                                     if k in ("n", "mdape", "mape", "mae", "bias_pct")},
               "coverage90": evaluate.coverage(c["y_true"], c["lo90"], c["hi90"])}
        for key in ("mdape", "mape", "mae", "bias_pct"):
            row[f"{key}_lo"], row[f"{key}_hi"] = _boot(d, g, lambda x, k=key: evaluate.metrics(x["y_true"], x["y_pred"])[k],
                                                       n_boot, seed)
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def best_baseline(raw: pd.DataFrame) -> str:
    """Baseline (B0-B3) with the lowest pooled MdAPE."""
    b = raw[raw["candidate"].isin(mode_a.BASELINES)]
    if b.empty:
        raise ValueError("no baseline rows in the predictions")
    return str(b.loc[b["mdape"].idxmin(), "candidate"])


def show_pooled(raw: pd.DataFrame, target: str) -> pd.DataFrame:
    u = UNITS[target]
    return pd.DataFrame({
        "candidate": raw["candidate"].map(SHORT), "n": raw["n"],
        "MdAPE % [95 % CI]": [pct_ci(*r) for r in zip(raw["mdape"], raw["mdape_lo"], raw["mdape_hi"])],
        "MAPE % [95 % CI]": [pct_ci(*r) for r in zip(raw["mape"], raw["mape_lo"], raw["mape_hi"])],
        f"MAE {u} [95 % CI]": [num_ci(*r) for r in zip(raw["mae"], raw["mae_lo"], raw["mae_hi"])],
        "bias % [95 % CI]": [pct_ci(*r) for r in zip(raw["bias_pct"], raw["bias_pct_lo"], raw["bias_pct_hi"])],
        "coverage90": raw["coverage90"]})


# ------------------------------------------------------------------ paired comparison

def _by_pair(df: pd.DataFrame, cand: str) -> pd.DataFrame:
    return df[df["candidate"] == cand].set_index(["fold", "config_key"])[["y_true", "y_pred"]]


def verdict(med_ape_diff: float, p_holm: float, noise: float) -> str:
    if not np.isfinite(med_ape_diff):
        return "n/a"
    if np.isfinite(noise) and abs(med_ape_diff) < noise:
        return "indistinguishable (below noise floor)"
    if p_holm >= P_SIG:
        return "not significant"
    return "better" if med_ape_diff < 0 else "worse"


def paired_table(df: pd.DataFrame, baseline: str, noise: float) -> pd.DataFrame:
    """Each learned candidate against `baseline`, paired by (fold, configuration): Wilcoxon signed-rank on
    absolute log errors, Holm over the family, median paired differences, and the verdict."""
    base = _by_pair(df, baseline)
    rows = []
    for cand in [c for c in CAND_ORDER if c in set(df["candidate"]) and c not in mode_a.BASELINES]:
        a = _by_pair(df, cand).reindex(base.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.abs(np.log(a["y_pred"] / a["y_true"])).to_numpy()
            lb = np.abs(np.log(base["y_pred"] / base["y_true"])).to_numpy()
        ape_d = np.abs(a["y_pred"] / a["y_true"] - 1).to_numpy() - np.abs(base["y_pred"] / base["y_true"] - 1).to_numpy()
        w = evaluate.paired_wilcoxon(la, lb)
        ok = np.isfinite(la) & np.isfinite(lb)
        rows.append({"candidate": cand, "n": int(ok.sum()), "p": w["p"], "median_dlog": w["median_diff"],
                     "median_dape": float(np.median(ape_d[np.isfinite(ape_d)])) if np.isfinite(ape_d).any() else float("nan")})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = evaluate.holm(out["p"])
    out["verdict"] = [verdict(m, p, noise) for m, p in zip(out["median_dape"], out["p_holm"])]
    return out


def show_paired(p: pd.DataFrame, baseline: str) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate": p["candidate"].map(SHORT), "vs best baseline": SHORT[baseline], "n": p["n"],
        "median paired diff, abs log error": p["median_dlog"].map(lambda v: f"{v:+.4f}"),
        "median paired diff, APE (pp)": p["median_dape"].map(lambda v: f"{100 * v:+.2f}"),
        "Wilcoxon p": p["p"].map(lambda v: f"{v:.3g}"), "Holm p": p["p_holm"].map(lambda v: f"{v:.3g}"),
        "verdict": p["verdict"]})


# ------------------------------------------------------------------ subgroups

def similarity_table(df: pd.DataFrame) -> pd.DataFrame:
    t = ordered(evaluate.subgroup_table(df, ["similarity_class", "candidate"]))
    t = t.sort_values("similarity_class", kind="stable")
    return pd.DataFrame({"similarity class": t["similarity_class"], "candidate": t["candidate"].map(SHORT), "n": t["n"],
                         "MdAPE %": t["mdape"].map(pct), "MAPE %": t["mape"].map(pct), "bias %": t["bias_pct"].map(pct),
                         "coverage90": t["coverage90"]})


def model_table(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per base model: AUTO and the best baseline."""
    rows = []
    for m, g in df.groupby("base_model", sort=True):
        a, b = g[g["candidate"] == mode_a.AUTO], g[g["candidate"] == baseline]
        ma, mb = evaluate.metrics(a["y_true"], a["y_pred"]), evaluate.metrics(b["y_true"], b["y_pred"])
        rows.append({"base model": m, "similarity class": ";".join(sorted(set(a["similarity_class"]))), "n": ma["n"],
                     "novelty": float(a["novelty"].median()) if len(a) else float("nan"),
                     "flagged": int(a["flagged"].sum()), "AUTO MdAPE %": pct(ma["mdape"]),
                     f"{SHORT[baseline]} MdAPE %": pct(mb["mdape"]), "AUTO bias %": pct(ma["bias_pct"]),
                     "AUTO coverage90": evaluate.coverage(a["y_true"], a["lo90"], a["hi90"])})
    return pd.DataFrame(rows)


def factor_table(df: pd.DataFrame, configs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """AUTO and the best baseline by precision, image size, threads, batch, clock and rig (fixed subgroups)."""
    cols = [c for c in FACTORS if c in configs]
    f = configs.drop_duplicates("config_key").set_index("config_key")[cols]
    a = df[df["candidate"] == mode_a.AUTO].join(f, on="config_key")
    b = df[df["candidate"] == baseline].join(f, on="config_key")
    rows = []
    for c in cols:
        for lvl, g in a.groupby(c, sort=True):
            ma = evaluate.metrics(g["y_true"], g["y_pred"])
            gb = b[b[c] == lvl]
            mb = evaluate.metrics(gb["y_true"], gb["y_pred"])
            rows.append({"factor": c, "level": lvl, "n": ma["n"], "AUTO MdAPE %": pct(ma["mdape"]),
                         "AUTO bias %": pct(ma["bias_pct"]), f"{SHORT[baseline]} MdAPE %": pct(mb["mdape"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ seed spread, worst 10, v2 vs pooled

def seed_table(seed_runs: dict, configs: pd.DataFrame, n_boot: int, boot_seed: int) -> tuple:
    """S-LOMO energy per protocol seed: AUTO and best baseline pooled MdAPE. Returns (table, raw rows)."""
    rows, raws = [], []
    for s, ref in sorted(seed_runs.items()):
        raw = pooled_table(ref.df, "S-LOMO", n_boot, boot_seed)
        bb = best_baseline(raw)
        a, b = raw.set_index("candidate").loc["AUTO"], raw.set_index("candidate").loc[bb]
        raws.append({"seed": s, "run_id": ref.run_id, "auto": a["mdape"], "baseline": b["mdape"]})
        rows.append({"seed": s, "run_id": ref.run_id, "n": int(a["n"]),
                     "AUTO MdAPE % [95 % CI]": pct_ci(a["mdape"], a["mdape_lo"], a["mdape_hi"]),
                     "best baseline": SHORT[bb], "baseline MdAPE % [95 % CI]": pct_ci(b["mdape"], b["mdape_lo"], b["mdape_hi"]),
                     "AUTO minus baseline (pp)": f"{100 * (a['mdape'] - b['mdape']):+.2f}"})
    return pd.DataFrame(rows), pd.DataFrame(raws)


def _agg_rules(s: pd.Series) -> str:
    return ";".join(sorted({r for v in s.dropna() for r in str(v).split(";") if r}))


def worst_table(df: pd.DataFrame, configs: pd.DataFrame, qc: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The `k` AUTO predictions with the largest APE, joined to the QC rows of the configuration's cells."""
    a = df[df["candidate"] == mode_a.AUTO].copy()
    a["ape"] = np.abs(a["y_pred"] / a["y_true"] - 1)
    a = a.sort_values("ape", ascending=False, kind="stable").head(k)
    cells = configs.drop_duplicates("config_key").set_index("config_key")["cell_ids"].astype(str)
    q = qc.set_index("cell_id")
    rows = []
    for rank, r in enumerate(a.itertuples(), 1):
        ids = [c for c in cells.get(r.config_key, "").split(";") if c]
        g = q.loc[[c for c in ids if c in q.index]] if len(ids) else q.iloc[:0]

        def agg(col, fn):
            return fn(g[col]) if col in g and len(g) else float("nan")
        rows.append({"rank": rank, "config_key": r.config_key, "similarity class": r.similarity_class,
                     "measured": r.y_true, "predicted": r.y_pred, "APE %": pct(r.ape), "novelty": r.novelty,
                     "flagged": bool(r.flagged), "n_cells": len(ids),
                     "QC all primary": bool(g["primary"].astype(bool).all()) if "primary" in g and len(g) else "",
                     "QC rules": _agg_rules(g["rules"]) if "rules" in g else "",
                     "max slip frac": agg("slip_scored_frac", lambda s: float(s.max())),
                     "max ctx-read frac": agg("ctx_read_frac_scored", lambda s: float(s.max())),
                     "max residual ms": agg("max_residual_ms", lambda s: float(s.max())),
                     "any throttle now": bool(g["throttle_current"].astype(bool).any()) if "throttle_current" in g and len(g) else "",
                     "max temp C": agg("max_temp_C", lambda s: float(s.max()))})
    return pd.DataFrame(rows)


def restrict_to_nov(pooled: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    """Rows of the pooled run whose test configuration is on the nov2026 rig: the v2-only run's test set."""
    nov = set(configs.loc[configs["rig"] == "nov2026", "config_key"])
    return pooled[pooled["config_key"].isin(nov)]


def v2_vs_pooled(pooled: pd.DataFrame, v2only: pd.DataFrame, configs: pd.DataFrame, n_boot: int, seed: int) -> tuple:
    """S-LOMO YOLO folds, energy: training on v1+v2 against training on `rig == nov2026` rows only, scored on
    the same test configurations (the nov2026 ones: the v2-only run cannot test on v1 rows). Returns
    (table, n, note on the paired AUTO comparison)."""
    p = restrict_to_nov(pooled, configs)
    got = set(zip(p.loc[p["candidate"] == "AUTO", "fold"], p.loc[p["candidate"] == "AUTO", "config_key"]))
    want = set(zip(v2only.loc[v2only["candidate"] == "AUTO", "fold"], v2only.loc[v2only["candidate"] == "AUTO", "config_key"]))
    if got != want:
        raise ValueError(f"pooled and v2-only runs do not score the same (fold, configuration) pairs: "
                         f"{len(got - want)} only in pooled, {len(want - got)} only in v2-only")
    a = pooled_table(p, "S-LOMO", n_boot, seed).set_index("candidate")
    b = pooled_table(v2only, "S-LOMO", n_boot, seed).set_index("candidate")
    rows = []
    for c in [c for c in CAND_ORDER if c in a.index and c in b.index]:
        rows.append({"candidate": SHORT[c], "n": int(b.loc[c, "n"]),
                     "pooled v1+v2 MdAPE % [95 % CI]": pct_ci(a.loc[c, "mdape"], a.loc[c, "mdape_lo"], a.loc[c, "mdape_hi"]),
                     "v2-only MdAPE % [95 % CI]": pct_ci(b.loc[c, "mdape"], b.loc[c, "mdape_lo"], b.loc[c, "mdape_hi"]),
                     "v2-only minus pooled (pp)": f"{100 * (b.loc[c, 'mdape'] - a.loc[c, 'mdape']):+.2f}"})
    pa = p[p["candidate"] == "AUTO"].set_index(["fold", "config_key"])
    pb = v2only[v2only["candidate"] == "AUTO"].set_index(["fold", "config_key"]).reindex(pa.index)
    w = evaluate.paired_wilcoxon(np.abs(np.log(pb["y_pred"] / pb["y_true"])).to_numpy(),
                                 np.abs(np.log(pa["y_pred"] / pa["y_true"])).to_numpy())
    n = int(len(pa))
    note = (f"AUTO, v2-only against pooled, paired by configuration (n = {n}): median paired difference in absolute log "
            f"error {w['median_diff']:+.4f}, Wilcoxon p = {w['p']:.3g} (one test, not part of a Holm family).")
    return pd.DataFrame(rows), n, note


def learning_table(lc: pd.DataFrame) -> pd.DataFrame:
    """S-LOMO MdAPE against the number of training YOLO models k: median and quartiles over (held-out model,
    training subset) runs, and over held-out models."""
    rows = []
    for k, g in lc.groupby("k", sort=True):
        by = g.groupby("heldout")["mdape"].median()
        rows.append({"k training YOLO models": int(k), "n": len(g), "held-out models": int(g["heldout"].nunique()),
                     "median MdAPE %": pct(g["mdape"].median()),
                     "IQR %": f"{pct(g['mdape'].quantile(.25))} to {pct(g['mdape'].quantile(.75))}",
                     "worst held-out model (median over subsets) %": f"{by.idxmax()} {pct(by.max())}"})
    return pd.DataFrame(rows)
