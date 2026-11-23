"""Mode L, PMIC-only, probe-mode and deployment-validation tables of the results report (pure functions).
Exploratory rungs carry the suffix "(exploratory)" in their name cell."""
import numpy as np
import pandas as pd

import evaluate
from report_tables import P_SIG, pct, pct_ci, verdict

RUNG_ORDER = ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees"]
PHASES = ["idle", "load_weights", "warmup", "forward", "serve"]
EXPLORATORY_NOTE = ("`L1-cell` is exploratory: it was added after a scored smoke fold (protocol amendment of "
                    "2026-11-23, `analytics/PROTOCOL.md`). `L1` is the pre-registered PMIC rung.")


def rung_label(rung: str, exploratory: bool) -> str:
    return f"{rung} (exploratory)" if exploratory else rung


def _rank(s: pd.Series) -> pd.Series:
    return s.map(lambda r: RUNG_ORDER.index(r) if r in RUNG_ORDER else 99)


def sort_rungs(df: pd.DataFrame, col: str = "rung") -> pd.DataFrame:
    return df.sort_values(col, key=_rank, kind="stable").reset_index(drop=True)


def _expl(df: pd.DataFrame) -> dict:
    return df.drop_duplicates("rung").set_index("rung")["exploratory"].astype(bool).to_dict()


def _boot(d, g, fn, n_boot, seed):
    return evaluate.cluster_bootstrap(d, g, fn, n_boot, seed=seed)


def ladder_raw(lomo: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Per rung over held-out cells: whole-cell and measured-segment MdAPE with cluster-bootstrap intervals
    (clusters = held-out base models), medians of the 1 s and 10 s block APE, window MAE, L1 fallback share."""
    ex = _expl(lomo)
    rows = []
    for rung, g in lomo.groupby("rung", sort=False):
        whole = g[["fold", "E_true_J", "E_pred_J"]].reset_index(drop=True)
        meas = g[["fold", "E_true_meas_J", "E_pred_meas_J"]].reset_index(drop=True)
        mw, mm = evaluate.metrics(g["E_true_J"], g["E_pred_J"]), evaluate.metrics(g["E_true_meas_J"], g["E_pred_meas_J"])
        r = {"rung": rung, "exploratory": ex[rung], "n": mw["n"], "mdape": mw["mdape"], "mape": mw["mape"],
             "bias_pct": mw["bias_pct"], "n_meas": mm["n"], "mdape_meas": mm["mdape"],
             "ape_1s": float(g["ape_1s"].median()), "ape_10s": float(g["ape_10s"].median()),
             "mae_window_W": float(g["mae_window_W"].median()),
             "l1_fallback": float(g["l1_fallback_frac"].mean()) if g["l1_fallback_frac"].notna().any() else float("nan")}
        r["mdape_lo"], r["mdape_hi"] = _boot(whole, "fold", lambda x: evaluate.metrics(x["E_true_J"], x["E_pred_J"])["mdape"], n_boot, seed)
        r["bias_lo"], r["bias_hi"] = _boot(whole, "fold", lambda x: evaluate.metrics(x["E_true_J"], x["E_pred_J"])["bias_pct"], n_boot, seed)
        r["meas_lo"], r["meas_hi"] = _boot(meas, "fold", lambda x: evaluate.metrics(x["E_true_meas_J"], x["E_pred_meas_J"])["mdape"], n_boot, seed)
        rows.append(r)
    return sort_rungs(pd.DataFrame(rows))


def show_ladder(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "rung": [rung_label(r, e) for r, e in zip(raw["rung"], raw["exploratory"])], "n": raw["n"],
        "whole-cell MdAPE % [95 % CI]": [pct_ci(*r) for r in zip(raw["mdape"], raw["mdape_lo"], raw["mdape_hi"])],
        "whole-cell bias % [95 % CI]": [pct_ci(*r) for r in zip(raw["bias_pct"], raw["bias_lo"], raw["bias_hi"])],
        "measured-segment MdAPE % [95 % CI]": [pct_ci(*r) for r in zip(raw["mdape_meas"], raw["meas_lo"], raw["meas_hi"])],
        "n measured": raw["n_meas"], "1 s block APE %": raw["ape_1s"].map(pct), "10 s block APE %": raw["ape_10s"].map(pct),
        "window MAE W": raw["mae_window_W"], "L1 fallback share": raw["l1_fallback"]})


def phase_table(lomo: pd.DataFrame) -> pd.DataFrame:
    """Median whole-phase energy APE per rung and phase over held-out cells that have the phase."""
    ex, rows = _expl(lomo), []
    for rung, g in lomo.groupby("rung", sort=False):
        r = {"rung": rung_label(rung, ex[rung]), "n": len(g), "_k": rung}
        for ph in PHASES:
            v = g[f"ape_phase_{ph}"].dropna()
            r[ph] = f"{pct(v.median())} (n={len(v)})" if len(v) else ""
        rows.append(r)
    return sort_rungs(pd.DataFrame(rows), "_k").drop(columns="_k")


def family_table(lomo: pd.DataFrame, family: pd.Series) -> pd.DataFrame:
    """Median whole-cell APE per rung and dataset family of the held-out cell."""
    d = lomo.assign(family=lomo["cell_id"].map(family).astype(object).fillna("unknown"))
    fams = sorted(d["family"].unique())
    ex, rows = _expl(lomo), []
    for rung, g in d.groupby("rung", sort=False):
        r = {"rung": rung_label(rung, ex[rung]), "n": len(g), "_k": rung}
        for f in fams:
            v = g.loc[g["family"] == f, "ape"].dropna()
            r[f] = f"{pct(v.median())} (n={len(v)})" if len(v) else ""
        rows.append(r)
    return sort_rungs(pd.DataFrame(rows), "_k").drop(columns="_k")


def vs_l0_raw(lomo: pd.DataFrame, noise: float, base: str = "L0") -> pd.DataFrame:
    """Each other rung against L0, paired by held-out cell: Wilcoxon on whole-cell APE, Holm over the family
    (all other rungs, `L1-cell` included for conservatism), verdict as for Mode A."""
    ex = _expl(lomo)
    ref = lomo[lomo["rung"] == base].set_index("cell_id")["ape"]
    rows = []
    for rung, g in lomo.groupby("rung", sort=False):
        if rung == base:
            continue
        a = g.set_index("cell_id")["ape"].reindex(ref.index)
        w = evaluate.paired_wilcoxon(a.to_numpy(), ref.to_numpy())
        rows.append({"rung": rung, "exploratory": ex[rung], "n": int((a.notna() & ref.notna()).sum()),
                     "p": w["p"], "median_d": w["median_diff"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = sort_rungs(out)
    out["p_holm"] = evaluate.holm(out["p"])
    out["verdict"] = [verdict(m, p, noise) for m, p in zip(out["median_d"], out["p_holm"])]
    return out


def show_vs_l0(out: pd.DataFrame, base: str = "L0") -> pd.DataFrame:
    return pd.DataFrame({
        "rung": [rung_label(r, e) for r, e in zip(out["rung"], out["exploratory"])], "vs": base, "n": out["n"],
        "median paired diff, APE (pp)": out["median_d"].map(lambda v: f"{100 * v:+.2f}"),
        "Wilcoxon p": out["p"].map(lambda v: f"{v:.3g}"), "Holm p": out["p_holm"].map(lambda v: f"{v:.3g}"),
        "verdict": out["verdict"]})


def pmic_table(pm: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """L1 (and exploratory L1-cell) leave-one-workload-out: all cells, v1 cells, v2 cells and each family.
    Interval: cluster bootstrap over `model_key` (workloads)."""
    ex, rows = _expl(pm), []
    scopes = [("all cells", lambda d: d), ("v1 cells", lambda d: d[d["family"] == "v1"]),
              ("v2 cells", lambda d: d[d["family"] != "v1"])]
    scopes += [(f"v2 family {f}", lambda d, f=f: d[d["family"] == f]) for f in sorted(set(pm["family"]) - {"v1"})]
    for rung, g in pm.groupby("rung", sort=False):
        for name, sel in scopes:
            s = sel(g)
            if s.empty:
                continue
            d = s[["model_key", "E_true_J", "E_pred_J"]].reset_index(drop=True)
            m = evaluate.metrics(d["E_true_J"], d["E_pred_J"])
            lo, hi = _boot(d, "model_key", lambda x: evaluate.metrics(x["E_true_J"], x["E_pred_J"])["mdape"], n_boot, seed)
            rows.append({"rung": rung_label(rung, ex[rung]), "scope": name, "n": m["n"],
                         "workloads": int(s["model_key"].nunique()), "MdAPE % [95 % CI]": pct_ci(m["mdape"], lo, hi),
                         "bias %": pct(m["bias_pct"]), "median slope b": float(s["slope"].median()),
                         "L1 fallback share": float(s["l1_fallback_frac"].mean()), "_k": rung})
    return sort_rungs(pd.DataFrame(rows), "_k").drop(columns="_k")


def probe_raw(probe: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Probe mode against Mode A on the same (configuration, j, k) pairs: MdAPE with intervals (clusters = base
    models) and a paired Wilcoxon over configurations (per-configuration median error), Holm over rungs."""
    p = probe[np.isfinite(probe["mode_a_mJ"]) & np.isfinite(probe["probe_mJ"])].copy()
    p["base_model"] = p["config_key"].str.split("|").str[0]
    ex, rows = _expl(probe), []
    for rung, g in p.groupby("rung", sort=False):
        d = g[["base_model", "true_mJ", "probe_mJ", "mode_a_mJ"]].reset_index(drop=True)
        mp = evaluate.metrics(d["true_mJ"], d["probe_mJ"])
        ma = evaluate.metrics(d["true_mJ"], d["mode_a_mJ"])
        lp = _boot(d, "base_model", lambda x: evaluate.metrics(x["true_mJ"], x["probe_mJ"])["mdape"], n_boot, seed)
        la = _boot(d, "base_model", lambda x: evaluate.metrics(x["true_mJ"], x["mode_a_mJ"])["mdape"], n_boot, seed)
        c = g.groupby("config_key")[["ape", "mode_a_ape"]].median()
        w = evaluate.paired_wilcoxon(c["ape"].to_numpy(), c["mode_a_ape"].to_numpy())
        rows.append({"rung": rung, "exploratory": ex[rung], "n": mp["n"], "configs": len(c), "p": w["p"],
                     "probe": (mp["mdape"], *lp), "modea": (ma["mdape"], *la),
                     "median_d": float((c["ape"] - c["mode_a_ape"]).median())})
    out = sort_rungs(pd.DataFrame(rows))
    if not out.empty:
        out["p_holm"] = evaluate.holm(out["p"])
        out["better"] = [bool(h < P_SIG and m < 0) for h, m in zip(out["p_holm"], out["median_d"])]
    return out


def show_probe(out: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "rung": [rung_label(r, e) for r, e in zip(out["rung"], out["exploratory"])], "n pairs": out["n"],
        "n configurations": out["configs"],
        "probe MdAPE % [95 % CI]": [pct_ci(*t) for t in out["probe"]],
        "Mode A MdAPE % [95 % CI]": [pct_ci(*t) for t in out["modea"]],
        "median paired diff, probe minus Mode A (pp)": out["median_d"].map(lambda v: f"{100 * v:+.2f}"),
        "Wilcoxon p": out["p"].map(lambda v: f"{v:.3g}"), "Holm p": out["p_holm"].map(lambda v: f"{v:.3g}"),
        "probe better?": ["yes" if b else "no" for b in out["better"]]})


def deploy_summary(dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, phase), g in dep.groupby(["case", "phase"], sort=True):
        lo, hi = g.get("lo90_J"), g.get("hi90_J")
        have = lo.notna() & hi.notna() if lo is not None else pd.Series(False, index=g.index)
        inside = ((g["measured_J"] >= lo) & (g["measured_J"] <= hi))[have] if have.any() else pd.Series(dtype=bool)
        rows.append({"case": case, "phase": phase, "n": len(g), "median APE %": pct(g["ape"].median()),
                     "max APE %": pct(g["ape"].max()),
                     "90 % interval covers (n with interval)": f"{int(inside.sum())} of {int(have.sum())}",
                     "notes": "; ".join(sorted({str(x) for x in g["notes"].dropna()}))[:160]})
    return pd.DataFrame(rows)


def deploy_rows(dep: pd.DataFrame) -> pd.DataFrame:
    """Row-level view without the bulk `serve-default` case (it is in the CSV)."""
    d = dep[dep["case"] != "serve-default"]
    return pd.DataFrame({"case": d["case"], "cell_id": d["cell_id"], "phase": d["phase"], "measured J": d["measured_J"],
                         "predicted J": d["predicted_J"], "APE %": d["ape"].map(pct), "lo90 J": d["lo90_J"],
                         "hi90 J": d["hi90_J"], "flagged": d["flagged"].map(lambda v: "" if pd.isna(v) else ("yes" if v else "no"))})
