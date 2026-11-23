"""Figures of the results report (PNG, 150 dpi, via plotstyle.save). Each function returns the file name."""
import os

import numpy as np
import pandas as pd

import plotstyle as ps
from report_tables_l import RUNG_ORDER, rung_label


def error_vs_novelty(auto: pd.DataFrame, figdir) -> str:
    """S-LOMO energy, AUTO: absolute percentage error against novelty, coloured by similarity class; the
    applicability threshold of each fold is its own, so flagged predictions are drawn hollow."""
    a = auto.copy()
    a["ape"] = np.abs(a["y_pred"] / a["y_true"] - 1) * 100
    fig, ax = ps.subplots()
    for i, (cls, g) in enumerate(a.groupby("similarity_class", sort=True)):
        fl = g["flagged"].astype(bool)
        c = ps.COLORS[i % len(ps.COLORS)]
        ax.scatter(g.loc[~fl, "novelty"], g.loc[~fl, "ape"], color=c, marker=ps.MARKERS[i % len(ps.MARKERS)],
                   s=22, label=f"{cls} (n={len(g)})")
        ax.scatter(g.loc[fl, "novelty"], g.loc[fl, "ape"], facecolors="none", edgecolors=c,
                   marker=ps.MARKERS[i % len(ps.MARKERS)], s=26, label=f"{cls}, flagged no-similar-model (n={int(fl.sum())})"
                   if fl.any() else None)
    ax.set_xlabel("novelty (distance to the nearest training model, standardized static features)")
    ax.set_ylabel("absolute percentage error, % (energy per image)")
    ax.set_title("S-LOMO, AUTO: error against novelty")
    ax.legend()
    return ps.save(fig, "report_error_vs_novelty", figdir).name


def learning_curve(lc: pd.DataFrame, figdir) -> str:
    fig, ax = ps.subplots()
    for i, (m, g) in enumerate(lc.groupby("heldout", sort=True)):
        s = g.groupby("k")["mdape"].median() * 100
        ax.plot(s.index, s.values, color="#b8b7b0", lw=1, marker=ps.MARKERS[i % len(ps.MARKERS)], ms=4)
        ax.annotate(m, (s.index[-1], s.values[-1]), textcoords="offset points", xytext=(4, 0), fontsize=7, color=ps.MUTED)
    med = lc.groupby("k")["mdape"].median() * 100
    ax.plot(med.index, med.values, color=ps.COLORS[0], lw=2.4, marker="o", label="median over runs")
    ax.set_xticks(sorted(lc["k"].unique()))
    ax.set_xlabel("training YOLO models (k)")
    ax.set_ylabel("MdAPE, % (energy per image)")
    ax.set_title("Learning curve, S-LOMO YOLO folds (grey: one line per held-out model)")
    ax.legend()
    return ps.save(fig, "report_learning_curve", figdir).name


def mode_l_boxplot(lomo: pd.DataFrame, figdir) -> str:
    """Whole-cell energy APE per rung over held-out cells."""
    ex = lomo.drop_duplicates("rung").set_index("rung")["exploratory"].astype(bool).to_dict()
    rungs = [r for r in RUNG_ORDER if r in ex] + [r for r in ex if r not in RUNG_ORDER]
    data = [lomo.loc[lomo["rung"] == r, "ape"].dropna().to_numpy() * 100 for r in rungs]
    fig, ax = ps.subplots(size=(7.2, 3.8))
    bp = ax.boxplot(data, tick_labels=[f"{rung_label(r, ex[r])}\nn={len(d)}".replace(" (exploratory)", "\n(exploratory)")
                                       for r, d in zip(rungs, data)], patch_artist=True, showfliers=True,
                    medianprops={"color": ps.INK})
    for box, r in zip(bp["boxes"], rungs):
        box.set(facecolor="#dfe9f7" if not ex[r] else "#fbe3d6", edgecolor=ps.MUTED)
    ax.set_ylabel("whole-cell energy APE, %")
    ax.set_title("Mode L ladder, leave one YOLO model out")
    ax.tick_params(axis="x", labelsize=7)
    return ps.save(fig, "report_mode_l_ape", figdir).name


def probe_scatter(probe: pd.DataFrame, rung: str, figdir) -> str:
    """Predicted against measured energy per image: probe (`rung`) and Mode A on the same pairs."""
    p = probe[(probe["rung"] == rung)]
    p = p[np.isfinite(p["probe_mJ"]) & np.isfinite(p["mode_a_mJ"])]
    fig, ax = ps.subplots()
    ax.scatter(p["true_mJ"], p["mode_a_mJ"], s=20, color=ps.COLORS[1], marker="s", label="Mode A (S-LOMO AUTO)")
    ax.scatter(p["true_mJ"], p["probe_mJ"], s=20, color=ps.COLORS[0], marker="o", label=f"probe, {rung}")
    lim = [min(p["true_mJ"].min(), p["probe_mJ"].min(), p["mode_a_mJ"].min()) * 0.9,
           max(p["true_mJ"].max(), p["probe_mJ"].max(), p["mode_a_mJ"].max()) * 1.1] if len(p) else [1, 10]
    ax.plot(lim, lim, color=ps.MUTED, lw=1, ls="--", label="perfect")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("measured energy per image of the other repeat, mJ")
    ax.set_ylabel("predicted energy per image, mJ")
    ax.set_title(f"Probe mode against Mode A (n={len(p)} pairs; rung chosen post hoc: lowest probe MdAPE)")
    ax.legend()
    return ps.save(fig, "report_probe_vs_mode_a", figdir).name


def fig_link(paths, name: str, alt: str) -> str:
    """Markdown image of figure `name` relative to the report's folder."""
    return f"![{alt}]({os.path.relpath(paths.fig_dir, paths.out_dir)}/{name})\n"
