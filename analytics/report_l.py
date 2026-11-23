"""Mode L, PMIC-only, probe-mode and counter-overhead sections of the results report."""
import pandas as pd

import eda_write as ew
import report_figs as figs
import report_tables as rt
import report_tables_l as tl

LIMITS = [
    "Counters cover the measured process's user-space code only (`perf_event_paranoid=2`). Other processes are invisible, so Mode L claims "
    "hold for a dedicated single-process deployment.",
    "Counter data is YOLO-only until the lockbox adds non-YOLO cells; Mode L folds are the five YOLO models with counter cells "
    "(yolov8n has none: its v2 cells are the counters-off `ref` family).",
    "Perf runs during deployment too, so its overhead is inside both the training data and any later use.",
    "One Raspberry Pi 5 and one Monsoon rig; nothing here transfers to other boards or meters.",
]


def ladder(a) -> pd.DataFrame:
    if "ladder" not in a.cache:
        a.cache["ladder"] = tl.ladder_raw(a.ml["lomo"], a.n_boot, a.boot_seed)
    return a.cache["ladder"]


def _vs_l0(a) -> pd.DataFrame:
    if "vs_l0" not in a.cache:
        a.cache["vs_l0"] = tl.vs_l0_raw(a.ml["lomo"], a.noise["energy_mJ"])
    return a.cache["vs_l0"]


def _probe(a) -> pd.DataFrame:
    if "probe" not in a.cache:
        a.cache["probe"] = tl.probe_raw(a.ml["probe"], a.n_boot, a.boot_seed)
    return a.cache["probe"]


def overhead_text(a) -> str:
    o = a.overhead
    if o is None or len(o) == 0:
        return ("Counter overhead: perf was running in every v2 counter cell, so it is inside every number here; the only "
                "counters-off comparison is the `ref` family (`reports/eda.md`, section 7.2), which bounds it loosely.")
    return (f"Counter overhead: counters-off (`ref`) cells drew {o['delta_W'].mean():+.3f} W ({o['delta_pct'].mean():+.2f} %) against the same "
            f"{ew.plural(len(o), 'configuration')} with counters on (`reports/eda.md`, section 7.2; family, order and the counters switch are "
            "confounded, so this bounds the overhead only loosely). Perf was running in every v2 counter cell, so the overhead is inside every number here.")


def _prov(a) -> str:
    lomo = a.ml["lomo"]
    return f"n = {lomo['cell_id'].nunique()} held-out cells in {lomo['fold'].nunique()} folds (dataset v2, nov2026 rig)"


def _cap(a, extra: str) -> str:
    return rt.caption(split="leave one YOLO base model out", target="whole-cell energy", prov=_prov(a),
                      run_ids=a.ml["run_id"], extra=extra)


def _seed_matched(pr: pd.DataFrame) -> str:
    v = pr["mode_a_seed_matched"].dropna().astype(bool)
    return "unknown" if v.empty else "yes" if v.all() else "no (the Mode A run used another seed)"


def _probe_best(a) -> str | None:
    p = a.ml["probe"]
    p = p[~p["exploratory"].astype(bool)]
    if p.empty:
        return None
    return str(p.groupby("rung")["ape"].median().idxmin())


def findings(a) -> list:
    out = []
    for r in _vs_l0(a).itertuples():
        out.append(f"Mode L, whole-cell energy: {tl.rung_label(r.rung, r.exploratory)} against L0: **{r.verdict}** (median paired APE "
                   f"difference {100 * r.median_d:+.2f} pp, Holm p = {r.p_holm:.3g}, n = {r.n}; run_id `{a.ml['run_id']}`).")
    for r in _probe(a).itertuples():
        out.append(f"Probe mode, {tl.rung_label(r.rung, r.exploratory)}: probe MdAPE {rt.pct(r.probe[0])} % against Mode A {rt.pct(r.modea[0])} % "
                   f"on the same pairs; {'the probe beats Mode A' if r.better else 'the probe does not beat Mode A'} "
                   f"(Holm p = {r.p_holm:.3g}, n = {r.n} pairs; run_id `{a.ml['run_id']}`).")
    return out


def sections(a) -> list:
    lo, ml = a.ml["lomo"], a.ml
    raw = ladder(a)
    box = figs.mode_l_boxplot(lo, a.paths.fig_dir)
    L = ["## 9. Mode L: live power ladder\n",
         "Rungs in protocol order; each is fitted on the other YOLO base models and scored on the held-out model's cells (whole-cell energy "
         "from 100 ms window power predictions). Intervals are cluster bootstrap over the held-out base models (5 clusters, so wide). "
         f"{tl.EXPLORATORY_NOTE} {overhead_text(a)}\n",
         _cap(a, "Whole cell, measured segment, 1 s and 10 s horizons."), ew.md_table(tl.show_ladder(raw), {"window MAE W": "{:.3f}", "L1 fallback share": "{:.3f}"}) + "\n",
         _cap(a, "Median phase energy APE per rung and phase (cells that have the phase)."), ew.md_table(tl.phase_table(lo)) + "\n",
         _cap(a, "Median whole-cell APE per rung and dataset family."), ew.md_table(tl.family_table(lo, a.family)) + "\n",
         _cap(a, "Each rung against L0, paired by held-out cell, Holm over the family; "
              + rt.floor_text(a.noise["energy_mJ"], "energy_mJ") + " (Mode A's energy floor, used here as the indistinguishability margin)."), ew.md_table(tl.show_vs_l0(_vs_l0(a))) + "\n",
         _cap(a, "Whole-cell energy APE per rung."), figs.fig_link(a.paths, box, "Mode L whole-cell APE per rung")]
    pm = ml["pmic_v1"]
    n1, n2 = (pm.drop_duplicates("cell_id")["family"] == "v1").sum(), (pm.drop_duplicates("cell_id")["family"] != "v1").sum()
    L += ["## 10. PMIC-only check\n",
          "`L1` (pre-registered) and `L1-cell` (exploratory) fitted on all other workloads (`model_key`) and scored on the held-out workload, "
          "over v1 and v2 cells. v1 has no counters, so this is the one cross-family test of the PMIC rung with current data.\n",
          rt.caption(split="leave one workload (model_key) out", target="whole-cell energy",
                     prov=f"n = {pm['cell_id'].nunique()} cells (dataset v1 {n1}, v2 {n2} cells)", run_ids=ml["run_id"],
                     extra=tl.EXPLORATORY_NOTE),
          ew.md_table(tl.pmic_table(pm, a.n_boot, a.boot_seed), {"median slope b": "{:.3f}", "L1 fallback share": "{:.3f}"}) + "\n"]
    pt = _probe(a)
    best = _probe_best(a)
    pr = ml["probe"]
    L += ["## 11. Probe mode against Mode A\n",
          "For configurations with at least two repeats, Mode L predicts repeat k's energy per image from repeat j's measured segment (j != k) and is "
          "compared with the Mode A S-LOMO AUTO prediction for the same configuration. Pairs are not independent, so the paired test is run over "
          "configurations (per-configuration median error).\n",
          rt.caption(split="S-LOMO (Mode A) and leave one YOLO base model out (Mode L)", target="energy_mJ per image",
                     prov=f"n = {len(pr)} ordered pairs of repeats in {pr['config_key'].nunique()} configurations (dataset v2)",
                     run_ids=[ml["run_id"], *sorted({str(x) for x in pr["mode_a_run_id"].dropna()})],
                     extra=f"Second run id: the Mode A S-LOMO energy run the comparison used; seed matched: "
                           f"{_seed_matched(pr)}. {tl.EXPLORATORY_NOTE}"),
          ew.md_table(tl.show_probe(pt)) + "\n"]
    if best:
        name = figs.probe_scatter(pr, best, a.paths.fig_dir)
        L += [rt.caption(split="probe against Mode A", target="energy_mJ per image", prov=f"n = {int((pr['rung'] == best).sum())} pairs",
                         run_ids=ml["run_id"], extra=f"Rung {best} is the non-exploratory rung with the lowest probe median APE (selected post hoc)."),
              figs.fig_link(a.paths, name, "probe and Mode A predictions against measured energy per image")]
    L += ["## 12. Counter overhead and stated limits\n", overhead_text(a) + "\n"]
    if a.overhead is not None and len(a.overhead):
        L += [rt.caption(split="counters off (ref) against on", target="forward mean power, W",
                         prov=f"n = {int(a.overhead['n_off'].sum() + a.overhead['n_on'].sum())} cells in {len(a.overhead)} configurations (dataset v2)",
                         run_ids="not a model run (descriptive)"), ew.md_table(a.overhead) + "\n"]
    L += ["Stated limits of Mode L:\n", *[f"- {x}" for x in LIMITS], "\n"]
    return L

