"""R2 against R1 comparison of the Mode L ladder -> reports/r2/mode_l_r2.md.

R2 is the latest Mode L leave-one-YOLO-model-out run that contains the R2 rungs (`L5-cmos`), made on the
slip-corrected windows with selection on the inference-segment error; R1 is the run behind reports/results.md
(its run_id is read from that file). Pure formatting in `build`; `main` only reads runs.jsonl, the run
parquets and the window cache attributes. Usage: `uv run python mode_l_r2_report.py [--r1-run ID] [--r2-run ID]`."""
import argparse
import re

import numpy as np
import pandas as pd

import evaluate
import mode_l
import report_inputs as ri
import settings

OUT = settings.HERE / "reports" / "r2" / "mode_l_r2.md"
RUNG_ORDER = ["L0", "L1", "L1-cell", "L2", "L3-ridge", "L3-trees", "L4-ridge", "L4-trees", "L5-cmos", "L5-cmos-trees"]
R2_MARK = "L5-cmos"
_RUN_ID = re.compile(r"run_id `([0-9a-f]{32})`")


def _pct(x, d=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def _num(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def r1_run_id(results_md: str) -> str | None:
    """The Mode L run_id quoted in a results.md: the first one on a line that mentions Mode L."""
    for line in results_md.splitlines():
        if "Mode L" in line and (m := _RUN_ID.search(line)):
            return m.group(1)
    return None


def rung_stats(lomo: pd.DataFrame) -> pd.DataFrame:
    """One row per rung of a `run_lomo` frame, in ladder order: whole-cell MdAPE / P90 / MaxAPE, inference-segment
    MdAPE, median 1 s block APE (fractions) and, when the frame carries the R2 sums, forward-window MAE (W) and
    R^2 over the forward windows and over all windows (NaN for an R1 frame)."""
    rows = {}
    for r, g in lomo.groupby("rung", sort=False):
        w = evaluate.metrics(g["E_true_J"], g["E_pred_J"])
        row = {"n": len(g), "whole": w["mdape"], "p90": w["p90ape"], "max": w["maxape"],
               "meas": evaluate.metrics(g["E_true_meas_J"], g["E_pred_meas_J"])["mdape"],
               "block_1s": float(g["ape_1s"].median())}
        row.update(mode_l.pooled_window_metrics(g) if "n_fwd" in g.columns
                   else {"mae_fwd_W": np.nan, "r2_fwd": np.nan, "r2_all": np.nan})
        rows[r] = row
    out = pd.DataFrame(rows).T
    return out.reindex([r for r in RUNG_ORDER if r in out.index] + [r for r in out.index if r not in RUNG_ORDER])


def slip_summary(attrs: dict) -> dict:
    """Size of the slip correction on the window labels from the windows-cache attrs (`slip_cells` etc.)."""
    cells = pd.DataFrame(attrs.get("slip_cells") or [])
    out = {"n_corrected": int(attrs.get("slip_corrected", 0)), "n_no_pulses": int(attrs.get("slip_no_pulses", 0)),
           "n_windows_total": int(attrs.get("n_intervals", 0)) - int(attrs.get("dropped", 0))}
    if cells.empty:
        return out
    out.update(n_windows=int(cells["n_windows"].sum()), n_changed=int(cells["n_windows_changed"].sum()),
               n_gt_0p1W=int(cells["n_windows_gt_0p1W"].sum()),
               med_of_cell_med_W=float(cells["dP_med_W"].median()), max_of_cell_med_W=float(cells["dP_med_W"].max()),
               med_of_cell_p99_W=float(cells["dP_p99_W"].median()), max_of_cell_p99_W=float(cells["dP_p99_W"].max()),
               max_W=float(cells["dP_max_W"].max()), max_corr_ms=float(cells["max_corr_ms"].max()),
               med_corr_ms=float(cells["max_corr_ms"].median()))
    return out


def _table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def build(r1: pd.DataFrame | None, r2: pd.DataFrame, slip: dict, run_ids: dict, params: dict | None = None) -> str:
    """Markdown from the R1 and R2 `run_lomo` frames (R1 may be None), the `slip_summary` and
    {"r1": id, "r2": id}; `params` maps rung -> {fold: chosen grid point} for the R2 run (optional)."""
    s2 = rung_stats(r2)
    s1 = rung_stats(r1) if r1 is not None else None
    L = ["# Mode L, R2 against R1 (slip-corrected window labels, CMOS features, selection on the inference segment)\n",
         "Exploratory R2 revision; the lockbox has been seen, so nothing here is a pre-registered result. "
         "Leave one YOLO base model out, QC-primary cells. Percent columns are percent; MdAPE is the median "
         "of |E_pred - E_true| / E_true over cells. **Whole** is the whole-cell energy, **inference** the forward "
         "(measured) phase only, **1 s block** the median over 10-window blocks. The forward-window MAE (W) and the R^2 "
         "are over the 100 ms windows of the held-out cells pooled together (the forward R^2 uses forward "
         "windows only; the all-window R^2 is inflated by the idle / active bimodality and is shown for "
         "comparison). R1 did not record window sums, so its R^2 and MAE are blank.\n",
         f"Runs: R2 `{run_ids.get('r2')}`, R1 `{run_ids.get('r1')}`. R2 changes: window times corrected for sample-loss "
         "slips (cells collected before the align.py fix), `ctx_sw_rate` removed from every rung (identically 0), "
         "rungs `L5-cmos` / `L5-cmos-trees` (L3 features + llc_miss_rate, f_eff_mhz, vcore_eff, v2_cycles_rate, "
         "v2_instr_rate), grid points chosen on the median inference-segment APE instead of the whole-cell APE.\n",
         "## Ladder: R2 with R1 beside it\n"]
    hdr = ["rung", "n cells", "whole MdAPE", "inference MdAPE", "1 s block MdAPE", "fwd MAE (W)", "fwd R^2",
           "all-window R^2", "R1 whole", "R1 inference", "R1 1 s block"]
    rows = []
    for r, x in s2.iterrows():
        o = s1.loc[r] if s1 is not None and r in s1.index else None
        rows.append([r, str(int(x["n"])), _pct(x["whole"]), _pct(x["meas"]), _pct(x["block_1s"]),
                     _num(x["mae_fwd_W"]), _num(x["r2_fwd"]), _num(x["r2_all"]),
                     *(["-"] * 3 if o is None else [_pct(o["whole"]), _pct(o["meas"]), _pct(o["block_1s"])])])
    L += _table(hdr, rows) + [""]
    L += ["## Whole-cell error tail (R2)\n"]
    L += _table(["rung", "MdAPE", "P90", "MaxAPE"], [[r, _pct(x["whole"]), _pct(x["p90"]), _pct(x["max"], 1)]
                                                   for r, x in s2.iterrows()]) + [""]
    if params:
        L += ["## Grid points chosen per held-out base model (R2)\n"]
        for r, d in params.items():
            L.append(f"- `{r}`: " + "; ".join(f"{k}: {v}" for k, v in sorted(d.items())))
        L.append("")
    L += ["## Size of the slip correction on the window labels\n"]
    if not slip.get("n_corrected"):
        L += [f"No cell was corrected ({slip.get('n_no_pulses', 0)} slip cells skipped for lack of sync pulses).\n"]
    else:
        L += [f"- Cells corrected: {slip['n_corrected']} (slip cells without >= 12 sync pulse events, left as "
              f"stored: {slip['n_no_pulses']}). Cells already corrected by the collection code are untouched.",
              f"- Largest time shift per cell: median {slip['med_corr_ms']:.1f} ms, maximum {slip['max_corr_ms']:.1f} ms.",
              f"- |dP| per 100 ms window (corrected axis against stored axis), over the {slip['n_windows']} windows of "
              f"the corrected cells: the median over cells of the cell's median is {slip['med_of_cell_med_W']:.4f} W "
              f"(largest cell median {slip['max_of_cell_med_W']:.4f} W), the median over cells of the cell's p99 is "
              f"{slip['med_of_cell_p99_W']:.3f} W (largest {slip['max_of_cell_p99_W']:.3f} W), the largest single "
              f"window {slip['max_W']:.3f} W.",
              f"- {slip['n_changed']} of those windows changed at all, {slip['n_gt_0p1W']} by more than 0.1 W.", ""]
    return "\n".join(L)


def _latest_r2(recs, paths, seed, run_id=None):
    best = None
    for r in recs:
        if r.get("command") == "mode-l" and r.get("part") == "lomo" and r.get("seed") == seed \
                and R2_MARK in (r.get("rungs") or []) and (run_id in (None, r["run_id"])) \
                and (paths.mode_l_dir / f"{r['run_id']}.parquet").is_file():
            best = r
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r1-run", help="R1 Mode L run id (default: the one quoted in reports/results.md)")
    ap.add_argument("--r2-run", help="R2 Mode L run id (default: the latest run with the L5 rungs)")
    args = ap.parse_args(argv)
    import data
    import windows
    paths = ri.Paths.default()
    recs = ri.read_records(paths.runs_log)
    seed = settings.seed()
    rec = _latest_r2(recs, paths, seed, args.r2_run)
    if rec is None:
        raise SystemExit(f"no R2 Mode L run (seed {seed}, rung {R2_MARK}); run `uv run python cli.py mode-l`")
    res = settings.HERE / "reports" / "results.md"
    r1_id = args.r1_run or (r1_run_id(res.read_text()) if res.is_file() else None)
    r1 = None
    if r1_id and (paths.mode_l_dir / f"{r1_id}.parquet").is_file():
        r1 = pd.read_parquet(paths.mode_l_dir / f"{r1_id}.parquet")
    r2 = pd.read_parquet(paths.mode_l_dir / f"{rec['run_id']}.parquet")
    pr = {}
    for rung, g in r2.groupby("rung", sort=False):
        if len(set(g["params"])) > 1 or g["params"].iloc[0] != "{}":
            pr[rung] = dict(zip(g["fold"], g["params"]))
    slip = slip_summary(windows.build_windows(data.catalogue()).attrs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(r1, r2, slip, {"r1": r1_id, "r2": rec["run_id"]}, pr))
    print(f"wrote {OUT}: R2 run {rec['run_id']}, R1 run {r1_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
