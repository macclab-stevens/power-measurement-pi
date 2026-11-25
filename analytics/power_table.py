"""Power per configuration: measured against predicted, for every model, image size, clock and precision.

One row per measured configuration (model x precision x image size x clock x threads x batch x hooks x rig) of the
training data (v1 + v2 + v3), plus the two lockboxes. Measured: mean forward-phase power over the configuration's cells
(`derive`, slip-excluded), its standard deviation and coefficient of variation across those repeats, and the number of
cells. Predicted (each from a model that did not see the configuration):

  pred_new_model_W    R3 Mode A AUTO with the whole base model held out (S-LOMO run)
  pred_seen_model_W   R3 Mode A AUTO with only this configuration held out (RANDOM run; the model's other configs seen)
  pred_live_W         R3 Mode L L3-trees, base model held out (leave-one-model-out run): the predicted forward-segment
                      energy scaled to power (measured power x E_pred / E_true); cells with counters only
  frozen rows         lockbox 1 (R1 frozen primary, M1Ridge) and lockbox 2 (R3 frozen AUTO): predictions written
                      before the cells were measured; pred_new_model_W holds them

Errors are signed percent (pred - measured) / measured. Run from analytics/:  uv run python power_table.py
Writes reports/power_table.csv and reports/power_table.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import derive
import report_inputs as ri
import runlog
import settings

HERE = Path(__file__).parent
OUT_CSV = HERE / "reports" / "power_table.csv"
OUT_MD = HERE / "reports" / "power_table.md"
COLS = ["source", "model", "precision", "imgsz", "cpu_mhz", "threads", "batch", "hooks", "rig", "n_cells",
        "measured_W", "sd_W", "cv_pct", "pred_new_model_W", "err_new_pct", "pred_seen_model_W", "err_seen_pct",
        "pred_live_W", "err_live_pct", "config_key"]


def _latest(recs, paths, **kw):
    hits = [r for r in recs if all(r.get(k) == v for k, v in kw.items()) and (paths.mode_a_dir / f"{r['run_id']}.parquet").is_file()]
    return hits[-1] if hits else None


def cell_power(der: pd.DataFrame) -> pd.Series:
    f = der[der["mode"] == "forward"].set_index("cell_id")
    return f["mean_P_W"].astype(float)


def measured(cell_ids, pw: pd.Series) -> tuple:
    v = pw.reindex(cell_ids).dropna().to_numpy(float)
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    return float(v.mean()), sd, 100 * sd / v.mean() if len(v) > 1 else np.nan, len(v)


def build() -> pd.DataFrame:
    paths = ri.Paths.default()
    recs = ri.read_records(paths.runs_log)
    sha = runlog.file_hash(paths.configs_csv)
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    pw = cell_power(der)
    t = pd.read_csv(paths.configs_csv)

    def auto(split):
        rec = _latest(recs, paths, command="mode-a", split=split, target="mean_P_W", seed=settings.seed(), table_sha256=sha)
        if rec is None:
            raise SystemExit(f"no R3 {split} mean_P_W run for the current table; run `uv run python cli.py mode-a --split {split}`")
        d = pd.read_parquet(paths.mode_a_dir / f"{rec['run_id']}.parquet")
        return d[d["candidate"] == "AUTO"].groupby("config_key")["y_pred"].mean()
    new, seen = auto("S-LOMO"), auto("RANDOM")

    import mode_l_r2_report as mlr
    mrec = mlr._latest_r2(recs, paths, settings.seed(), None)
    ml = pd.read_parquet(paths.mode_l_dir / f"{mrec['run_id']}.parquet")
    ml = ml[ml["rung"] == "L3-trees"].set_index("cell_id")
    ratio = (ml["E_pred_meas_J"] / ml["E_true_meas_J"]).astype(float)

    rows = []
    for r in t.itertuples():
        ids = str(r.cell_ids).split(";")
        m, sd, cv, n = measured(ids, pw)
        rr = ratio.reindex(ids).dropna()
        live = m * float(rr.mean()) if len(rr) else np.nan
        rows.append({"source": "training (v1/v2/v3)", "model": r.base_model, "precision": r.precision, "imgsz": int(r.imgsz),
                     "cpu_mhz": int(r.cpu_mhz), "threads": int(r.threads), "batch": int(r.batch),
                     "hooks": getattr(r, "layer_hooks", "on"), "rig": r.rig, "n_cells": n, "measured_W": m, "sd_W": sd,
                     "cv_pct": cv, "pred_new_model_W": new.get(r.config_key, np.nan),
                     "pred_seen_model_W": seen.get(r.config_key, np.nan), "pred_live_W": live, "config_key": r.config_key})

    # lockbox 1: R1 frozen primary predictions; lockbox 2: R3 frozen AUTO
    lb1 = pd.read_csv(HERE / "reports" / "lockbox_predictions.csv")
    lb1 = lb1[(lb1["role"] == "primary") & (lb1["target"] == "mean_P_W")].set_index(["model", "imgsz"])["pred"]
    lb2 = pd.read_csv(HERE / "reports" / "r3" / "lockbox2_predictions.csv")
    lb2 = lb2[(lb2["candidate"] == "AUTO") & (lb2["target"] == "mean_P_W")].set_index(["model", "imgsz"])["pred"]
    for src, frozen, label in (("lockbox", lb1, "lockbox 1 (frozen R1)"), ("lockbox2", lb2, "lockbox 2 (frozen R3)")):
        c = cat[cat["source"] == src]
        for (mdl, s), g in c.groupby(["base_model", "imgsz"]):
            m, sd, cv, n = measured(list(g["cell_id"]), pw)
            first = g.iloc[0]
            rows.append({"source": label, "model": mdl, "precision": first["precision"], "imgsz": int(s), "cpu_mhz": int(first["cpu_mhz"]),
                         "threads": int(first["threads"]), "batch": int(first["batch"]), "hooks": first["layer_hooks"], "rig": first["rig"],
                         "n_cells": n, "measured_W": m, "sd_W": sd, "cv_pct": cv,
                         "pred_new_model_W": float(frozen.get((mdl, int(s)), np.nan)), "pred_seen_model_W": np.nan,
                         "pred_live_W": np.nan, "config_key": ""})
    out = pd.DataFrame(rows)
    for p, e in (("pred_new_model_W", "err_new_pct"), ("pred_seen_model_W", "err_seen_pct"), ("pred_live_W", "err_live_pct")):
        out[e] = 100 * (out[p] - out["measured_W"]) / out["measured_W"]
    order = {"training (v1/v2/v3)": 0, "lockbox 1 (frozen R1)": 1, "lockbox 2 (frozen R3)": 2}
    out = out.sort_values(["source", "model", "precision", "imgsz", "cpu_mhz", "threads", "batch", "hooks", "rig"],
                          key=lambda s: s.map(order) if s.name == "source" else s).reset_index(drop=True)
    return out[COLS]


def fmt(v, d, signed=False):
    return "" if pd.isna(v) else (f"{v:+.{d}f}" if signed else f"{v:.{d}f}")


def to_md(df: pd.DataFrame) -> str:
    h = ["model", "prec.", "img px", "MHz", "thr", "batch", "hooks", "rig", "n", "measured W", "sd W", "CV %",
         "pred new-model W", "err %", "pred seen-model W", "err %", "pred live W", "err %"]
    L = ["# Power per configuration: measured against predicted\n",
         "Mean forward-phase power of the Raspberry Pi 5 (Monsoon, W) per configuration, its spread across repeats, and "
         "the predictions of models that did not see the configuration. **new-model**: R3 a priori predictor (AUTO) with "
         "the whole base model held out. **seen-model**: the same predictor with only this configuration held out. "
         "**live**: R3 live model (L3-trees, counters every 100 ms) with the base model held out, where counters were "
         "recorded. Lockbox rows: predictions frozen before measurement. err = (pred - measured) / measured. "
         "Generated by `analytics/power_table.py`; the CSV beside this file has full precision.\n"]
    for src, g in df.groupby("source", sort=False):
        L += [f"\n## {src} ({len(g)} configurations)\n", "| " + " | ".join(h) + " |", "|" + "---|" * 8 + "---:|" * 10]
        for r in g.itertuples():
            L.append("| " + " | ".join([r.model, r.precision, str(r.imgsz), str(r.cpu_mhz), str(r.threads), str(r.batch), str(r.hooks),
                                         r.rig.replace("2026", ""), str(r.n_cells), fmt(r.measured_W, 2), fmt(r.sd_W, 3), fmt(r.cv_pct, 2),
                                         fmt(r.pred_new_model_W, 2), fmt(r.err_new_pct, 1, True), fmt(r.pred_seen_model_W, 2),
                                         fmt(r.err_seen_pct, 1, True), fmt(r.pred_live_W, 2), fmt(r.err_live_pct, 1, True)]) + " |")
    tr = df[df["source"].str.startswith("training")]
    L += ["\n## Summary (median absolute error, training configurations)\n",
          f"- new-model: {tr['err_new_pct'].abs().median():.2f} % over {tr['err_new_pct'].notna().sum()} configurations",
          f"- seen-model: {tr['err_seen_pct'].abs().median():.2f} % over {tr['err_seen_pct'].notna().sum()} configurations",
          f"- live: {tr['err_live_pct'].abs().median():.2f} % over {tr['err_live_pct'].notna().sum()} configurations with counters",
          f"- repeat spread: median CV {tr['cv_pct'].median():.2f} % (sd across repeats of one configuration)"]
    return "\n".join(L) + "\n"


def main() -> int:
    df = build()
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(to_md(df))
    print(f"{len(df)} rows -> {OUT_CSV}, {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
