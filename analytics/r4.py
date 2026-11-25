"""R4 (docs/review/2026-11-25-r4-plan.md): deployment-energy analysis on the hooks-off cells, with v4.

The R4 Mode A target is deployment energy: only configurations measured with the layer hooks OFF enter (v3 calibration
and additions, the v4 replica and sweep, and lockbox 2, which was scored once with the frozen R3 models and is now
training data). Hooks-on cells are kept out of the fits and used for the instrumentation study (`hooks`).
Lockbox 3 stays out of everything.

  uv run python r4.py table                         reports/r4/mode_a_configs_r4.csv (Monsoon and board labels)
  uv run python r4.py mode-a [--split S] [--target T] [--label monsoon|board] [--n-jobs N]
  uv run python r4.py hooks                         reports/r4/hooks_by_clock.csv (hooks-on v2/v3 vs hooks-off v3/v4)
  uv run python r4.py mode-l [--n-jobs N]           Mode L leave-one-model-out on the v4 cells, kernel rungs included
  uv run python r4.py summary                       reports/r4/*.csv summaries of the runs above
  uv run python r4.py power-table                   reports/r4/power_table_r4.{csv,md}: measured vs held-out predictions

Run outputs go to derived/mode_a_r4/ and derived/mode_l_r4/; every run appends a runs.jsonl record (command
"mode-a-r4" / "mode-l-r4") with the table sha256, so a summary only reads runs of the current table.
"""
import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import data
import derive
import evaluate
import qc
import runlog
import settings
import static
import tables

HERE = Path(__file__).parent
OUT = HERE / "reports" / "r4"
TABLE = OUT / "mode_a_configs_r4.csv"
TRAIN_SOURCES = ("v3", "v4", "lockbox2")
HOLDOUTS = ("lockbox", "lockbox3")          # lockbox 1 cells are all hooks-on; lockbox 3 is R4's holdout
LOCKBOX2_MODELS = ["resnet34", "mbv3l", "effb1", "swins", "yolo12m"]
MODELS = data.VISION_MODELS + LOCKBOX2_MODELS
TARGETS = ("energy_mJ", "latency_ms", "mean_P_W")


def _qc_primary(cat: pd.DataFrame, der: pd.DataFrame) -> set:
    """QC-primary cell ids: reports/qc_cells.csv for the training sources, plus QC run here on lockbox 2 (cli qc
    leaves the holdout sources out)."""
    q = pd.read_csv(HERE / "reports" / "qc_cells.csv")
    ok = set(q.loc[q["primary"].astype(bool), "cell_id"])
    lb = cat[cat["source"] == "lockbox2"]
    if len(lb):
        lq = qc.run_qc(lb.reset_index(drop=True), der, qc.bounds_from_cfg())
        ok |= set(lq.loc[lq["primary"].astype(bool), "cell_id"])
    return ok


def deploy_cells(cat: pd.DataFrame) -> pd.DataFrame:
    """The catalogue rows of the R4 table: hooks-off image cells of TRAIN_SOURCES and MODELS, pinned clock, fan full,
    no serve rate, soak or concurrency (the R1-R3 Mode A rule without the lockbox exclusion)."""
    c = cat[cat["source"].isin(TRAIN_SOURCES)].copy()
    ok = ((c["task"] == "image") & c["base_model"].isin(MODELS) & c["serve_fps"].isna() & c["soak_minutes"].isna()
          & c["concurrent_mode"].isna() & (c["governor"] == "pinned") & (c["fan_mode"] == "full")
          & (c["layer_hooks"].astype(str) == "off"))
    c["mode_a_eligible"] = ok
    return c[ok].reset_index(drop=True)


def static_features(cat: pd.DataFrame) -> pd.DataFrame:
    extra = {**static.static_layer_tables(), **static.static_layer_tables(HERE / "lockbox2_static")}
    return static.static_table(cat, extra)


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(R4 configuration table, model descriptor)."""
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    ok = _qc_primary(cat, der)
    st = static_features(cat)
    qcf = pd.DataFrame({"cell_id": sorted(ok), "primary": True})
    t = tables.mode_a_table(deploy_cells(cat), der, qcf, st, holdouts=HOLDOUTS)
    return t, static.model_descriptor(st)


def load_table(label: str = "monsoon") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    t = pd.read_csv(TABLE)
    if label == "board":
        t = tables.board_view(t)
    sha = hashlib.sha256(t.to_csv(index=False).encode()).hexdigest()
    cat = data.catalogue()
    return t, static.model_descriptor(static_features(cat)), sha


def cmd_table(a) -> int:
    t, desc = build_table()
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLE, index=False)
    by = t.groupby("base_model").size()
    print(f"{len(t)} configurations, {t['base_model'].nunique()} base models, {int(t['n_cells'].sum())} cells -> {TABLE}")
    print(by.to_string())
    return 0


def cmd_mode_a(a) -> int:
    import mode_a
    import splits
    t, desc, sha = load_table(a.label)
    seed = settings.seed()
    sp = splits.make_splits(t, seed)
    names = [a.split] if a.split else list(sp)
    targets = [a.target] if a.target else list(TARGETS)
    out = settings.derived_dir() / "mode_a_r4"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        for target in targets:
            t0 = time.perf_counter()
            res = mode_a.run_split(t, sp[name], target, seed, desc, n_jobs=a.n_jobs)
            wall = time.perf_counter() - t0
            run_id = uuid.uuid4().hex
            res.to_parquet(out / f"{run_id}.parquet", index=False)
            runlog.append({"run_id": run_id, "command": "mode-a-r4", "label": a.label, "split": name, "target": target,
                           "seed": seed, "table_sha256": sha, "n_folds": len(sp[name]), "wall_s": round(wall, 1)})
            g = res[res["candidate"] == "AUTO_R4"]
            print(f"{a.label} {name} {target}: {len(sp[name])} folds, AUTO_R4 MdAPE "
                  f"{evaluate.metrics(g['y_true'], g['y_pred'])['mdape']:.4f}, {wall:.0f} s", flush=True)
    return 0


def latest_runs(command: str, **kw) -> dict:
    """{(label, split, target): parquet path} of the latest runs of the current R4 table(s)."""
    shas = {lab: load_table(lab)[2] for lab in ("monsoon", "board")}
    out = {}
    for line in (HERE / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("command") != command:
            continue
        lab = r.get("label", "monsoon")
        if command == "mode-a-r4" and r.get("table_sha256") != shas.get(lab):
            continue
        p = settings.derived_dir() / ("mode_a_r4" if command == "mode-a-r4" else "mode_l_r4") / f"{r['run_id']}.parquet"
        if p.is_file():
            out[(lab, r.get("split"), r.get("target"))] = p
    return out


def score(res: pd.DataFrame) -> pd.DataFrame:
    """Per candidate: MdAPE, MaxAPE, the per-model median (balanced), and the two 90 % coverages."""
    rows = []
    for c, g in res.groupby("candidate", sort=False):
        ape = (g["y_pred"] / g["y_true"] - 1).abs()
        inside = lambda lo, hi: float(((g["y_true"] >= g[lo]) & (g["y_true"] <= g[hi])).mean())  # noqa: E731
        rows.append({"candidate": c, "n": len(g), "mdape_pct": 100 * ape.median(), "maxape_pct": 100 * ape.max(),
                     "model_balanced_pct": 100 * ape.groupby(g["base_model"]).median().median(),
                     "cov90_pct": 100 * inside("lo90", "hi90"),
                     "cov90_cf_pct": 100 * inside("lo90_cf", "hi90_cf") if "lo90_cf" in g else np.nan,
                     "width_cf_median": float((g["hi90_cf"] / g["lo90_cf"]).median()) if "lo90_cf" in g else np.nan})
    return pd.DataFrame(rows)


def cmd_hooks(a) -> int:
    """Instrumentation study: each hooks-off R4 configuration against its hooks-on twin of the same rig (nov2026),
    energy, latency and power ratios per model and clock."""
    r3 = pd.read_csv(HERE / "reports" / "mode_a_configs.csv")
    on = r3[r3.get("layer_hooks", "on").astype(str) != "off"].set_index("config_key")
    t = pd.read_csv(TABLE)
    rows = []
    for r in t.itertuples():
        k = r.config_key.replace("|hooks=off", "")
        if k not in on.index or r.rig != "nov2026":
            continue
        o = on.loc[k]
        rows.append({"base_model": r.base_model, "variant": r.variant, "precision": r.precision, "imgsz": r.imgsz,
                     "threads": r.threads, "batch": r.batch, "cpu_mhz": r.cpu_mhz,
                     "energy_saved_pct": 100 * (1 - r.energy_mJ / o["energy_mJ"]),
                     "latency_saved_pct": 100 * (1 - r.latency_ms / o["latency_ms"]),
                     "power_change_pct": 100 * (r.mean_P_W / o["mean_P_W"] - 1),
                     "n_module_calls": r.n_module_calls, "latency_off_ms": r.latency_ms,
                     "calls_per_ms": r.n_module_calls * r.batch / (r.latency_ms * r.batch), "on_cells": int(o["n_cells"]), "off_cells": int(r.n_cells)})
    h = pd.DataFrame(rows).sort_values(["base_model", "precision", "imgsz", "threads", "batch", "cpu_mhz"])
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_csv(OUT / "hooks_by_clock.csv", index=False)
    print(h.round(2).to_string(index=False))
    return 0


def cmd_mode_l(a) -> int:
    import mode_l
    import mode_l_models as mlm
    import windows
    cat = data.catalogue()
    v4 = cat[(cat["source"] == "v4") & cat["has_counters"].astype(bool)]
    w = windows.build_windows(cat)
    w = w[w["cell_id"].isin(v4["cell_id"])]
    q = pd.read_csv(HERE / "reports" / "qc_cells.csv")
    rungs = {k: v for k, v in mlm.default_rungs().items()
             if k in ("L0", "L3-trees", "L3k-trees", "L5-cmos-trees", "L5k-trees")}
    t0 = time.perf_counter()
    res = mode_l.run_lomo(w, v4, settings.seed(), qc=q, rungs=rungs, n_jobs=a.n_jobs)
    wall = time.perf_counter() - t0
    out = settings.derived_dir() / "mode_l_r4"
    out.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    res.to_parquet(out / f"{run_id}.parquet", index=False)
    runlog.append({"run_id": run_id, "command": "mode-l-r4", "label": "monsoon", "split": "LOMO-v4", "target": "energy",
                   "seed": settings.seed(), "n_cells": int(res["cell_id"].nunique()), "wall_s": round(wall, 1)})
    print(f"{res['cell_id'].nunique()} cells, {res['fold'].nunique()} folds, {wall:.0f} s -> {out / run_id}.parquet")
    return 0


def cmd_summary(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (lab, split, target), p in sorted(latest_runs("mode-a-r4").items()):
        s = score(pd.read_parquet(p))
        rows.append(s.assign(label=lab, split=split, target=target))
    if rows:
        s = pd.concat(rows)[["label", "split", "target", "candidate", "n", "mdape_pct", "maxape_pct",
                             "model_balanced_pct", "cov90_pct", "cov90_cf_pct", "width_cf_median"]]
        s.to_csv(OUT / "mode_a_summary.csv", index=False)
        print(s[s["candidate"].isin(["AUTO", "AUTO_R4", "M1Ridge", "M4Physics", "M5Pooled", "ENS", "STACK"])]
              .round(2).to_string(index=False))
        # per held-out model, S-LOMO energy
        for lab in ("monsoon", "board"):
            p = latest_runs("mode-a-r4").get((lab, "S-LOMO", "energy_mJ"))
            if p is None:
                continue
            r = pd.read_parquet(p)
            r = r[r["candidate"].isin(["AUTO", "AUTO_R4", "M1Ridge", "M4Physics", "M5Pooled", "ENS", "STACK"])]
            r["ape_pct"] = 100 * (r["y_pred"] / r["y_true"] - 1).abs()
            pm = r.pivot_table(index="base_model", columns="candidate", values="ape_pct", aggfunc="median")
            pm.to_csv(OUT / f"lomo_by_model_{lab}.csv")
            if lab == "monsoon":
                print(pm.round(1).to_string())
    ml = latest_runs("mode-l-r4")
    if ml:
        r = pd.read_parquet(sorted(ml.values(), key=lambda p: p.stat().st_mtime)[-1])
        cols = [c for c in ("ape", "ape_meas") if c in r.columns]
        lm = r.groupby("rung")[cols].median().mul(100)
        fam = r.assign(family=r["fold"].map(lambda m: __import__("mode_a_models").family_of(m)))
        lf = fam.pivot_table(index="family", columns="rung", values=cols[-1], aggfunc="median").mul(100)
        lm.to_csv(OUT / "mode_l_v4_summary.csv")
        lf.to_csv(OUT / "mode_l_v4_by_family.csv")
        print(lm.round(2).to_string())
        print(lf.round(2).to_string())
    return 0


def _auto_pred(split: str, target: str, label: str = "monsoon") -> pd.DataFrame:
    p = latest_runs("mode-a-r4").get((label, split, target))
    if p is None:
        return pd.DataFrame(columns=["y_pred", "lo90_cf", "hi90_cf"])
    d = pd.read_parquet(p)
    return d[d["candidate"] == "AUTO"].groupby("config_key")[["y_pred", "lo90_cf", "hi90_cf"]].mean()


def cmd_power_table(a) -> int:
    """One row per R4 configuration: measured power and energy per image (Monsoon and board), the repeat spread, and
    the held-out AUTO predictions (S-LOMO: base model held out; RANDOM: only the configuration held out) with the
    conformal 90 % interval, plus the live model (best kernel rung, base model held out) where it ran."""
    t = pd.read_csv(TABLE)
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    f = der[der["mode"] == "forward"].set_index("cell_id")
    pw_new, pw_seen = _auto_pred("S-LOMO", "mean_P_W"), _auto_pred("RANDOM", "mean_P_W")
    e_new, e_seen = _auto_pred("S-LOMO", "energy_mJ"), _auto_pred("RANDOM", "energy_mJ")
    ml = latest_runs("mode-l-r4")
    live = None
    if ml:
        r = pd.read_parquet(sorted(ml.values(), key=lambda p: p.stat().st_mtime)[-1])
        r = r[r["rung"] == a.live_rung].set_index("cell_id")
        live = (r["E_pred_meas_J"] / r["E_true_meas_J"]).astype(float)
    rows = []
    for c in t.itertuples():
        ids = c.cell_ids.split(";")
        p = f.reindex(ids)["mean_P_W"].astype(float).dropna()
        sd = float(p.std(ddof=1)) if len(p) > 1 else np.nan
        k = c.config_key
        lr = live.reindex(ids).dropna() if live is not None else pd.Series(dtype=float)
        rows.append({"model": c.base_model, "precision": c.precision, "imgsz": int(c.imgsz), "cpu_mhz": int(c.cpu_mhz),
                     "threads": int(c.threads), "batch": int(c.batch), "n_cells": int(c.n_cells),
                     "measured_W": c.mean_P_W, "sd_W": sd, "cv_pct": 100 * sd / c.mean_P_W if len(p) > 1 else np.nan,
                     "board_W": c.mean_P_board_W,
                     "pred_new_model_W": pw_new["y_pred"].get(k, np.nan), "pred_seen_model_W": pw_seen["y_pred"].get(k, np.nan),
                     "pred_live_W": c.mean_P_W * float(lr.mean()) if len(lr) else np.nan,
                     "measured_mJ": c.energy_mJ, "pred_new_model_mJ": e_new["y_pred"].get(k, np.nan),
                     "lo90_new_mJ": e_new["lo90_cf"].get(k, np.nan), "hi90_new_mJ": e_new["hi90_cf"].get(k, np.nan),
                     "pred_seen_model_mJ": e_seen["y_pred"].get(k, np.nan), "config_key": k})
    o = pd.DataFrame(rows)
    for pcol, ecol, m in (("pred_new_model_W", "err_new_W_pct", "measured_W"), ("pred_seen_model_W", "err_seen_W_pct", "measured_W"),
                          ("pred_live_W", "err_live_W_pct", "measured_W"), ("pred_new_model_mJ", "err_new_mJ_pct", "measured_mJ"),
                          ("pred_seen_model_mJ", "err_seen_mJ_pct", "measured_mJ")):
        o[ecol] = 100 * (o[pcol] - o[m]) / o[m]
    o["inside90_new"] = (o["measured_mJ"] >= o["lo90_new_mJ"]) & (o["measured_mJ"] <= o["hi90_new_mJ"])
    o = o.sort_values(["model", "precision", "imgsz", "cpu_mhz", "threads", "batch"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_csv(OUT / "power_table_r4.csv", index=False)
    fm = lambda v, d, sg=False: "" if pd.isna(v) else (f"{v:+.{d}f}" if sg else f"{v:.{d}f}")  # noqa: E731
    h = ["model", "prec.", "img px", "MHz", "thr", "batch", "n", "measured W", "CV %", "board W", "pred new-model W", "err %",
         "pred seen-model W", "err %", "pred live W", "err %", "measured mJ", "pred new-model mJ [90 %]", "err %"]
    L = ["# R4 power and energy per configuration: deployment (hooks off), measured on the Pi 5 against held-out predictions\n",
         "Measured: mean forward-phase power (Monsoon) and energy per image over the configuration's hooks-off cells; CV over "
         "repeats; board W = Monsoon minus the harness I²R. **new-model**: R4 AUTO with the whole base model held out "
         "(S-LOMO), with the conformal 90 % interval. **seen-model**: R4 AUTO with only this configuration held out "
         f"(RANDOM). **live**: Mode L {a.live_rung}, base model held out, v4 cells only. err = (pred − measured) / measured. "
         "Generated by `analytics/r4.py power-table`; `power_table_r4.csv` has full precision.\n",
         "| " + " | ".join(h) + " |", "|" + "---|" * 6 + "---:|" * 13]
    for r in o.itertuples():
        iv = "" if pd.isna(r.lo90_new_mJ) else f" [{r.lo90_new_mJ:.0f}, {r.hi90_new_mJ:.0f}]"
        L.append("| " + " | ".join([r.model, r.precision, str(r.imgsz), str(r.cpu_mhz), str(r.threads), str(r.batch), str(r.n_cells),
                                     fm(r.measured_W, 2), fm(r.cv_pct, 2), fm(r.board_W, 2), fm(r.pred_new_model_W, 2),
                                     fm(r.err_new_W_pct, 1, True), fm(r.pred_seen_model_W, 2), fm(r.err_seen_W_pct, 1, True),
                                     fm(r.pred_live_W, 2), fm(r.err_live_W_pct, 1, True), fm(r.measured_mJ, 0),
                                     fm(r.pred_new_model_mJ, 0) + iv, fm(r.err_new_mJ_pct, 1, True)]) + " |")
    med = lambda c: o[c].abs().median()  # noqa: E731
    L += ["\n## Summary (median absolute error over configurations)\n",
          f"- power, new-model: {med('err_new_W_pct'):.2f} %; seen-model: {med('err_seen_W_pct'):.2f} %; "
          f"live: {med('err_live_W_pct'):.2f} % ({o['err_live_W_pct'].notna().sum()} configurations with v4 counter cells)",
          f"- energy per image, new-model: {med('err_new_mJ_pct'):.2f} % ({100 * o['inside90_new'].mean():.0f} % inside the 90 % "
          f"interval); seen-model: {med('err_seen_mJ_pct'):.2f} %",
          f"- repeat spread: median CV {o['cv_pct'].median():.2f} %; harness share {100 * (1 - o['board_W'] / o['measured_W']).median():.1f} %"]
    (OUT / "power_table_r4.md").write_text("\n".join(L) + "\n")
    print(f"{len(o)} rows -> {OUT / 'power_table_r4.csv'}")
    print("\n".join(L[-4:]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    p = sub.add_parser("mode-a")
    p.add_argument("--split")
    p.add_argument("--target", choices=TARGETS)
    p.add_argument("--label", choices=["monsoon", "board"], default="monsoon")
    p.add_argument("--n-jobs", type=int, default=4)
    sub.add_parser("hooks")
    p = sub.add_parser("mode-l")
    p.add_argument("--n-jobs", type=int, default=4)
    sub.add_parser("summary")
    p = sub.add_parser("power-table")
    p.add_argument("--live-rung", default="L5-cmos-trees")
    a = ap.parse_args(argv)
    return {"table": cmd_table, "mode-a": cmd_mode_a, "hooks": cmd_hooks, "mode-l": cmd_mode_l,
            "summary": cmd_summary, "power-table": cmd_power_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
