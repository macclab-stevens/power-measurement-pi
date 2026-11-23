"""Analytics entry point: `cd analytics && uv run python cli.py <subcommand>`.

Later tasks add a subcommand by writing one `cmd_<name>(args)` function (plus an optional
`add_<name>_args(parser)`) and one line in COMMANDS.
"""
import argparse
import sys

import settings

REPORTS = settings.HERE / "reports"


def cmd_catalogue(args) -> int:
    import data
    df = data.catalogue()
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "catalogue.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} cells -> {out}")
    print("by source:")
    print(df.groupby("source").size().to_string())
    print("by family:")
    print(df.groupby(["source", "family"]).size().to_string())
    return 0


def cmd_derive(args) -> int:
    import data
    import derive
    import settings as st
    cat = data.catalogue()
    df = derive.derive_all(cat, force=args.force)
    print(f"{len(df)} rows from {df['cell_id'].nunique()} cells -> {st.derived_dir() / derive.CACHE_NAME}")
    print(df.groupby("mode").size().to_string())
    worst, worst_cell = 0.0, None
    for _, c in cat.iterrows():
        dj = data.read_json_opt(c["path"], "derived.json")
        sub = df[(df["cell_id"] == c["cell_id"]) & (df["n_frames"] > 0)]
        if dj is None or len(sub) != 1:
            continue
        r = sub.iloc[0]
        for got, want in ((r["energy_mJ_per_image_raw"], dj.get("energy_per_image_mJ")),
                          (r["latency_ms_per_image_raw"] * r["batch"], dj.get("latency_ms_per_frame"))):
            if want:
                rel = abs(got - want) / abs(want)
                if rel > worst:
                    worst, worst_cell = rel, c["cell_id"]
    print(f"worst relative difference raw vs derived.json: {worst:.3e}"
          + (f" ({worst_cell})" if worst_cell else ""))
    return 0


def cmd_qc(args) -> int:
    import resource
    import time
    import data
    import derive
    import qc
    import runlog
    t0 = time.perf_counter()
    cat = data.catalogue()
    der = derive.derive_all(cat)
    bounds = qc.bounds_from_cfg()
    df = qc.run_qc(cat, der, bounds)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "qc_cells.csv", index=False)
    (REPORTS / "qc.md").write_text(qc.report_md(df, cat, bounds))
    wall = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KiB on Linux
    print(f"{len(df)} cells -> {REPORTS / 'qc_cells.csv'}, {REPORTS / 'qc.md'}")
    print(f"wall time {wall:.1f} s, peak RSS {rss_mb:.0f} MB")
    print(qc.rule_counts(df, cat).to_string())
    print(qc.quantiles(df).to_string())
    labels = df.attrs["phase_labels"]
    print("phase labels (cells; distinct raw):", ", ".join(
        f"{k!r}({v}; {df.attrs['phase_distinct'][k]})" for k, v in sorted(labels.items())))
    print("unknown phase labels:", sorted(x for x in labels if not qc.is_known_phase(x)) or "none")
    runlog.append({"command": "qc", "bounds": bounds, "n_cells": len(df), "n_include": int(df["include"].sum()),
                   "n_primary": int(df["primary"].sum()), "wall_s": round(wall, 1), "peak_rss_mb": round(rss_mb)})
    return 0


def cmd_tables(args) -> int:
    import data
    import derive
    import pandas as pd
    import runlog
    import static
    import tables
    qc_path = REPORTS / "qc_cells.csv"
    if not qc_path.is_file():
        print(f"error: {qc_path} is missing; run `uv run python cli.py qc` first", file=sys.stderr)
        return 1
    cat = data.catalogue()
    der = derive.derive_all(cat)
    t = tables.mode_a_table(cat, der, pd.read_csv(qc_path), static.static_table(cat))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "mode_a_configs.csv"
    t.to_csv(out, index=False)
    nf = tables.noise_floor(t)
    print(f"{len(t)} configurations from {int(t['n_cells'].sum())} cells -> {out}")
    print("configurations per base model:")
    print(t.groupby("base_model").size().to_string())
    src = cat.set_index("cell_id")["source"]
    by_src = t["cell_ids"].map(lambda ids: "+".join(sorted({src[i] for i in ids.split(";")})))
    print("configurations per source:")
    print(by_src.value_counts().to_string())
    print(f"noise floor (median energy_cv, n_cells >= 2): {nf:.5f}")
    runlog.append({"command": "tables", "n_configs": len(t), "noise_floor": nf})
    return 0


def cmd_splits(args) -> int:
    import pandas as pd
    import runlog
    import splits
    src = REPORTS / "mode_a_configs.csv"
    if not src.is_file():
        print(f"error: {src} is missing; run `uv run python cli.py tables` first", file=sys.stderr)
        return 1
    t = pd.read_csv(src)
    seed = settings.seed()
    sp = splits.make_splits(t, seed)
    outdir = settings.HERE / "splits"
    hashes = splits.write_splits(sp, outdir)
    for name, folds in sp.items():
        print(f"{name}: {len(folds)} folds -> {outdir / (name + '.csv')}")
        for f in folds:
            print(f"  {f.fold} [{f.similarity_class}] train {len(f.train_keys)} test {len(f.test_keys)}")
    for name, h in hashes.items():
        print(f"sha256 {h}  {name}")
    runlog.append({"command": "splits", "seed": seed, "n_configs": len(t), "sha256": hashes})
    return 0


def _mode_a_inputs():
    """(configuration table, model descriptor) for Mode A, or None if the table is missing."""
    import data
    import pandas as pd
    import static
    src = REPORTS / "mode_a_configs.csv"
    if not src.is_file():
        print(f"error: {src} is missing; run `uv run python cli.py tables` first", file=sys.stderr)
        return None
    return pd.read_csv(src), static.model_descriptor(static.static_table(data.catalogue()))


def cmd_mode_a(args) -> int:
    import tempfile
    import time
    import uuid
    import evaluate
    import mode_a
    import runlog
    import splits
    inputs = _mode_a_inputs()
    if inputs is None:
        return 1
    t, desc = inputs
    seed = args.seed if args.seed is not None else settings.seed()
    sp = splits.make_splits(t, seed)
    names = [args.split] if args.split else list(sp)
    unknown = [n for n in names if n not in sp]
    if unknown:
        print(f"error: unknown split {unknown}; expected one of {list(sp)}", file=sys.stderr)
        return 1
    targets = [args.target] if args.target else list(mode_a.CANDIDATES_BY_TARGET)
    table_sha = runlog.file_hash(REPORTS / "mode_a_configs.csv")
    out = settings.derived_dir() / "mode_a"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:   # hash of the folds actually used (RANDOM depends on seed)
        used = splits.write_splits({n: sp[n] for n in names}, tmp)
    for name in names:
        committed = settings.HERE / "splits" / f"{name}.csv"
        for target in targets:
            t0 = time.perf_counter()
            res = mode_a.run_split(t, sp[name], target, seed, desc, n_jobs=args.n_jobs)
            wall = time.perf_counter() - t0
            per = {c: {**evaluate.metrics(g["y_true"], g["y_pred"]),
                       "coverage90": evaluate.coverage(g["y_true"], g["lo90"], g["hi90"])}
                   for c, g in res.groupby("candidate", sort=False)}
            run_id = uuid.uuid4().hex
            res.to_parquet(out / f"{run_id}.parquet", index=False)
            runlog.append({"run_id": run_id, "command": "mode-a", "split": name, "target": target, "seed": seed,
                           "split_sha256": used[f"{name}.csv"],
                           "split_file_sha256": runlog.file_hash(committed) if committed.is_file() else None,
                           "table_sha256": table_sha, "n_jobs": args.n_jobs, "metrics": per,
                           "wall_s": round(wall, 1)})
            auto = per["AUTO"]
            print(f"{name} {target}: {len(sp[name])} folds, AUTO MdAPE {auto['mdape']:.4f}, "
                  f"coverage {auto['coverage90']:.2f}, {wall:.0f} s -> {out / (run_id + '.parquet')}")
    return 0


def cmd_deploy_validate(args) -> int:
    import data
    import derive
    import pandas as pd
    import deploy
    import runlog
    qc_path = REPORTS / "qc_cells.csv"
    if not qc_path.is_file():
        print(f"error: {qc_path} is missing; run `uv run python cli.py qc` first", file=sys.stderr)
        return 1
    try:
        preds = deploy.load_mode_a_preds()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cat = data.catalogue()
    v = deploy.validate(cat, derive.derive_all(cat), pd.read_csv(qc_path), preds)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "deploy_validation.csv"
    v.to_csv(out, index=False)
    print(f"{len(v)} rows from {len(preds)} held-out predictions -> {out}")
    summ = v.groupby(["case", "phase"])["ape"].agg(n="size", median="median", max="max")
    print(summ.to_string())
    print("skipped (no prediction):", v.attrs["skipped"] or "none")
    print("fan offsets:", v.attrs["fan_offsets"])
    print("soak factors (leave-one-soak-out):", v.attrs["soak_factors"])
    runlog.append({"command": "deploy-validate", "n_rows": len(v), "skipped": v.attrs["skipped"],
                   "median_ape": {f"{c}/{p}": float(r["median"]) for (c, p), r in summ.iterrows()}})
    return 0


def _energy_check(df, cat) -> str:
    """Kept-window energy vs an independent per-window recomputation from the trace (finite mean x
    n_samples x dt, the same non-finite handling as windows._label), for one counter cell."""
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import data
    cid = "yolo11n_480_t4_2400_r00"
    cid = cid if cid in set(df["cell_id"]) else df["cell_id"].iloc[0]
    cell = Path(cat.loc[cat["cell_id"] == cid, "path"].iloc[0])
    w = df[df["cell_id"] == cid]
    tr = data.read_trace(cell, ["t_pi_ns", "P_W"])
    dt = 1.0 / json.loads((cell / "alignment.json").read_text())["slip_rate_hz"]
    t, P = tr["t_pi_ns"].to_numpy(), tr["P_W"].to_numpy(dtype=float)
    ends, starts = w["t_end_ns"].to_numpy(), w["t_start_ns"].to_numpy()
    wid = np.searchsorted(ends, t, side="left")
    ok = wid < len(ends)
    ok[ok] = t[ok] > starts[wid[ok]]
    g = pd.DataFrame({"w": wid[ok], "P": P[ok]}).groupby("w")["P"]
    trace_e = float((g.mean() * g.size() * dt * 1000).sum())     # mean skips NaN; imputed for dropouts
    kept = float(w["E_mJ"].sum())
    return (f"{cid}: window E sum {kept:.1f} mJ, trace E {trace_e:.1f} mJ, rel diff "
            f"{abs(kept - trace_e) / trace_e:.3e} ({int((~np.isfinite(P)).sum())} non-finite samples)")


def cmd_windows(args) -> int:
    import time
    import data
    import runlog
    import windows
    t0 = time.perf_counter()
    cat = data.catalogue()
    w = windows.build_windows(cat, force=args.force)
    p = windows.build_pmic_windows(cat, force=args.force)
    wall = time.perf_counter() - t0
    rss_mb = windows._peak_rss_mb()
    a = w.attrs
    print(f"windows: {len(w)} rows from {w['cell_id'].nunique()} cells -> "
          f"{settings.derived_dir() / windows.WINDOWS_CACHE}")
    print(f"pmic windows: {len(p)} rows from {p['cell_id'].nunique()} cells -> "
          f"{settings.derived_dir() / windows.PMIC_CACHE}")
    print(f"dropped (cover_frac < {windows.MIN_COVER}): {a.get('dropped')} of {a.get('n_intervals')} intervals")
    print(f"guarded windows (zero denominator): {a.get('guards')}; by feature {a.get('guards_by_feature')}")
    print(f"not-counted windows (counted == 0, rates set to 0): {a.get('not_counted')}")
    print("counted / not counted by phase (top 12 phases by windows):")
    by = w.groupby("phase")["counted"].agg(counted="sum", n="size")
    by["not_counted"] = by["n"] - by["counted"]
    print(by.sort_values("n", ascending=False).head(12)[["counted", "not_counted", "n"]].to_string())
    print(f"PMIC read availability fallbacks (t_avail < t_read): windows cells {a.get('pmic_avail_fallback')}, "
          f"pmic cells {p.attrs.get('pmic_avail_fallback')}")
    print(f"min pmic_age_s {w['pmic_age_s'].min():.6f}, NaN pmic_age_s {int(w['pmic_age_s'].isna().sum())}")
    print(f"PMIC pairs per cell (pairs: n windows-cells): {a.get('pmic_pairs')}; pmic {p.attrs.get('pmic_pairs')}")
    print(f"wall time {wall:.1f} s, peak RSS {rss_mb:.0f} MB")
    print(_energy_check(w, cat))
    runlog.append({"command": "windows", "n_windows": len(w), "n_pmic_windows": len(p),
                   "dropped": a.get("dropped"), "guards": a.get("guards"), "wall_s": round(wall, 1),
                   "peak_rss_mb": round(rss_mb)})
    return 0


def add_derive_args(p) -> None:
    p.add_argument("--force", action="store_true", help="ignore the cache and recompute")


def add_mode_a_args(p) -> None:
    p.add_argument("--split", help="one of S-LOMO S-LSO S-LVO S-LCO RANDOM (default: all)")
    p.add_argument("--target", choices=["energy_mJ", "latency_ms", "mean_P_W"], help="default: all three")
    p.add_argument("--seed", type=int, help="default: the config.toml seed")
    p.add_argument("--n-jobs", type=int, default=2, help="outer folds in parallel (default 2)")


# name -> (help, handler, optional function adding arguments to the subparser)
COMMANDS = {
    "catalogue": ("write reports/catalogue.csv and print counts by source and family", cmd_catalogue, None),
    "derive": ("write the derive.parquet cache; print row counts and the worst raw vs derived.json "
               "difference", cmd_derive, add_derive_args),
    "qc": ("apply data-quality rules Q1-Q6 to every cell; write reports/qc_cells.csv and reports/qc.md",
           cmd_qc, None),
    "tables": ("write reports/mode_a_configs.csv (one row per Mode A configuration; needs reports/"
               "qc_cells.csv from `qc`)", cmd_tables, None),
    "splits": ("write analytics/splits/<split>.csv from reports/mode_a_configs.csv; print sha256 per file",
               cmd_splits, None),
    "mode-a": ("nested Mode A evaluation per split and target -> derived/mode_a/<run_id>.parquet and runs.jsonl",
               cmd_mode_a, add_mode_a_args),
    "deploy-validate": ("validate the deployment calculator on serve, ondemand, fan and soak cells using the "
                        "latest S-LOMO Mode A runs -> reports/deploy_validation.csv", cmd_deploy_validate, None),
    "windows": ("build derived/windows.parquet (100 ms counter windows) and derived/pmic_windows.parquet "
                "(5 s PMIC windows); print drop, guard and memory figures", cmd_windows, add_derive_args),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, (help_, fn, add_args) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_)
        if add_args:
            add_args(sp)
        sp.set_defaults(fn=fn)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
