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


def add_derive_args(p) -> None:
    p.add_argument("--force", action="store_true", help="ignore the cache and recompute")


# name -> (help, handler, optional function adding arguments to the subparser)
COMMANDS = {
    "catalogue": ("write reports/catalogue.csv and print counts by source and family", cmd_catalogue, None),
    "derive": ("write the derive.parquet cache; print row counts and the worst raw vs derived.json "
               "difference", cmd_derive, add_derive_args),
    "qc": ("apply data-quality rules Q1-Q6 to every cell; write reports/qc_cells.csv and reports/qc.md",
           cmd_qc, None),
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
