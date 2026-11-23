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


def add_derive_args(p) -> None:
    p.add_argument("--force", action="store_true", help="ignore the cache and recompute")


# name -> (help, handler, optional function adding arguments to the subparser)
COMMANDS = {
    "catalogue": ("write reports/catalogue.csv and print counts by source and family", cmd_catalogue, None),
    "derive": ("write the derive.parquet cache; print row counts and the worst raw vs derived.json "
               "difference", cmd_derive, add_derive_args),
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
