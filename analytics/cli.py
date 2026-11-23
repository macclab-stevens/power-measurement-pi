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


# name -> (help, handler, optional function adding arguments to the subparser)
COMMANDS = {
    "catalogue": ("write reports/catalogue.csv and print counts by source and family", cmd_catalogue, None),
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
