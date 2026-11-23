"""CLI commands that produce the report inputs (`mode-a-extras`) and the report itself (`report`).
Registered in cli.py's COMMANDS."""
import sys

import settings


def cmd_mode_a_extras(args) -> int:
    import lockbox
    try:
        lockbox.refuse_if_lockbox_cells("mode-a-extras")
    except lockbox.LockboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    import data
    import pandas as pd
    import report_inputs as ri
    import static
    import mode_a_extras
    paths = ri.Paths.default()
    if not paths.configs_csv.is_file():
        print(f"error: {paths.configs_csv} is missing; run `uv run python cli.py tables` first", file=sys.stderr)
        return 1
    seed = args.seed if args.seed is not None else settings.seed()
    t = pd.read_csv(paths.configs_csv)
    desc = static.model_descriptor(static.static_table(data.catalogue()))
    try:
        out = mode_a_extras.run(t, desc, seed, paths, n_jobs=args.n_jobs)
    except ri.ReportInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for target, rec in out["fits"].items():
        print(f"final {target}: {rec['candidate']} (AUTO chose {rec['auto_counts']}), sha256 {rec['sha256'][:16]}")
    print(f"learning curve -> {paths.extras_dir / (out['learning_curve'] + '.parquet')}")
    print(f"v2-only vs pooled S-LOMO YOLO folds: {out['v2_vs_pooled']}")
    return 0


def cmd_report(args) -> int:
    import time
    import data
    import deploy
    import derive
    import eda_trace
    import pandas as pd
    import report
    import report_inputs as ri
    import runlog
    paths = ri.Paths.default()
    seed = args.seed if args.seed is not None else settings.seed()
    t0 = time.perf_counter()
    try:
        cat = data.catalogue()
        qc = pd.read_csv(paths.qc_csv) if paths.qc_csv.is_file() else None
        overhead = (eda_trace.perf_overhead(deploy.cell_table(cat, derive.derive_all(cat, cache=True), qc))
                    if qc is not None and len(cat) else None)
        files = report.build(paths, seed=seed, n_boot=args.n_boot, family=data.training_cells(cat).set_index("cell_id")["family"], overhead=overhead)
    except ri.ReportInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    written = report.write(paths, files)
    figs = sorted(paths.fig_dir.glob("report_*.png"))
    for p in written:
        print(f"{len(p.read_text().splitlines())} lines -> {p}")
    print(f"{len(figs)} figures, {sum(f.stat().st_size for f in figs) / 1e6:.2f} MB; {time.perf_counter() - t0:.0f} s")
    runlog.append({"command": "report", "seed": seed, "n_boot": args.n_boot, "files": [p.name for p in written],
                   "n_figures": len(figs)}, paths.runs_log)
    return 0


def add_extras_args(p) -> None:
    p.add_argument("--seed", type=int, help="default: the config.toml seed")
    p.add_argument("--n-jobs", type=int, default=2, help="outer folds in parallel (default 2)")


def add_report_args(p) -> None:
    p.add_argument("--seed", type=int, help="seed of the runs the tables use (default: the config.toml seed)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")


COMMANDS = {
    "mode-a-extras": ("final Mode A fits per target, energy learning curve and the v2-only vs pooled S-LOMO YOLO runs -> "
                      "derived/mode_a_extras/ and runs.jsonl; needs the S-LOMO runs from `mode-a`", cmd_mode_a_extras, add_extras_args),
    "report": ("write reports/results.md, model_card_mode_a.md, model_card_mode_l.md and reports/fig/report_*.png from the "
               "saved runs; fails naming the command that makes any missing input", cmd_report, add_report_args),
}
