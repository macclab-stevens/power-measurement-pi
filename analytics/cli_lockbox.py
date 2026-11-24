"""CLI commands that freeze the models the lockbox is scored with and run the lockbox pipeline:
`final-mode-l`, `final-mode-a-secondary`, `lockbox-predict`, `lockbox-score`, and the post hoc (R2)
`lockbox-calculator-gap`. Registered in cli.py's COMMANDS.
None of them fits anything on lockbox data: the fit commands drop `source == lockbox` rows first."""
import sys

import settings


def _fail(exc) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


def _seed(args) -> int:
    return args.seed if args.seed is not None else settings.seed()


def guard(what: str, force: bool = False) -> int | None:
    """1 (after printing why) when `what`, a command that fits or freezes a model, must not run: lockbox cells exist
    (never overridable), or `lockbox-predict` has written its predictions or manifest (overridable by `force`, which
    is logged as a `freeze-override` line)."""
    import lockbox
    import report_inputs as ri
    import runlog
    paths = ri.Paths.default()
    try:
        lockbox.refuse_if_lockbox_cells(what)
        over = lockbox.refuse_if_frozen(what, paths.out_dir, force)
    except lockbox.LockboxError as exc:
        return _fail(exc)
    if over:
        runlog.append({"command": "freeze-override", "what": what, "markers": over}, paths.runs_log)
    return None


def _training_catalogue():
    import data
    cat = data.catalogue()
    return cat[cat["source"] != "lockbox"].reset_index(drop=True)


def cmd_final_mode_l(args) -> int:
    if (bad := guard("final-mode-l", args.force)) is not None:
        return bad
    import pandas as pd
    import report_inputs as ri
    import mode_l
    import mode_l_models
    import runlog
    import windows
    from joblib import Parallel, delayed
    paths = ri.Paths.default()
    if not paths.qc_csv.is_file():
        return _fail(f"{paths.qc_csv} is missing; run `uv run python cli.py qc` first")
    seed = _seed(args)
    names = args.rung or list(mode_l_models.default_rungs())
    unknown = [n for n in names if n not in mode_l_models.default_rungs()]
    if unknown:
        return _fail(f"unknown rung {unknown}; expected {list(mode_l_models.default_rungs())}")
    cat, qc = _training_catalogue(), pd.read_csv(paths.qc_csv)
    w = windows.build_windows(cat)
    res = Parallel(n_jobs=args.n_jobs)(delayed(mode_l.fit_final)(w, cat, qc, n, seed) for n in names)
    for name, (_, info) in zip(names, res):
        runlog.append({"command": "final-mode-l", "part": "fit_final", "seed": seed, "rung": name, **info}, paths.runs_log)
        print(f"{name}: params {info['params']}, {info['n_cells']} cells, {info['n_windows']} windows, sha256 {info['sha256'][:16]} "
              f"-> {info['path']}")
    return 0


def cmd_final_mode_a_secondary(args) -> int:
    if (bad := guard("final-mode-a-secondary", args.force)) is not None:
        return bad
    import pandas as pd
    import report_inputs as ri
    import static
    import mode_a_extras
    paths = ri.Paths.default()
    if not paths.configs_csv.is_file():
        return _fail(f"{paths.configs_csv} is missing; run `uv run python cli.py tables` first")
    seed = _seed(args)
    desc = static.model_descriptor(static.static_table(_training_catalogue()))
    try:
        out = mode_a_extras.run_secondary(pd.read_csv(paths.configs_csv), desc, seed, paths)
    except ri.ReportInputError as exc:
        return _fail(exc)
    for (target, cand), rec in out.items():
        print(f"{cand} / {target}: params {rec['params']}, sha256 {rec['sha256'][:16]} -> {rec['path']}")
    if not out:
        print("nothing to fit: every secondary candidate is already the AUTO-final model")
    return 0


def cmd_lockbox_predict(args) -> int:
    import data
    import derive
    import pandas as pd
    import deploy
    import lockbox
    import report_inputs as ri
    import static
    paths = ri.Paths.default()
    try:
        lockbox.predict_guard(paths, args.force)               # first: nothing is derived or cached when this run is refused
    except lockbox.LockboxError as exc:
        return _fail(exc)
    for p in (paths.configs_csv, paths.qc_csv):
        if not p.is_file():
            return _fail(f"{p} is missing; run `uv run python cli.py tables` / `qc` first")
    full = data.catalogue()                                    # derive_all caches only for the full catalogue
    cat = full[full["source"] != "lockbox"].reset_index(drop=True)
    try:
        out = lockbox.run_predict(
            paths, args.config or lockbox.DEFAULT_CONFIGS, args.static_dir, _seed(args), t=pd.read_csv(paths.configs_csv),
            desc_train=static.model_descriptor(static.static_table(cat)),
            idle=deploy.idle_table(cat, derive.derive_all(full, cache=True), pd.read_csv(paths.qc_csv)), force=args.force)
    except (lockbox.LockboxError, ri.ReportInputError) as exc:
        return _fail(exc)
    pred = out["pred"]
    prim = pred[(pred["role"] == "primary") & (pred["target"] == "energy_mJ")]
    print(f"{len(pred)} predictions ({pred['config_key'].nunique()} configurations) -> {out['path']}")
    print(prim[["model", "imgsz", "candidate", "pred", "lo90", "hi90", "novelty", "threshold", "flagged"]].to_string(index=False))
    print(f"sha256 {out['rec']['predictions_sha256']}; commit this file before any lockbox cell is measured")
    return 0


def cmd_lockbox_score(args) -> int:
    import derive
    import pandas as pd
    import data
    import lockbox
    import lockbox_score as ls
    import qc
    import report_inputs as ri
    paths = ri.Paths.default()
    if not lockbox.lockbox_cells_exist():
        return _fail(f"no lockbox cells under {settings.data_root() / 'lockbox' / 'cells'}; lockbox-score runs after they are measured")
    if not paths.qc_csv.is_file():
        return _fail(f"{paths.qc_csv} is missing; run `uv run python cli.py qc` first")
    try:
        cat = data.catalogue()
        cat_l = ls.require_lockbox(cat)
        der = derive.derive_all(cat, cache=True)
        qc_l = qc.run_qc(cat_l, der)
        out = ls.run_score(paths, _seed(args), cat=cat, der=der, qc_train=pd.read_csv(paths.qc_csv), qc_l=qc_l,
                           w=ls.lockbox_windows(cat_l, qc_l), starts=ls.cell_starts(cat_l), static_dir=args.static_dir,
                           allow_code_change=args.allow_code_change)
    except (lockbox.LockboxError, ri.ReportInputError, ValueError) as exc:
        return _fail(exc)
    print(out["summary_a"].to_string(index=False))
    if len(out["mode_l_summary"]):
        print(out["mode_l_summary"].to_string(index=False))
    print(f"report -> {out['report']}; run {out['run_id']}")
    return 0


def cmd_lockbox_calculator_gap(args) -> int:
    """Post hoc (R2): the frozen predictions through the closed-loop gap scenario, no measured duration. Fits nothing and
    does not touch the frozen lockbox outputs."""
    import derive
    import pandas as pd
    import data
    import lockbox
    import lockbox_score as ls
    import qc
    import report_inputs as ri
    paths = ri.Paths.default()
    if not lockbox.lockbox_cells_exist():
        return _fail(f"no lockbox cells under {settings.data_root() / 'lockbox' / 'cells'}")
    if not paths.qc_csv.is_file():
        return _fail(f"{paths.qc_csv} is missing; run `uv run python cli.py qc` first")
    try:
        cat = data.catalogue()
        der = derive.derive_all(cat, cache=True)
        out = ls.run_posthoc_gap(paths, cat=cat, der=der, qc_train=pd.read_csv(paths.qc_csv),
                                 qc_l=qc.run_qc(ls.require_lockbox(cat), der))
    except (lockbox.LockboxError, ri.ReportInputError, ValueError) as exc:
        return _fail(exc)
    print(out["summary"].to_string(index=False))
    print(f"post hoc (R2): {len(out['df'])} cells -> {out['csv']}, {out['md']}; run {out['rec']['run_id']}")
    return 0


def add_seed(p) -> None:
    p.add_argument("--seed", type=int, help="seed of the frozen fits (default: the config.toml seed)")


def add_force(p) -> None:
    p.add_argument("--force", action="store_true", help="run although the models are frozen (lockbox_predictions.csv or "
                   "lockbox_frozen_models.json exists); invalidates the lockbox, logged as freeze-override")


def add_seed_force(p) -> None:
    add_seed(p)
    add_force(p)


def add_final_l_args(p) -> None:
    add_seed_force(p)
    p.add_argument("--n-jobs", type=int, default=2, help="rungs fitted in parallel (default 2)")
    p.add_argument("--rung", action="append", help="fit only this rung (repeatable; default all eight)")


def add_predict_args(p) -> None:
    add_seed(p)
    p.add_argument("--config", action="append", help="<model>:<imgsz>, repeatable (default: the five protocol configurations)")
    p.add_argument("--static-dir", required=True, help="folder with <model>_<imgsz>.csv written by tools/static_features.py")
    p.add_argument("--force", action="store_true", help="overwrite an existing lockbox_predictions.csv")


def add_score_args(p) -> None:
    add_seed(p)
    p.add_argument("--static-dir", default=str(settings.HERE / "lockbox_static"),
                   help="folder with the <model>_<imgsz>.csv the predictions were made with (default: analytics/lockbox_static)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="score although tracked analytics/*.py or config.toml differ from the pin in lockbox_frozen_models.json; "
                        "lockbox.md and the runlog line are then labelled exploratory")


COMMANDS = {
    "final-mode-l": ("freeze the eight Mode L rungs (grid by leave-one-YOLO-model-out, refit on all QC-primary counter "
                     "cells) -> derived/models/mode_l_<rung>.pkl and runs.jsonl", cmd_final_mode_l, add_final_l_args),
    "final-mode-a-secondary": ("freeze B3 (all targets) and M3 (energy) on all configurations -> derived/models/ and "
                               "runs.jsonl", cmd_final_mode_a_secondary, add_seed_force),
    "lockbox-predict": ("write reports/lockbox_predictions.csv from the frozen Mode A models and tool static features, "
                        "before any lockbox cell exists", cmd_lockbox_predict, add_predict_args),
    "lockbox-score": ("score the frozen models on ~/pi-power-data/lockbox -> reports/lockbox.md and derived/lockbox/; "
                      "fits nothing", cmd_lockbox_score, add_score_args),
    "lockbox-calculator-gap": ("post hoc (R2): serve-phase calculator on the lockbox with the frozen predictions and the "
                               "closed-loop gap scenario (no measured duration) -> reports/r2/lockbox_calculator_gap.{csv,md}; "
                               "fits nothing", cmd_lockbox_calculator_gap, None),
}
