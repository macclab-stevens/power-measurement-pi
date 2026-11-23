"""Where the report reads its inputs from, and how it picks runs.

Every input is a saved output of an earlier CLI command. A missing one raises `ReportInputError`
whose message names the command that produces it. "Latest run" means the last matching line of
runs.jsonl whose parquet exists (a run whose file was deleted is skipped, not an error)."""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import runlog
import settings

SEED_SPREAD = (20261122, 20261123, 20261124)
SPLITS = ["S-LOMO", "S-LSO", "S-LVO", "S-LCO", "RANDOM"]
TARGETS = ["energy_mJ", "latency_ms", "mean_P_W"]
MODE_A_CMD = "uv run python cli.py mode-a"


class ReportInputError(RuntimeError):
    """A required input of the report is absent; the message names the command that makes it."""


@dataclass(frozen=True)
class Paths:
    runs_log: Path
    mode_a_dir: Path
    mode_l_dir: Path
    extras_dir: Path
    configs_csv: Path
    qc_csv: Path
    deploy_csv: Path
    out_dir: Path
    fig_dir: Path

    @classmethod
    def default(cls) -> "Paths":
        d, rep = settings.derived_dir(), settings.HERE / "reports"
        return cls(runlog.DEFAULT_LOG, d / "mode_a", d / "mode_l", d / "mode_a_extras",
                   rep / "mode_a_configs.csv", rep / "qc_cells.csv", rep / "deploy_validation.csv",
                   rep, rep / "fig")

    def under(self, root) -> "Paths":
        """The same layout with every output (reports and figures) under `root`."""
        return replace(self, out_dir=Path(root), fig_dir=Path(root) / "fig")


@dataclass
class RunRef:
    run_id: str
    seed: int | None
    df: pd.DataFrame
    rec: dict


def read_records(path) -> list:
    p = Path(path)
    if not p.is_file():
        raise ReportInputError(f"{p} is missing; no runs have been logged yet "
                               f"(run `{MODE_A_CMD}` first)")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def latest(recs: list, directory, *, suffix: str = "", **match) -> dict | None:
    """Last record with `rec[k] == v` for every item of `match` whose `<run_id><suffix>.parquet` exists
    in `directory`."""
    best = None
    for r in recs:
        if all(r.get(k) == v for k, v in match.items()) and "run_id" in r \
                and (Path(directory) / f"{r['run_id']}{suffix}.parquet").is_file():
            best = r
    return best


def ref_of(rec: dict, directory, suffix: str = "") -> RunRef:
    return RunRef(rec["run_id"], rec.get("seed"),
                  pd.read_parquet(Path(directory) / f"{rec['run_id']}{suffix}.parquet"), rec)


def load_mode_a(paths: Paths, recs: list, seed: int) -> dict:
    """{(split, target): RunRef} of the seed-`seed` runs of every split and target."""
    out = {}
    for split in SPLITS:
        for target in TARGETS:
            rec = latest(recs, paths.mode_a_dir, command="mode-a", split=split, target=target, seed=seed)
            if rec is None:
                raise ReportInputError(
                    f"no Mode A run with seed {seed} for split {split}, target {target}; run "
                    f"`{MODE_A_CMD} --split {split} --target {target} --seed {seed}`")
            out[(split, target)] = ref_of(rec, paths.mode_a_dir)
    return out


def load_seed_runs(paths: Paths, recs: list, master: int) -> dict:
    """{seed: RunRef} of the S-LOMO energy runs of the protocol seeds that exist; the master seed must."""
    out = {}
    for s in dict.fromkeys([master, *SEED_SPREAD]):
        rec = latest(recs, paths.mode_a_dir, command="mode-a", split="S-LOMO", target="energy_mJ", seed=s)
        if rec is not None:
            out[s] = ref_of(rec, paths.mode_a_dir)
    return out


def load_mode_l(paths: Paths, recs: list, seed: int) -> dict:
    rec = latest(recs, paths.mode_l_dir, command="mode-l", part="lomo", seed=seed)
    if rec is None:
        raise ReportInputError(f"no Mode L run with seed {seed}; run `uv run python cli.py mode-l --seed {seed}`")
    rid = rec["run_id"]
    out = {"run_id": rid, "rec": rec, "lomo": ref_of(rec, paths.mode_l_dir).df}
    for part, suffix in (("pmic_v1", "_pmic_v1"), ("probe", "_probe")):
        f = paths.mode_l_dir / f"{rid}{suffix}.parquet"
        if not f.is_file():
            why = ("; probe mode is skipped when no Mode A S-LOMO predictions exist, so run "
                   f"`{MODE_A_CMD} --split S-LOMO` and then " if part == "probe" else "; run ")
            raise ReportInputError(f"Mode L run {rid} has no {part} output ({f.name}){why}"
                                   f"`uv run python cli.py mode-l --seed {seed}`")
        out[part] = pd.read_parquet(f)
        out[f"{part}_rec"] = latest(recs, paths.mode_l_dir, suffix=suffix, command="mode-l", part=part, run_id=rid) or {}
    return out


def load_extras(paths: Paths, recs: list, seed: int) -> dict:
    """Outputs of `mode-a-extras`: final fits per target, the learning curve and the pooled vs v2-only runs."""
    cmd = "uv run python cli.py mode-a-extras"
    fits = {}
    for target in TARGETS:
        hits = [r for r in recs if r.get("command") == "mode-a-extras" and r.get("part") == "fit_final"
                and r.get("target") == target and r.get("seed") == seed]
        if not hits:
            raise ReportInputError(f"no final-model fit for target {target} with seed {seed}; run `{cmd} --seed {seed}`")
        fits[target] = hits[-1]
    rec = latest(recs, paths.extras_dir, command="mode-a-extras", part="learning_curve", seed=seed)
    if rec is None:
        raise ReportInputError(f"no learning-curve output with seed {seed}; run `{cmd} --seed {seed}`")
    out = {"fits": fits, "learning_curve": ref_of(rec, paths.extras_dir)}
    for training in ("pooled", "v2only"):
        r = latest(recs, paths.extras_dir, command="mode-a-extras", part="v2_vs_pooled", training=training, seed=seed)
        if r is None:
            raise ReportInputError(f"no {training} S-LOMO YOLO run with seed {seed}; run `{cmd} --seed {seed}`")
        out[training] = ref_of(r, paths.extras_dir)
    return out


def load_table(path, what: str, command: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise ReportInputError(f"{what} {p} is missing; run `{command}`")
    return pd.read_csv(p)


def latest_record(recs: list, **match) -> dict | None:
    """Last record matching `match` (no parquet needed)."""
    hits = [r for r in recs if all(r.get(k) == v for k, v in match.items())]
    return hits[-1] if hits else None
