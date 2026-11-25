"""`cli.py mode-a --label board`: the board-level run is recorded as such, on its own table, and the readers of
run records keep selecting Monsoon runs."""
import json

import pandas as pd
import pytest

import cli
import deploy
import mode_a
import report_inputs as ri
import runlog
import settings
import splits


class _Fold:
    def __init__(self, name):
        self.fold = name


def _table():
    return pd.DataFrame({"config_key": ["a", "b", "c"], "energy_mJ": [100.0, 60.0, 40.0], "latency_ms": [50.0, 30.0, 20.0],
                         "mean_P_W": [2.0, 2.0, 2.0], "energy_board_mJ": [93.0, 56.0, float("nan")],
                         "mean_P_board_W": [1.86, 1.86, 1.86]})


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    t = _table()
    t.to_csv(tmp_path / "mode_a_configs.csv", index=False)
    seen = []

    def run_split(table, folds, target, seed, desc, n_jobs=1):
        seen.append(table.copy())
        return pd.DataFrame({"candidate": ["AUTO"] * 2, "y_true": [1.0, 2.0], "y_pred": [1.1, 1.9],
                             "lo90": [0.5, 1.0], "hi90": [2.0, 3.0]})

    monkeypatch.setattr(cli, "_mode_a_inputs", lambda: (pd.read_csv(tmp_path / "mode_a_configs.csv"), None))
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    monkeypatch.setattr(mode_a, "run_split", run_split)
    monkeypatch.setattr(splits, "make_splits", lambda t, seed: {"S-LOMO": [_Fold("x")]})
    monkeypatch.setattr(splits, "write_splits", lambda sp, d: {f"{n}.csv": "split-sha" for n in sp})
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(runlog, "DEFAULT_LOG", log)
    return log, seen


def _run(*extra):
    args = cli.build_parser().parse_args(["mode-a", "--split", "S-LOMO", "--target", "energy_mJ", *extra])
    assert cli.cmd_mode_a(args) == 0


def test_label_default_is_monsoon(cli_env):
    log, seen = cli_env
    assert cli.build_parser().parse_args(["mode-a"]).label == "monsoon"
    _run()
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["label"] == "monsoon" and len(seen[0]) == 3 and seen[0]["energy_mJ"].tolist() == [100.0, 60.0, 40.0]


def test_board_run_uses_board_targets_and_its_own_table_sha(cli_env):
    log, seen = cli_env
    _run()
    _run("--label", "board")
    mon, brd = (json.loads(x) for x in log.read_text().splitlines())
    assert brd["label"] == "board" and brd["command"] == "mode-a"
    assert brd["table_sha256"] != mon["table_sha256"] and brd["run_id"] != mon["run_id"]
    b = seen[1]
    assert len(b) == 2 and b["energy_mJ"].tolist() == [93.0, 56.0] and b["mean_P_W"].tolist() == [1.86, 1.86]
    assert b["latency_ms"].tolist() == [50.0, 30.0]
    assert brd["table_sha256"] != runlog.file_hash(cli.REPORTS / "mode_a_configs.csv")
    assert mon["table_sha256"] == runlog.file_hash(cli.REPORTS / "mode_a_configs.csv")
    assert (settings.derived_dir() / "mode_a" / f"{brd['run_id']}.parquet").is_file()


def test_board_run_needs_the_board_columns(cli_env, tmp_path, monkeypatch, capsys):
    log, seen = cli_env
    old = _table().drop(columns=["energy_board_mJ", "mean_P_board_W"])
    monkeypatch.setattr(cli, "_mode_a_inputs", lambda: (old, None))
    args = cli.build_parser().parse_args(["mode-a", "--label", "board"])
    assert cli.cmd_mode_a(args) == 1 and not seen and not log.exists()
    assert "re-run" in capsys.readouterr().err


def _recs(tmp_path):
    out = []
    for rid, label in (("m1", None), ("m2", "monsoon"), ("b1", "board")):
        (tmp_path / f"{rid}.parquet").write_bytes(b"x")
        rec = {"run_id": rid, "command": "mode-a", "split": "S-LOMO", "target": "energy_mJ", "seed": 7}
        out.append({**rec, "label": label} if label else rec)
    return out


def test_readers_keep_selecting_monsoon_runs(tmp_path):
    recs = _recs(tmp_path)            # the board run is the newest record
    assert ri.latest(recs, tmp_path, command="mode-a", target="energy_mJ")["run_id"] == "m2"
    assert ri.latest(recs, tmp_path, label="board", command="mode-a", target="energy_mJ")["run_id"] == "b1"
    assert ri.latest_record(recs, command="mode-a")["run_id"] == "m2"
    assert ri.latest_record(recs, label="board", command="mode-a")["run_id"] == "b1"
    log = tmp_path / "runs.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert deploy._latest_parquet("S-LOMO", "energy_mJ", log, tmp_path, 7) == tmp_path / "m2.parquet"
    assert deploy._run_ids_with_seed("energy_mJ", 7, log) == {"m1", "m2"}
