"""Once `lockbox-predict` has written its predictions or the frozen-models manifest, the commands that fit or freeze a
model refuse to run (unless --force, which is logged and invalidates the lockbox)."""
import json

import pytest

import cli
import lockbox as lb
import report_inputs as ri
from synth_report import make_inputs

COMMANDS = [["mode-a-extras"], ["final-mode-a-secondary"], ["final-mode-l"]]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    p = make_inputs(tmp_path)
    p.configs_csv.unlink()                      # a command that gets past the guard stops at this missing input, fitting nothing
    p.qc_csv.unlink()
    p.out_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(ri.Paths, "default", classmethod(lambda cls: p))
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "data"))
    return p


def run(argv):
    args = cli.build_parser().parse_args(argv)
    return args.fn(args)


@pytest.mark.parametrize("marker", [lb.PREDICTIONS_CSV, lb.MANIFEST])
@pytest.mark.parametrize("argv", COMMANDS)
def test_fit_commands_refuse_once_the_predictions_or_manifest_exist(paths, capsys, argv, marker):
    (paths.out_dir / marker).write_text("x\n")
    before = paths.runs_log.read_text()
    assert run(argv) == 1
    err = capsys.readouterr().err
    assert marker in err and "--force" in err and "invalidates the lockbox" in err
    assert paths.runs_log.read_text() == before


@pytest.mark.parametrize("argv", COMMANDS)
def test_without_the_markers_the_commands_get_past_the_guard(paths, capsys, argv):
    assert run(argv) == 1
    err = capsys.readouterr().err
    assert "invalidates the lockbox" not in err and "is missing; run" in err


@pytest.mark.parametrize("argv", COMMANDS)
def test_force_passes_the_guard_and_is_logged(paths, capsys, argv):
    (paths.out_dir / lb.MANIFEST).write_text("x\n")
    assert run([*argv, "--force"]) == 1
    assert "is missing; run" in capsys.readouterr().err
    last = ri.read_records(paths.runs_log)[-1]
    assert last["command"] == "freeze-override" and last["what"] == argv[0] and last["markers"] == [lb.MANIFEST]


@pytest.mark.parametrize("argv", COMMANDS)
def test_lockbox_cells_still_refuse_even_with_force(paths, tmp_path, capsys, argv):
    (tmp_path / "data" / "lockbox" / "cells" / "x_r00").mkdir(parents=True)
    (paths.out_dir / lb.MANIFEST).write_text("x\n")
    assert run([*argv, "--force"]) == 1
    assert "lockbox cells already exist" in capsys.readouterr().err


def test_refuse_if_frozen_names_every_marker_present_and_none_when_absent(tmp_path):
    lb.refuse_if_frozen("x", tmp_path)
    (tmp_path / lb.PREDICTIONS_CSV).write_text("x")
    (tmp_path / lb.MANIFEST).write_text("x")
    with pytest.raises(lb.LockboxError) as exc:
        lb.refuse_if_frozen("final-mode-l", tmp_path)
    assert lb.PREDICTIONS_CSV in str(exc.value) and lb.MANIFEST in str(exc.value) and "final-mode-l" in str(exc.value)
    assert lb.refuse_if_frozen("final-mode-l", tmp_path, force=True) == [lb.PREDICTIONS_CSV, lb.MANIFEST]
