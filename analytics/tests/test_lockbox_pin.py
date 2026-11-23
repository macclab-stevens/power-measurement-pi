"""Scoring code pinned at lockbox-predict, the measured model checked against the predicted one, and the idle table's hash."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import lockbox as lb
import lockbox_checks as lc
import lockbox_score as ls
import report_inputs as ri
import runlog
import settings
from synth_lockbox import MASTER
from test_lockbox_score import _rewrite, env, score  # noqa: F401  (env is a fixture)


# ------------------------------------------------------------------ runlog.code_files / code_sha256

def test_code_files_are_the_tracked_top_level_package_modules_with_their_sha256():
    files = runlog.code_files()
    assert "analytics/runlog.py" in files and "analytics/lockbox_score.py" in files and "analytics/cli.py" in files
    assert not any(k.startswith("analytics/tests/") or k.count("/") != 1 for k in files) and list(files) == sorted(files)
    assert files["analytics/runlog.py"] == runlog.file_hash(settings.HERE / "runlog.py")
    import hashlib
    want = hashlib.sha256("\n".join(f"{k}  {v}" for k, v in sorted(files.items())).encode()).hexdigest()
    assert runlog.code_sha256() == want == runlog.code_sha256(files)


def test_code_sha256_changes_with_one_byte_of_one_file():
    files = runlog.code_files()
    other = {**files, "analytics/cli.py": "0" * 64}
    assert runlog.code_sha256(other) != runlog.code_sha256(files)


def test_every_runlog_line_records_the_code_sha256(tmp_path):
    rec = runlog.append({"command": "x"}, tmp_path / "r.jsonl")
    assert rec["code_sha256"] == runlog.code_sha256() and json.loads((tmp_path / "r.jsonl").read_text())["code_sha256"] == rec["code_sha256"]


# ------------------------------------------------------------------ manifest pin

def test_manifest_pins_every_module_the_config_the_commit_and_the_idle_table(env):
    paths, _, _ = env
    man = json.loads((paths.out_dir / lb.MANIFEST).read_text())
    code = man["code"]
    assert code["files"] == runlog.code_files() and code["code_sha256"] == runlog.code_sha256()
    assert code["config_toml_sha256"] == runlog.file_hash(settings.HERE / "config.toml")
    assert len(code["commit"]) == 40 or code["commit"] == "unknown"
    assert man["idle_table_sha256"] and len(man["idle_table_sha256"]) == 64
    rec = ri.latest_record(ri.read_records(paths.runs_log), command="lockbox-predict")
    assert rec["code_sha256"] == code["code_sha256"] and rec["idle_table_sha256"] == man["idle_table_sha256"]


def test_idle_hash_ignores_row_order_and_changes_with_a_value():
    a = pd.DataFrame([("nov2026", "2400", "full", 5.0, 3), ("nov2026", "1500", "full", 4.9, 2)], columns=["rig", "cpu_key", "fan_mode", "P_idle_W", "n"])
    assert lc.idle_sha256(a) == lc.idle_sha256(a.iloc[::-1]) != lc.idle_sha256(a.assign(P_idle_W=[5.0, 4.8]))


# ------------------------------------------------------------------ code check at scoring

def tamper(monkeypatch, **edit):
    files = dict(runlog.code_files())
    for k, v in edit.items():
        if v is None:
            files.pop(k)
        else:
            files[k] = v
    monkeypatch.setattr(runlog, "code_files", lambda: files)


def test_unchanged_code_scores_and_reports_no_changed_files(env):
    out = score(env)
    text = out["report"].read_text()
    assert "EXPLORATORY" not in text and out["rec"]["exploratory"] is False and out["rec"]["changed_files"] == []
    assert "scoring code identical to `lockbox-predict`" in text


@pytest.mark.parametrize("edit", [{"analytics/lockbox_score.py": "1" * 64}, {"analytics/cli.py": None}, {"analytics/new_module.py": "2" * 64}])
def test_a_changed_added_or_removed_module_refuses_scoring(env, monkeypatch, edit):
    tamper(monkeypatch, **edit)
    with pytest.raises(lb.LockboxError, match="--allow-code-change") as exc:
        score(env)
    assert next(iter(edit)) in str(exc.value)
    assert not (env[0].out_dir / "lockbox.md").exists()


def test_a_changed_config_toml_refuses_scoring(env, monkeypatch):
    real = runlog.file_hash
    monkeypatch.setattr(runlog, "file_hash", lambda p: "9" * 64 if str(p).endswith("config.toml") else real(p))
    with pytest.raises(lb.LockboxError, match="config.toml"):
        score(env)


def test_allow_code_change_scores_but_labels_lockbox_md_and_the_runlog_exploratory(env, monkeypatch):
    tamper(monkeypatch, **{"analytics/lockbox_score.py": "1" * 64})
    paths, kw, _ = env
    out = ls.run_score(paths, MASTER, **kw, allow_code_change=True)
    text = out["report"].read_text()
    assert text.startswith("# Lockbox results") and "EXPLORATORY" in text.split("## Provenance")[0]
    assert "analytics/lockbox_score.py" in text.split("## Provenance")[0]
    assert out["rec"]["exploratory"] is True and out["rec"]["changed_files"] == ["analytics/lockbox_score.py"]


def test_a_manifest_without_a_code_pin_refuses_scoring(env):
    paths = env[0]
    man = paths.out_dir / lb.MANIFEST
    m = json.loads(man.read_text())
    m.pop("code")
    man.write_text(json.dumps(m))
    with pytest.raises(lb.LockboxError, match="no code pin"):
        score(env)


def test_commit_relation_is_reported_and_a_non_descendant_commit_refuses(env, monkeypatch):
    out = score(env)
    assert "recorded commit" in out["report"].read_text()
    monkeypatch.setattr(lc, "commit_relation", lambda old: "not an ancestor of HEAD")
    with pytest.raises(lb.LockboxError, match="not an ancestor of HEAD"):
        score(env)
    monkeypatch.setattr(lc, "commit_relation", lambda old: "unknown")
    assert score(env)["rec"]["exploratory"] is False


# ------------------------------------------------------------------ idle table

def test_score_reports_whether_the_recomputed_idle_table_matches_the_recorded_hash(env):
    import deploy
    paths, kw, _ = env
    man = paths.out_dir / lb.MANIFEST
    cat_t = kw["cat"][kw["cat"]["source"] != "lockbox"]
    # the fixture predicted with a hand-made idle table (5.0 W), the training cells give 5.1 W: the hashes differ
    out = score(env)
    assert "idle table matches the one used at `lockbox-predict`: NO" in out["report"].read_text() and out["rec"]["idle_table_matches"] is False
    m = json.loads(man.read_text())
    m["idle_table_sha256"] = lc.idle_sha256(deploy.idle_table(cat_t, kw["der"], kw["qc_train"]))
    man.write_text(json.dumps(m))
    out2 = score(env)
    assert "idle table matches the one used at `lockbox-predict`: yes" in out2["report"].read_text() and out2["rec"]["idle_table_matches"] is True


# ------------------------------------------------------------------ measured model against the predicted one

def break_layer(env, cell, edit):
    paths, kw, _ = env
    p = kw["cat"].set_index("cell_id").loc[cell, "path"]
    lt = pd.read_csv(f"{p}/layer_table.csv")
    new = edit(lt)
    lt = lt if new is None else new
    lt.to_csv(f"{p}/layer_table.csv", index=False)


@pytest.mark.parametrize("col, factor, feature", [("macs", 1.0000001, "macs"), ("params", 1.0000001, "params")])
def test_macs_and_params_must_match_exactly(env, col, factor, feature):
    break_layer(env, "mbv3_r1", lambda lt: lt.__setitem__(col, (lt[col].astype(float) * factor).where(lt["class"] == "Conv2d", lt[col])))
    with pytest.raises(lb.LockboxError, match=f"mbv3_r1.*{feature}") as exc:
        score(env)
    assert "mbv3:224" in str(exc.value) or "mbv3" in str(exc.value)


def test_another_feature_must_match_to_a_relative_1e_9(env):
    break_layer(env, "resnet18_r0", lambda lt: lt.__setitem__("bytes_moved", lt["bytes_moved"] * 1.001))
    with pytest.raises(lb.LockboxError, match="resnet18_r0.*act_bytes"):
        score(env)


def test_a_different_architecture_in_the_measured_cell_is_refused(env):
    break_layer(env, "yolov8n_r2", lambda lt: lt.iloc[:-1])
    with pytest.raises(lb.LockboxError, match="yolov8n_r2"):
        score(env)


def test_a_primary_cell_without_a_layer_table_cannot_be_verified(env):
    paths, kw, _ = env
    import os
    os.remove(f"{kw['cat'].set_index('cell_id').loc['mbv3_r0', 'path']}/layer_table.csv")
    with pytest.raises(lb.LockboxError, match="mbv3_r0.*layer_table.csv"):
        score(env)


def test_a_static_csv_that_is_not_the_one_the_predictions_recorded_is_refused(env):
    paths, kw, e = env
    d = paths.out_dir.parent / "static2"
    shutil.copytree(e.static_dir, d)
    (d / "mbv3_224.csv").write_text((d / "mbv3_224.csv").read_text() + "\n")
    with pytest.raises(lb.LockboxError, match="mbv3_224.csv.*sha256"):
        ls.run_score(paths, MASTER, **{**kw, "static_dir": d})


def test_non_primary_cells_are_not_checked_and_matching_cells_pass(env):
    break_layer(env, "yolo12n_r2", lambda lt: lt.__setitem__("macs", lt["macs"].astype(float) * 2))     # r2 is not QC-primary
    out = score(env)
    assert "layer tables of the measured cells equal the static features" in out["report"].read_text()
    assert out["rec"]["static_checked_cells"] == 11          # 12 lockbox cells, one of them (yolo12n_r2) not QC-primary


def test_compare_features_tolerances():
    a = {"macs": 100.0, "params": 7.0, "depth": 3, "frac_macs_conv": 0.5, "x": float("nan")}
    assert lc.feature_mismatches(a, dict(a)) == []
    assert lc.feature_mismatches(a, {**a, "frac_macs_conv": 0.5 * (1 + 1e-10)}) == []
    assert [m[0] for m in lc.feature_mismatches(a, {**a, "frac_macs_conv": 0.5 * (1 + 1e-8)})] == ["frac_macs_conv"]
    assert [m[0] for m in lc.feature_mismatches(a, {**a, "macs": 100.0 * (1 + 1e-12)})] == ["macs"]
    assert [m[0] for m in lc.feature_mismatches(a, {**a, "params": 8.0})] == ["params"]
    assert [m[0] for m in lc.feature_mismatches(a, {**a, "x": 1.0})] == ["x"]


# ------------------------------------------------------------------ minor items

def test_lockbox_md_discloses_the_novelty_scales(env):
    text = score(env)["report"].read_text()
    assert "standardised over the 8 training models of each fold" in text and "standardised over all 9 training models" in text


def test_lockbox_windows_take_cpu_key_fan_and_governor_from_the_catalogue_row(tmp_path, monkeypatch):
    import data
    import windows
    cat = pd.DataFrame({"cell_id": ["a", "b"], "path": [str(tmp_path / "a"), str(tmp_path / "b")], "has_counters": True,
                        "governor": ["pinned", "ondemand"], "cpu_mhz": [1500, 2400], "fan_mode": ["auto", "full"]})
    qc = pd.DataFrame({"cell_id": ["a", "b"], "primary": True})
    meta = pd.DataFrame({"cell_id": ["x"], "w_idx": [0], "cpu_key": "2400", "fan_mode": "meta", "governor": "meta"})
    monkeypatch.setattr(data, "read_meta", lambda p: {})
    monkeypatch.setattr(windows, "cell_windows", lambda cell, m: meta.assign(cell_id=cell.name))
    w = ls.lockbox_windows(cat, qc).set_index("cell_id")
    assert w.loc["a", ["cpu_key", "fan_mode", "governor"]].tolist() == ["1500", "auto", "pinned"]
    assert w.loc["b", ["cpu_key", "fan_mode", "governor"]].tolist() == ["ondemand", "full", "ondemand"]
    # the cached builders use the same function on the same row
    row = cat.iloc[0]
    assert windows.assign_settings(meta, row)[["cpu_key", "fan_mode", "governor"]].iloc[0].tolist() == ["1500", "auto", "pinned"]


def test_cli_predict_runs_its_guard_first_and_derives_over_the_full_catalogue(tmp_path, monkeypatch, capsys):
    import cli
    import data
    import derive
    import deploy
    import static
    from synth_report import make_inputs
    paths = make_inputs(tmp_path)
    paths.out_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(ri.Paths, "default", classmethod(lambda cls: paths))
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "data"))
    calls = []
    cat = pd.DataFrame({"cell_id": ["a", "z"], "source": ["v2", "lockbox"]})
    monkeypatch.setattr(data, "catalogue", lambda: cat)
    monkeypatch.setattr(derive, "derive_all", lambda c, cache=True: calls.append((list(c["cell_id"]), cache)) or pd.DataFrame())
    monkeypatch.setattr(static, "static_table", lambda c: c)
    monkeypatch.setattr(static, "model_descriptor", lambda c: c)
    monkeypatch.setattr(deploy, "idle_table", lambda c, d, q: (calls.append(("idle", list(c["cell_id"]))), pd.DataFrame())[1])
    monkeypatch.setattr(lb, "run_predict", lambda *a, **k: (_ for _ in ()).throw(lb.LockboxError("stop here")))
    (paths.out_dir / lb.PREDICTIONS_CSV).write_text("x\n")
    assert cli.main(["lockbox-predict", "--static-dir", "d"]) == 1
    assert "already exists" in capsys.readouterr().err and calls == []          # refused before any derive
    assert cli.main(["lockbox-predict", "--static-dir", "d", "--force"]) == 1
    assert "stop here" in capsys.readouterr().err
    assert calls == [(["a", "z"], True), ("idle", ["a"])]        # derive over the full catalogue, idle over the training cells
    (tmp_path / "data" / "lockbox" / "cells" / "x_r00").mkdir(parents=True)
    calls.clear()
    assert cli.main(["lockbox-predict", "--static-dir", "d", "--force"]) == 1
    assert "lockbox cells already exist" in capsys.readouterr().err and calls == []


def test_lockbox_score_cli_arguments():
    import cli
    a = cli.build_parser().parse_args(["lockbox-score"])
    assert a.static_dir.endswith("analytics/lockbox_static") and a.allow_code_change is False
    b = cli.build_parser().parse_args(["lockbox-score", "--static-dir", "d", "--allow-code-change"])
    assert b.static_dir == "d" and b.allow_code_change is True


# ------------------------------------------------------------------ follow-up: dependency pin, nothing checked, caption

def test_manifest_pins_uv_lock_and_pyproject(env):
    paths, _, _ = env
    code = json.loads((paths.out_dir / lb.MANIFEST).read_text())["code"]
    assert code["dependency_files"] == {"analytics/uv.lock": runlog.file_hash(settings.HERE / "uv.lock"),
                                        "analytics/pyproject.toml": runlog.file_hash(settings.HERE / "pyproject.toml")}


@pytest.mark.parametrize("name", ["analytics/uv.lock", "analytics/pyproject.toml"])
def test_a_changed_dependency_file_refuses_scoring_unless_allowed(env, monkeypatch, name):
    real = lc.dependency_files()
    monkeypatch.setattr(lc, "dependency_files", lambda: {**real, name: "7" * 64})
    with pytest.raises(lb.LockboxError, match=name):
        score(env)
    paths, kw, _ = env
    out = ls.run_score(paths, MASTER, **kw, allow_code_change=True)
    assert out["rec"]["exploratory"] is True and out["rec"]["changed_files"] == [name]
    assert name in out["report"].read_text().split("## Provenance")[0]


def test_a_manifest_pin_without_dependency_files_refuses(env):
    paths = env[0]
    man = paths.out_dir / lb.MANIFEST
    m = json.loads(man.read_text())
    m["code"].pop("dependency_files")
    man.write_text(json.dumps(m))
    with pytest.raises(lb.LockboxError, match="uv.lock"):
        score(env)


def test_static_check_refuses_when_no_cell_was_checked(env):
    paths, kw, _ = env
    cat = kw["cat"].copy()
    cat.loc[cat["source"] == "lockbox", "imgsz"] = 999            # a naming mismatch: nothing matches a prediction
    pred = pd.read_csv(paths.out_dir / lb.PREDICTIONS_CSV)
    cat_l = cat[cat["source"] == "lockbox"]
    with pytest.raises(lb.LockboxError, match="no lockbox cell was checked"):
        lc.check_static(cat_l, kw["qc_l"], pred, kw["static_dir"])
    with pytest.raises(lb.LockboxError, match="no lockbox cell was checked"):
        lc.check_static(kw["cat"][kw["cat"]["source"] == "lockbox"], kw["qc_l"].assign(primary=False), pred, kw["static_dir"])


def test_static_csv_path_is_the_one_lockbox_function():
    assert lb.static_csv_path("d", "yolo12n", 640) == lc.static_csv_path("d", "yolo12n", 640) == Path("d") / "yolo12n_640.csv"
