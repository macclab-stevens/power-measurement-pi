"""Lockbox 2 (R3): feature rows, training isolation, freeze and score refusals."""
import pandas as pd
import pytest

import lockbox as lb
import lockbox2


def test_feature_rows_from_the_committed_static_csvs():
    t, shas = lockbox2.feature_rows()
    assert sorted(zip(t["base_model"], t["imgsz"])) == sorted(lockbox2.CONFIGS) and len(shas) == 5
    assert (t["hooks_on"] == 0).all() and t["config_key"].str.endswith("|hooks=off").all()
    sw = t.set_index("base_model").loc["swins"]
    assert sw["frac_attn"] > 0.3 and t.set_index("base_model").loc["resnet34", "frac_attn"] < 0.02


def test_check_training_refuses_holdout_cells():
    lockbox2.check_training(pd.DataFrame({"cell_ids": ["v3add_a;v2_b"]}), pd.DataFrame({"cell_id": ["v3add_a"]}))
    with pytest.raises(lb.LockboxError):
        lockbox2.check_training(pd.DataFrame({"cell_ids": ["v2_b;lockbox2_x_r00"]}), None)
    with pytest.raises(lb.LockboxError):
        lockbox2.check_training(pd.DataFrame({"cell_ids": ["v2_b"]}), pd.DataFrame({"cell_id": ["lockbox_y_r01"]}))


def test_freeze_refuses_once_lockbox2_cells_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(lockbox2.settings, "data_root", lambda: tmp_path)
    (tmp_path / "lockbox2" / "cells" / "x_r00").mkdir(parents=True)
    with pytest.raises(lb.LockboxError, match="already exist"):
        lockbox2.freeze(None, 1, t=None, desc_train=None, w_train=None, cat=None, qc=None, threshold=0.0)


def test_score_refuses_a_second_time(tmp_path):
    class P:
        out_dir = tmp_path
    (tmp_path / "r3").mkdir()
    (tmp_path / "r3" / "lockbox2.md").write_text("scored")
    with pytest.raises(lb.LockboxError, match="scored once"):
        lockbox2.score(P, 1, cat=None, der=None, qc_train=None)
