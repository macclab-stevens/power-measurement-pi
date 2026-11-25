"""Lockbox 3 (R4): pre-registered configurations, feature rows, freeze refusals."""
import pytest

import lockbox as lb
import lockbox3


def test_configs_match_the_v4_manifest():
    import sys
    sys.path.insert(0, str(lockbox3.settings.HERE.parent / "host"))
    import matrix_v4
    planned = {(matrix_v4.model_key(c.model), c.imgsz, c.mhz) for c in matrix_v4.LOCKBOX3}
    assert planned == set(lockbox3.CONFIGS)


def test_feature_rows_are_hooks_off_at_their_clocks():
    t, shas = lockbox3.feature_rows()
    assert len(t) == 6 and set(shas) == {(m, s) for m, s, _ in lockbox3.CONFIGS}
    assert dict(zip(t["base_model"], t["cpu_mhz"])) == {m: mhz for m, _, mhz in lockbox3.CONFIGS}
    assert (t["hooks_on"] == 0).all() and t["config_key"].str.endswith("|hooks=off").all()
    attn = dict(zip(t["base_model"], t["frac_attn"]))
    assert attn["vitb32"] > 0.3 and all(v == 0 for m, v in attn.items() if m != "vitb32")   # only ViT has uncounted attention


def test_freeze_refuses_when_predictions_exist(tmp_path, monkeypatch):
    p = tmp_path / "lockbox3_predictions.csv"
    p.write_text("x\n")
    monkeypatch.setattr(lockbox3, "PRED", p)
    monkeypatch.setattr(lockbox3, "cells_exist", lambda: False)
    with pytest.raises(lb.LockboxError, match="exists"):
        lockbox3.freeze(0)


def test_freeze_refuses_when_cells_exist(monkeypatch):
    monkeypatch.setattr(lockbox3, "cells_exist", lambda: True)
    with pytest.raises(lb.LockboxError, match="already exist"):
        lockbox3.freeze(0)
