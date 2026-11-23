"""Deployment calculator: hand-computed predict cases, measured lookups and the validation frame."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import deploy

NAN = float("nan")


def _idle(rows=None):
    rows = rows or [("nov2026", "2400", "full", 5.0), ("nov2026", "ondemand", "full", 4.0),
                    ("nov2026", "1500", "full", 4.0), ("oct2026", "2400", "full", 5.2)]
    return pd.DataFrame(rows, columns=["rig", "cpu_key", "fan_mode", "P_idle_W"])


def _sc(**kw):
    return {"cpu_key": "2400", "fan_mode": "full", "duration_s": 100.0, **kw}


# ------------------------------------------------------------------ predict

def test_predict_saturates_at_15fps_with_290ms():
    p = deploy.predict(_sc(rate_fps=15), 2900.0, 290.0, _idle(), None, None)
    assert p["saturated"] and p["idle_s"] == 0.0 and p["busy_s"] == pytest.approx(100.0)
    assert p["n_images"] == pytest.approx(100 / 0.29)
    assert p["E_J"] == pytest.approx(1000.0) and p["mean_P_W"] == pytest.approx(10.0)
    assert p["E_J"] == p["E_J_no_soak"]
    mx = deploy.predict(_sc(rate_fps="max"), 2900.0, 290.0, _idle(), None, None)
    assert mx["saturated"] and mx["E_J"] == pytest.approx(1000.0)


def test_predict_one_fps_has_idle_time():
    p = deploy.predict(_sc(rate_fps=1), 2900.0, 290.0, _idle(), None, None)
    assert not p["saturated"] and p["n_images"] == pytest.approx(100.0)
    assert p["busy_s"] == pytest.approx(29.0) and p["idle_s"] == pytest.approx(71.0)
    assert p["E_J"] == pytest.approx(100 * 2.9 + 71 * 5.0)       # 645 J


def test_predict_idle_only_at_rate_zero():
    p = deploy.predict(_sc(rate_fps=0, duration_s=60), 2900.0, 290.0, _idle(), None, None)
    assert p["n_images"] == 0 and p["E_J"] == pytest.approx(300.0) and p["mean_P_W"] == pytest.approx(5.0)


def test_predict_ondemand_uses_ondemand_idle_row():
    p = deploy.predict(_sc(cpu_key="ondemand", rate_fps=1), 2900.0, 290.0, _idle(), None, None)
    assert p["E_J"] == pytest.approx(290 + 71 * 4.0)              # 574 J
    assert any("ondemand" in a for a in p["assumptions"])


def test_predict_rig_picks_its_idle_row():
    p = deploy.predict(_sc(rate_fps=0, rig="oct2026"), 1.0, 290.0, _idle(), None, None)
    assert p["E_J"] == pytest.approx(520.0)


def test_predict_soak_factor_ramps_with_duration():
    base = 645.0
    at = lambda dur, **kw: deploy.predict(_sc(rate_fps=1, duration_s=dur, **kw), 2900.0, 290.0, _idle(), None, 1.2)
    p = at(100.0)
    assert p["E_J_no_soak"] == pytest.approx(base)
    assert p["E_J"] == pytest.approx(base * (1 + 0.2 * 100 / 1800))
    # at 900 s: factor 1.1; at 1800 s and beyond: factor 1.2 (capped)
    p9 = at(900.0)
    assert p9["E_J"] == pytest.approx(p9["E_J_no_soak"] * 1.1)
    for d in (1800.0, 7200.0):
        pd_ = at(d)
        assert pd_["E_J"] == pytest.approx(pd_["E_J_no_soak"] * 1.2)
    assert any("soak" in a for a in p["assumptions"])


def test_predict_fan_offsets_adjust_active_and_idle():
    off = {"auto": {"active_W": 0.5, "idle_W": -0.6}, "assumption": "assumption: measured on yolo11n only"}
    p = deploy.predict(_sc(rate_fps=1, fan_mode="auto"), 2900.0, 290.0, _idle(), off, None)
    assert p["E_J"] == pytest.approx(290 + 0.5 * 29 + 71 * 4.4)
    assert any("yolo11n only" in a for a in p["assumptions"])
    with pytest.raises(ValueError):
        deploy.predict(_sc(rate_fps=1, fan_mode="off"), 2900.0, 290.0, _idle(), off, None)


def test_predict_n_images_clips_busy_time_never_negative_idle():
    p = deploy.predict(_sc(n_images=500), 2900.0, 290.0, _idle(), None, None)
    assert p["saturated"] and p["busy_s"] == 100.0 and p["idle_s"] == 0.0
    assert p["n_images"] == 500 and p["E_J"] == pytest.approx(1450.0)
    q = deploy.predict(_sc(n_images=100), 2900.0, 290.0, _idle(), None, None)
    assert not q["saturated"] and q["E_J"] == pytest.approx(290 + 71 * 5.0)


def test_predict_unknown_idle_row_raises():
    with pytest.raises(KeyError):
        deploy.predict(_sc(cpu_key="2000", rate_fps=1), 1.0, 290.0, _idle(), None, None)


def test_mode_a_key_for_cell():
    row = {"base_model": "yolo11s", "variant": "yolo11s", "precision": "fp32", "imgsz": 640, "batch": 1,
           "threads": 4, "cpu_mhz": 1500, "rig": "nov2026", "governor": "ondemand", "fan_mode": "auto",
           "serve_fps": 5.0, "soak_minutes": NAN}
    assert deploy.mode_a_key_for_cell(row) == "yolo11s|yolo11s|fp32|640|1|4|2400|nov2026"
    assert deploy.mode_a_key_for_cell({**row, "governor": "pinned", "cpu_mhz": 1500}).endswith("|1500|nov2026")


# ------------------------------------------------------------------ synthetic cells

def _crow(cell_id, **kw):
    r = {"cell_id": cell_id, "base_model": "yolo11n", "variant": "yolo11n", "precision": "fp32", "imgsz": 640,
         "batch": 1, "threads": 4, "cpu_mhz": 2400, "governor": "pinned", "fan_mode": "full",
         "serve_fps": NAN, "soak_minutes": NAN, "concurrent_mode": None, "rig": "nov2026",
         "input_source": "repeat", "mode_a_eligible": False, "family": "A"}
    r.update(kw)
    return r


def _drow(cid, e=3000.0, lat=300.0, P=10.0, base=5.0, n=29, batch=1, sreq=10, sdur=NAN, sE=NAN):
    return {"cell_id": cid, "mode": "forward", "batch": batch, "n_images": n, "energy_mJ_per_image": e,
            "latency_ms_per_image": lat, "mean_P_W": P, "baseline_W": base, "serve_n_requests": sreq,
            "serve_dur_ms": sdur, "serve_E_mJ": sE}


def _frames(spec):
    """spec: list of (cat kwargs dict with cell_id, der kwargs dict, primary)."""
    cat = pd.DataFrame([_crow(**c) for c, _, _ in spec])
    der = pd.DataFrame([_drow(c["cell_id"], **d) for c, d, _ in spec])
    qc = pd.DataFrame([{"cell_id": c["cell_id"], "include": True, "primary": p} for c, _, p in spec])
    return cat, der, qc


def test_idle_table_keys_primary_only_and_both_rigs():
    spec = [(dict(cell_id=f"a{i}"), dict(base=b), True) for i, b in enumerate([4.0, 5.0, 6.0])]
    spec += [(dict(cell_id="np"), dict(base=99.0), False)]
    spec += [(dict(cell_id="oct", rig="oct2026"), dict(base=5.5), True)]
    spec += [(dict(cell_id="od", governor="ondemand"), dict(base=4.1), True)]
    spec += [(dict(cell_id="f", fan_mode="off", cpu_mhz=1500), dict(base=3.9), True)]
    cat, der, qc = _frames(spec)
    t = deploy.idle_table(cat, der, qc).set_index(["rig", "cpu_key", "fan_mode"])
    a = t.loc[("nov2026", "2400", "full")]
    assert a["P_idle_W"] == 5.0 and a["n"] == 3 and a["iqr_W"] == pytest.approx(1.0)   # non-primary 99 ignored
    assert t.loc[("oct2026", "2400", "full"), "P_idle_W"] == 5.5       # own row, not mixed into nov2026
    assert t.loc[("nov2026", "ondemand", "full"), "P_idle_W"] == 4.1
    assert t.loc[("nov2026", "1500", "off"), "P_idle_W"] == 3.9


def _soak_spec():
    spec = [({"cell_id": f"n{i}"}, dict(e=e, P=P, base=b), True)
            for i, (e, P, b) in enumerate([(2900, 9.8, 4.9), (3100, 10.2, 5.1)])]            # mean 3000, 10.0, 5.0
    spec += [({"cell_id": f"fa{i}", "fan_mode": "auto"}, dict(P=9.4, base=4.3, sdur=4000.0, sE=36000.0), True) for i in range(3)]
    spec += [({"cell_id": f"fo{i}", "fan_mode": "off"}, dict(P=8.9, base=3.95, sdur=4000.0, sE=34000.0), True) for i in range(3)]
    spec += [({"cell_id": "oct", "rig": "oct2026", "fan_mode": "off"}, dict(P=1.0, base=1.0), True)]
    spec += [({"cell_id": "sn", "soak_minutes": 30.0}, dict(e=3300.0, lat=290.0, n=6000), True)]
    spec += [({"cell_id": f"m{i}", "base_model": "yolo11m", "variant": "yolo11m"}, dict(e=10000.0), True)
             for i in range(2)]
    spec += [({"cell_id": "sm", "base_model": "yolo11m", "variant": "yolo11m", "soak_minutes": 30.0},
              dict(e=12000.0, lat=1600.0, n=1000), True)]
    return spec


def test_fan_offsets_from_yolo11n_nov_cells_only():
    cat, der, qc = _frames(_soak_spec())
    off = deploy.fan_offsets(deploy.cell_table(cat, der, qc))
    assert off["auto"]["active_W"] == pytest.approx(9.4 - 10.0) and off["auto"]["idle_W"] == pytest.approx(4.3 - 5.0)
    assert off["off"]["active_W"] == pytest.approx(8.9 - 10.0) and off["off"]["idle_W"] == pytest.approx(3.95 - 5.0)
    assert "yolo11n only" in off["assumption"]


def test_soak_factor_mean_and_leave_one_soak_out():
    cat, der, qc = _frames(_soak_spec())
    assert deploy.soak_factor(der, cat) == pytest.approx((3300 / 3000 + 12000 / 10000) / 2)
    assert deploy.soak_factor(der, cat, exclude_cell="sn") == pytest.approx(1.2)
    assert deploy.soak_factor(der, cat, exclude_cell="sm") == pytest.approx(1.1)
    both = deploy.soak_factor(der[~der.cell_id.isin(["sn", "sm"])], cat)
    assert math.isnan(both)


# ------------------------------------------------------------------ validation

def _val_spec():
    spec = list(_soak_spec())
    spec += [({"cell_id": "ser", "mode_a_eligible": True}, dict(e=3000.0, lat=300.0, sdur=4900.0, sE=38000.0), True)]
    spec += [({"cell_id": "fps5", "serve_fps": 5.0}, dict(sdur=30000.0, sE=300000.0), True)]
    spec += [({"cell_id": "od", "base_model": "yolo11s", "variant": "yolo11s", "governor": "ondemand"},
              dict(e=6600.0, lat=640.0, sdur=8400.0, sE=74000.0, base=4.1), True)]
    spec += [({"cell_id": "od2", "base_model": "yolo11s", "variant": "yolo11s", "governor": "ondemand"},
              dict(e=6600.0, lat=640.0, sdur=8400.0, sE=74000.0, base=4.3), True)]
    spec += [({"cell_id": "alien", "base_model": "effb0", "variant": "effb0", "mode_a_eligible": True},
              dict(sdur=3000.0, sE=20000.0), True)]
    spec += [({"cell_id": "npr", "mode_a_eligible": True}, dict(sdur=4900.0, sE=38000.0), False)]
    return spec


def _preds():
    k = lambda m, mhz=2400, rig="nov2026": f"{m}|{m}|fp32|640|1|4|{mhz}|{rig}"
    return pd.DataFrame({"config_key": [k("yolo11n"), k("yolo11s"), k("yolo11m")],
                         "e_img_mJ": [3000.0, 6600.0, 10000.0], "t_img_ms": [300.0, 640.0, 1500.0],
                         "lo90_mJ": [2800.0, 6000.0, 9000.0], "hi90_mJ": [3300.0, 7000.0, 11000.0],
                         "flagged": [False, False, True]})


def _validated():
    cat, der, qc = _frames(_val_spec())
    return deploy.validate(cat, der, qc, _preds())


def test_validate_serve_default_hand_value_and_skips():
    v = _validated()
    r = v[(v.case == "serve-default")]
    assert list(r.cell_id) == ["ser"]            # non-primary 'npr' ignored, 'alien' has no prediction
    assert v.attrs["skipped"]["serve-default"] == 1
    row = r.iloc[0]
    # 10 images x 3 J + idle (4.9 - 3.0) s x P_idle (median baseline 2400 full nov2026 cells = 5.0 W)
    assert row.source == "lomo" and row.phase == "serve" and row.measured_J == pytest.approx(38.0)
    idle = v.attrs["idle"].set_index(["rig", "cpu_key", "fan_mode"]).loc[("nov2026", "2400", "full"), "P_idle_W"]
    assert row.predicted_J == pytest.approx(30.0 + 1.9 * idle)
    assert row.ape == pytest.approx(abs(row.predicted_J - 38.0) / 38.0)
    assert row.lo90_J < row.predicted_J < row.hi90_J and not row.flagged


def test_validate_serve_fps_saturation_flag_and_batch():
    v = _validated()
    r = v[v.case == "serve-fps"].iloc[0]
    assert r.cell_id == "fps5" and "serve_fps=5" in r.notes and r.measured_J == pytest.approx(300.0)
    cat, der, qc = _frames([({"cell_id": "b", "mode_a_eligible": True, "batch": 4},
                             dict(sdur=4000.0, sE=30000.0, batch=4), True),
                            ({"cell_id": "other"}, dict(), True)])      # idle power needs another cell
    k = "yolo11n|yolo11n|fp32|640|4|4|2400|nov2026"
    p = pd.DataFrame({"config_key": [k], "e_img_mJ": [1000.0], "t_img_ms": [150.0]})
    v2 = deploy.validate(cat, der, qc, p).iloc[0]
    # n_images = 10 requests x batch 4 = 40 images, busy 6 s > 4 s duration: clipped and saturated
    assert v2.saturated and v2.predicted_J == pytest.approx(40.0)


def test_validate_ondemand_forward_and_serve():
    v = _validated()
    r = v[(v.case == "ondemand") & (v.cell_id == "od")].set_index("phase")
    assert set(r.index) == {"forward", "serve"}
    assert r.loc["forward", "predicted_J"] == pytest.approx(6.6) and r.loc["forward", "ape"] == pytest.approx(0.0)
    # serve: 10 x 6.6 J + (8.4 - 6.4) s x the ondemand idle of the OTHER ondemand cell (4.3 W), not its own 4.1
    assert r.loc["serve", "predicted_J"] == pytest.approx(66.0 + 2.0 * 4.3)
    assert r.loc["serve", "measured_J"] == pytest.approx(74.0)


def test_validate_idle_excludes_the_validated_cell_for_every_case():
    # leave-one-cell-out: a validated cell's own baseline never enters the idle power predicting it
    spec = [(dict(cell_id=f"a{i}"), dict(base=b), True) for i, b in enumerate([4.0, 5.0, 6.0])]
    spec += [({"cell_id": "ser", "mode_a_eligible": True}, dict(sdur=4900.0, sE=38000.0, base=9.0), True)]
    cat, der, qc = _frames(spec)
    v = deploy.validate(cat, der, qc, _preds())
    row = v[v.cell_id == "ser"].iloc[0]
    # median(4, 5, 6) = 5.0 without the cell; including its own 9.0 it would be 5.5
    assert row.predicted_J == pytest.approx(30.0 + 1.9 * 5.0)
    assert v.attrs["idle"].set_index(["rig", "cpu_key", "fan_mode"]).loc[("nov2026", "2400", "full"), "P_idle_W"] == 5.5
    # ondemand and fan cells: each from the table that leaves it out
    spec = [({"cell_id": f"od{i}", "base_model": "yolo11s", "variant": "yolo11s", "governor": "ondemand"},
             dict(e=6600.0, lat=640.0, sdur=8400.0, sE=74000.0, base=b), True) for i, b in enumerate([4.0, 4.4, 5.0])]
    cat, der, qc = _frames(spec)
    o = deploy.validate(cat, der, qc, _preds())
    got = o[o.phase == "serve"].set_index("cell_id")["predicted_J"]
    for cid, other_median in (("od0", 4.7), ("od1", 4.5), ("od2", 4.2)):
        assert got[cid] == pytest.approx(66.0 + 2.0 * other_median)


def test_validate_skips_a_cell_with_no_idle_row_apart_from_itself():
    spec = [({"cell_id": "only", "base_model": "yolo11s", "variant": "yolo11s", "governor": "ondemand"},
             dict(e=6600.0, lat=640.0, sdur=8400.0, sE=74000.0, base=4.1), True)]
    cat, der, qc = _frames(spec)
    v = deploy.validate(cat, der, qc, _preds())
    assert v.empty and v.attrs["skipped"]["ondemand"] == 2          # forward and serve: no leak-free idle row


def test_idle_table_exclude_cell():
    spec = [(dict(cell_id=f"a{i}"), dict(base=b), True) for i, b in enumerate([4.0, 5.0, 9.0])]
    cat, der, qc = _frames(spec)
    full = deploy.idle_table(cat, der, qc).iloc[0]
    loco = deploy.idle_table(cat, der, qc, exclude_cell="a2").iloc[0]
    assert (full["P_idle_W"], full["n"]) == (5.0, 3) and (loco["P_idle_W"], loco["n"]) == (4.5, 2)
    assert deploy.idle_table(cat, der, qc, exclude_cell="nope").iloc[0]["n"] == 3


def test_validate_soak_uses_other_soak_only_and_reports_no_soak():
    v = _validated()
    s = v[v.case == "soak"].set_index("cell_id")
    # soak 'sn' (yolo11n, 6000 images): factor from the yolo11m soak only = 1.2; duration 6000 x 0.3 s = 1800 s
    assert s.loc["sn", "predicted_J_no_soak"] == pytest.approx(6000 * 3.0)
    assert s.loc["sn", "predicted_J"] == pytest.approx(6000 * 3.0 * 1.2)
    assert s.loc["sn", "measured_J"] == pytest.approx(6000 * 3.3)
    # soak 'sm' (yolo11m, 1000 images x 1.5 s = 1500 s): factor from the yolo11n soak = 1.1, ramped by 1500/1800
    assert s.loc["sm", "predicted_J"] == pytest.approx(1000 * 10.0 * (1 + 0.1 * 1500 / 1800))
    assert "other soak only" in s.loc["sm", "notes"]


def test_validate_fan_rows_are_labelled_not_a_validation():
    v = _validated()
    f = v[v.case == "fan"]
    assert len(f) == 12 and all("not a validation" in n for n in f.notes)
    fwd = f[(f.phase == "forward") & f.cell_id.str.startswith("fa")].iloc[0]
    # per image 3 J + active offset (9.4 - 10.0) W x 0.3 s busy
    assert fwd.predicted_J == pytest.approx(3.0 - 0.6 * 0.3)
    assert set(f.phase) == {"forward", "serve"}
    assert v.attrs["skipped"]["fan"] == 2       # the oct2026 fan cell: forward and serve rows, no prediction


def test_validate_skips_when_no_predictions():
    cat, der, qc = _frames(_val_spec())
    v = deploy.validate(cat, der, qc, _preds().iloc[0:0])
    assert v.empty and list(v.columns) == deploy.VALIDATION_COLS
    assert v.attrs["skipped"]["serve-default"] == 2


# ------------------------------------------------------------------ Mode A prediction loading

def _mode_a_df(target, scale):
    rows = []
    for cand in ("AUTO", "M1Ridge"):
        for k, y in (("A", 10.0), ("B", 20.0)):
            rows.append({"split": "S-LOMO", "fold": k, "similarity_class": "x", "config_key": k, "base_model": k,
                         "candidate": cand, "y_true": y, "y_pred": y * scale * (1 if cand == "AUTO" else 7),
                         "lo90": y * 0.9, "hi90": y * 1.1, "flagged": k == "B", "target": target})
    rows.append({**rows[0], "split": "S-LSO", "y_pred": -1.0})
    return pd.DataFrame(rows)


def test_preds_from_mode_a_takes_slomo_auto_rows():
    p = deploy.preds_from_mode_a(_mode_a_df("energy_mJ", 1.0), _mode_a_df("latency_ms", 0.1))
    assert list(p.columns) == ["config_key", "e_img_mJ", "t_img_ms", "lo90_mJ", "hi90_mJ", "flagged"]
    assert list(p.config_key) == ["A", "B"] and list(p.e_img_mJ) == [10.0, 20.0]
    assert list(p.t_img_ms) == pytest.approx([1.0, 2.0]) and list(p.flagged) == [False, True]


def test_load_mode_a_preds_latest_run_and_clear_failure(tmp_path):
    log, d = tmp_path / "runs.jsonl", tmp_path / "mode_a"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="energy_mJ"):
        deploy.load_mode_a_preds(log, d)
    recs = [("old", "energy_mJ", 5.0), ("e1", "energy_mJ", 1.0), ("l1", "latency_ms", 0.1)]
    for rid, tgt, sc in recs:
        _mode_a_df(tgt, sc).to_parquet(d / f"{rid}.parquet")
    lines = [{"run_id": rid, "command": "mode-a", "split": "S-LOMO", "target": tgt} for rid, tgt, _ in recs]
    lines.insert(1, {"run_id": "x", "command": "qc"})
    lines.append({"run_id": "gone", "command": "mode-a", "split": "S-LOMO", "target": "energy_mJ"})  # no parquet
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    p = deploy.load_mode_a_preds(log, d)
    assert list(p.e_img_mJ) == [10.0, 20.0]       # the latest energy run with a parquet ('e1', scale 1.0)


# ------------------------------------------------------------------ real data

def test_real_idle_fan_and_soak(real_data, capsys):
    import data
    import derive
    import settings
    cat = data.catalogue()
    der = derive.derive_all(cat, cache=True)
    qc = pd.read_csv(settings.HERE / "reports" / "qc_cells.csv")
    idle = deploy.idle_table(cat, der, qc)
    off = deploy.fan_offsets(deploy.cell_table(cat, der, qc))
    f = deploy.soak_factor(der, cat, qc=qc)
    with capsys.disabled():
        print("\n" + idle.to_string(index=False))
        print("fan offsets:", json.dumps(off))
        print("soak factor:", f)
    row = idle.set_index(["rig", "cpu_key", "fan_mode"]).loc[("nov2026", "2400", "full")]
    assert 4.0 < row["P_idle_W"] < 6.0 and row["n"] > 50
    assert ("nov2026", "ondemand", "full") in set(map(tuple, idle[["rig", "cpu_key", "fan_mode"]].values))
    assert off["off"]["idle_W"] < 0 < f
    assert deploy.soak_factor(der, cat, exclude_cell="yolo11n_soak30_640_t4_2400_r00", qc=qc) > 0


def test_cli_deploy_validate_passes_the_seed_and_records_run_ids(tmp_path, monkeypatch, capsys):
    import cli
    import data
    import derive
    import runlog
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(runlog, "DEFAULT_LOG", log)
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    pd.DataFrame({"cell_id": ["a"], "primary": [True]}).to_csv(tmp_path / "qc_cells.csv", index=False)
    seen = {}
    preds = _preds()
    preds.attrs["run_ids"] = {"energy_mJ": "rid_e", "latency_ms": "rid_l"}
    preds.attrs["seed_matched"] = {"energy_mJ": True, "latency_ms": False}

    def fake_load(log_path=None, mode_a_dir=None, seed=None):
        seen["seed"] = seed
        return preds
    cat, der, qc = _frames(_val_spec())
    monkeypatch.setattr(deploy, "load_mode_a_preds", fake_load)
    monkeypatch.setattr(data, "catalogue", lambda *a, **k: cat)
    monkeypatch.setattr(derive, "derive_all", lambda *a, **k: der)
    qc.to_csv(tmp_path / "qc_cells.csv", index=False)
    assert cli.main(["deploy-validate", "--seed", "77"]) == 0
    assert seen["seed"] == 77
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["command"] == "deploy-validate" and rec["seed"] == 77
    assert rec["mode_a_run_ids"] == {"energy_mJ": "rid_e", "latency_ms": "rid_l"}
    assert rec["mode_a_seed_matched"] == {"energy_mJ": True, "latency_ms": False}
    assert cli.main(["deploy-validate"]) == 0                      # default: the config seed
    import settings
    assert seen["seed"] == settings.seed()
    assert json.loads(log.read_text().splitlines()[-1])["seed"] == settings.seed()


def test_cli_deploy_validate_fails_clearly_without_mode_a_runs(tmp_path, monkeypatch, capsys):
    import cli
    import runlog
    monkeypatch.setattr(runlog, "DEFAULT_LOG", tmp_path / "none.jsonl")
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    pd.DataFrame({"cell_id": ["a"], "primary": [True]}).to_csv(tmp_path / "qc_cells.csv", index=False)
    assert cli.main(["deploy-validate"]) == 1
    assert "no S-LOMO Mode A run" in capsys.readouterr().err
