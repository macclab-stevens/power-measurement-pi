"""A lockbox cell of a training model (yolov8n at 480 px) must never reach a training table, window table,
idle table, Mode L fold or report input. Every check forces the lockbox row's eligibility flags to True, so it
tests the explicit `data.training_cells` filter and not only the flags."""
import numpy as np
import pandas as pd
import pytest

import data
import deploy
import eda_trace_write
import eda_write
import static
import tables
import windows
import mode_l
from conftest import make_cell

NAN = float("nan")
LAYERS = "path,class,leaf,params,macs,bytes_moved,count\nm.c,Conv2d,True,{p},{m},{b},1\nm.fc,Linear,True,10,100,80,1\n"


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """(cat without the lockbox cell, cat with it and its flags forced True, der, qc, root)."""
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    meta = {"variant": "yolov8n", "task": "image"}
    for i in range(2):
        make_cell(tmp_path, "v2", f"yolov8n_640_r0{i}", meta=meta, index={"model_key": "yolov8n"},
                  files={"layer_table.csv": LAYERS.format(p=100, m=1000, b=900)})
    without = data.catalogue()
    make_cell(tmp_path, "lockbox", "yolov8n_480_r00", meta={**meta, "imgsz": 480}, index={"model_key": "yolov8n"},
              files={"layer_table.csv": LAYERS.format(p=999, m=5000, b=4000), "counters.csv": "a\n1\n"})
    with_ = data.catalogue()
    forced = with_.assign(mode_a_eligible=True, mode_l_eligible=True)
    rows = [{"cell_id": c, "mode": "forward", "batch": 1, "energy_mJ_per_image": 1000.0 + 50 * (c.startswith("yolov8n_480")),
             "latency_ms_per_image": 100.0, "mean_P_W": 7.0 + 3 * c.startswith("yolov8n_480"),
             "dyn_energy_mJ_per_image": 500.0, "baseline_W": 5.0 + 4 * c.startswith("yolov8n_480"),
             "serve_n_requests": NAN, "serve_dur_ms": NAN, "serve_E_mJ": NAN} for c in forced["cell_id"]]
    der = pd.DataFrame(rows)
    qc = pd.DataFrame({"cell_id": forced["cell_id"], "primary": True})
    return without, forced, der, qc, tmp_path


def test_catalogue_never_marks_a_lockbox_cell_eligible(roots):
    _, _, _, _, root = roots
    cat = data.catalogue()
    lock = cat[cat["source"] == "lockbox"].iloc[0]
    assert lock["base_model"] == "yolov8n" and bool(lock["has_counters"])
    assert not lock["mode_a_eligible"] and not lock["mode_l_eligible"]
    assert cat[cat["source"] == "v2"]["mode_a_eligible"].all()
    assert list(data.training_cells(cat)["source"].unique()) == ["v2"]


def test_mode_a_table_and_static_table_ignore_the_lockbox_cell(roots):
    without, forced, der, qc, _ = roots
    st = static.static_table(without)
    assert list(st.index) == [("yolov8n", 640)]
    pd.testing.assert_frame_equal(static.static_table(forced), st)          # no (yolov8n, 480) row, same features
    pd.testing.assert_frame_equal(tables.mode_a_table(forced, der, qc, st), tables.mode_a_table(without, der, qc, st))
    assert len(tables.mode_a_table(forced, der, qc, st)) == 1


def test_window_tables_never_read_a_lockbox_cell_and_keep_their_cache_key(roots, monkeypatch):
    without, forced, _, _, _ = roots
    seen = []

    def fake(cell, meta):
        seen.append(cell.name)
        return pd.DataFrame({c: [0.0, 1.0] for c in windows.WINDOW_COLS + windows.PMIC_COLS if c != "cell_id"}).assign(cell_id=cell.name)
    monkeypatch.setattr(windows, "cell_windows", fake)
    monkeypatch.setattr(windows, "pmic_windows", fake)
    a, pa = windows.build_windows(without, force=True), windows.build_pmic_windows(without, force=True)
    n = len(seen)
    b, pb = windows.build_windows(forced, force=True), windows.build_pmic_windows(forced, force=True)
    assert not [c for c in seen if c.startswith("yolov8n_480")] and len(seen) == 2 * n
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pa, pb)
    assert windows._cache_key(data.training_cells(forced)) == windows._cache_key(without)   # the cache survives the lockbox


def test_idle_table_fan_offsets_and_soak_factor_ignore_the_lockbox_cell(roots):
    without, forced, der, qc, _ = roots
    pd.testing.assert_frame_equal(deploy.idle_table(forced, der, qc), deploy.idle_table(without, der, qc))
    assert deploy.idle_table(forced, der, qc)["P_idle_W"].iloc[0] == 5.0
    t_with, t_without = deploy.cell_table(forced, der, qc), deploy.cell_table(without, der, qc)
    flags = ["mode_a_eligible", "mode_l_eligible"]                     # forced True on the with-lockbox catalogue
    pd.testing.assert_frame_equal(t_with.drop(columns=flags), t_without.drop(columns=flags))
    assert deploy.fan_offsets(t_with).keys() == deploy.fan_offsets(t_without).keys()
    f1, f2 = deploy.soak_factor(der, forced, qc=qc), deploy.soak_factor(der, without, qc=qc)
    assert (np.isnan(f1) and np.isnan(f2)) or f1 == f2


def test_mode_l_selection_drops_lockbox_windows_even_for_a_yolo_base_model(roots):
    without, forced, _, qc, _ = roots
    w = pd.DataFrame({"cell_id": ["yolov8n_640_r00"] * 3 + ["yolov8n_640_r01"] * 3 + ["yolov8n_480_r00"] * 3,
                      "w_idx": [0, 1, 2] * 3, "P_W": 6.0})
    a, b = mode_l._prep(w, without, qc), mode_l._prep(w, forced, qc)
    pd.testing.assert_frame_equal(a, b)
    assert "yolov8n_480_r00" not in set(b["cell_id"])


@pytest.mark.parametrize("write", ["config", "trace"])
def test_eda_entry_points_drop_lockbox_cells_first(roots, monkeypatch, write):
    _, forced, der, qc, root = roots

    class Stop(Exception):
        pass
    got = []
    real = data.training_cells

    def spy(cat):
        got.append(real(cat))
        raise Stop
    monkeypatch.setattr(data, "training_cells", spy)
    with pytest.raises(Stop):
        if write == "config":
            eda_write.write_eda_config(forced, der, qc, pd.DataFrame(), root / "out")
        else:
            eda_trace_write.write_eda_trace(forced, der, qc, root / "out")
    assert len(got) == 1 and set(got[0]["source"]) == {"v2"}
