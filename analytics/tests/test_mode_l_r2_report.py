"""R2 Mode L comparison report: tables from synthetic LOMO frames."""
import numpy as np
import pandas as pd
import pytest

import mode_l
import mode_l_models as mm
import mode_l_r2_report as rep
from test_mode_l import make_windows


class Const:
    def fit(self, train_w, params, sample_weight):
        self.c = params["c"]
        return self

    def predict(self, x):
        return np.full(len(x), self.c)


@pytest.fixture(scope="module")
def lomo():
    w, cat = make_windows(models=("yolov8n", "yolov8s", "yolov8m"))
    rungs = {"L0": mm.Rung("L0", lambda s: mm.L0Lookup(), [{}]),
             "L5-cmos": mm.Rung("L5-cmos", lambda s: mm.L5Cmos("ridge", s), [{"alpha": 0.01}, {"alpha": 10.0}]),
             "c": mm.Rung("c", lambda s: Const(), [{"c": 6.0}])}
    return mode_l.run_lomo(w, cat, seed=0, rungs=rungs)


def test_rung_stats_columns_and_values(lomo):
    s = rep.rung_stats(lomo)
    assert list(s.index) == ["L0", "L5-cmos", "c"]
    g = lomo[lomo["rung"] == "L5-cmos"]
    assert s.loc["L5-cmos", "whole"] == pytest.approx(g["ape"].median())
    assert s.loc["L5-cmos", "block_1s"] == pytest.approx(g["ape_1s"].median())
    assert s.loc["L5-cmos", "mae_fwd_W"] == pytest.approx(mode_l.pooled_window_metrics(g)["mae_fwd_W"])
    assert s.loc["L5-cmos", "r2_fwd"] > 0.9 > s.loc["c", "r2_fwd"]          # P = 4 + 5 busy is recovered; a constant is not
    r1 = rep.rung_stats(lomo.drop(columns=[c for c in lomo.columns if c.endswith("_fwd") or c.endswith("_all")]))
    assert r1["r2_fwd"].isna().all() and r1["mae_fwd_W"].isna().all()


def test_slip_summary_and_build(lomo):
    attrs = {"slip_corrected": 2, "slip_no_pulses": 1, "n_intervals": 1000, "dropped": 10,
             "slip_cells": [{"cell_id": "a", "max_corr_ms": 4.0, "n_windows": 100, "dP_med_W": 0.001, "dP_p99_W": 0.2,
                             "dP_max_W": 0.9, "n_windows_changed": 30, "n_windows_gt_0p1W": 2},
                            {"cell_id": "b", "max_corr_ms": 20.0, "n_windows": 50, "dP_med_W": 0.003, "dP_p99_W": 0.4,
                             "dP_max_W": 1.5, "n_windows_changed": 20, "n_windows_gt_0p1W": 3}]}
    sl = rep.slip_summary(attrs)
    assert sl["n_corrected"] == 2 and sl["n_no_pulses"] == 1 and sl["n_windows"] == 150 and sl["n_gt_0p1W"] == 5
    assert sl["max_W"] == 1.5 and sl["max_corr_ms"] == 20.0 and sl["max_of_cell_med_W"] == 0.003
    r1 = lomo[lomo["rung"] != "L5-cmos"].drop(columns=["n_fwd"])
    md = rep.build(r1, lomo, sl, {"r1": "aaa", "r2": "bbb"}, {"L5-cmos": {"yolov8n": '{"alpha": 10.0}'}})
    assert "## Ladder: R2 with R1 beside it" in md and "| L5-cmos |" in md and "fwd R^2" in md
    assert "Cells corrected: 2" in md and "1.500 W" in md and "`bbb`" in md
    row = [x for x in md.splitlines() if x.startswith("| L0 |")][0].split("|")
    assert row[-4].strip() != "-"                       # R1 columns filled for a rung that R1 has
    row5 = [x for x in md.splitlines() if x.startswith("| L5-cmos |")][0].split("|")
    assert row5[-4].strip() == "-"                      # and blank for the R2-only rung
    assert rep.build(None, lomo, {"n_corrected": 0, "n_no_pulses": 0}, {}).count("No cell was corrected") == 1


def test_r1_run_id_from_results_md():
    txt = "| a |\n- Mode L, whole-cell energy: L1 against L0 (n = 38; run_id `acc4f1bf2b524febb2565e1d8fe3e79b`).\n"
    assert rep.r1_run_id(txt) == "acc4f1bf2b524febb2565e1d8fe3e79b"
    assert rep.r1_run_id("nothing") is None
