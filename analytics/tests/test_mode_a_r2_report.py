"""R2 comparison report: formatting of synthetic prediction frames."""
import json

import numpy as np
import pandas as pd

import mode_a_r2_report as rep


def frame(scale, cands=("B3LogLinear", "M1Ridge", "M4Physics", "AUTO")):
    rows = []
    for c in cands:
        for i, m in enumerate(["a", "b", "c"]):
            for j in range(2):
                y = 10.0 * (i + 1) + j
                p = y * (1 + scale * (1 + i))
                rows.append({"fold": m, "base_model": m, "candidate": c, "y_true": y, "y_pred": p,
                             "lo90": p * 0.8, "hi90": p * 1.6,
                             "params": json.dumps({"candidate": "M4Physics", "params": {}}) if c == "AUTO" else "{}"})
    return pd.DataFrame(rows)


def test_candidate_stats_has_the_r2_metrics_and_interval_width():
    s = rep.candidate_stats(frame(0.1))
    assert list(s.index) == ["B3LogLinear", "M1Ridge", "M4Physics", "AUTO"]
    assert {"mdape", "mape", "maxape", "p90ape", "r2_log", "width"} <= set(s.columns)
    assert np.allclose(s["width"], 2.0) and np.isclose(s.loc["AUTO", "maxape"], 0.3, atol=0.02)


def test_build_contains_the_tables_for_both_revisions():
    key = ("S-LOMO", "energy_mJ")
    md = rep.build({key: frame(0.3)}, {key: frame(0.1), ("RANDOM", "energy_mJ"): frame(0.05)}, {7: frame(0.1)})
    assert "## S-LOMO, energy per image" in md and "M4Physics" in md and "R1 MdAPE" in md
    assert "AUTO's choice per outer fold (R2): M4Physics x3" in md
    assert "| a | " in md and "## AUTO MdAPE (%) by split and target" in md and "| RANDOM | energy_mJ | - |" in md
    assert "| 7 |" in md
