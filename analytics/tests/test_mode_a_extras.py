"""mode-a-extras: AUTO's most chosen candidate, final fits, learning curve and the v2-only against pooled runs."""
import json
import os

import pandas as pd
import pytest

import mode_a
import mode_a_extras as ex
import report_inputs as ri
import runlog
import splits
from synth_report import MASTER
from test_mode_a import SMALL_GRIDS, make_dataset


def auto_rows(choices: list) -> pd.DataFrame:
    return pd.DataFrame([{"candidate": "AUTO", "fold": f"f{i}", "params": json.dumps({"candidate": c, "params": {}})}
                         for i, c in enumerate(choices)] +
                        [{"candidate": "AUTO", "fold": "f0", "params": json.dumps({"candidate": choices[0], "params": {}})}] * 3
                        + [{"candidate": "M1Ridge", "fold": "f9", "params": "{}"}])


def test_auto_choice_counts_one_vote_per_fold_and_ties_go_to_m1_m2_m3_order():
    assert ex.auto_counts(auto_rows(["M2Trees", "M1Ridge", "M2Trees"])) == {"M2Trees": 2, "M1Ridge": 1}
    assert ex.auto_choice(auto_rows(["M2Trees", "M1Ridge", "M2Trees"])) == "M2Trees"
    assert ex.auto_choice(auto_rows(["M3Factored", "M2Trees", "M1Ridge"])) == "M1Ridge"
    assert ex.auto_choice(auto_rows(["M3Factored", "M2Trees"])) == "M2Trees"
    with pytest.raises(ValueError):
        ex.auto_choice(pd.DataFrame({"candidate": ["B0Median"], "fold": ["f"], "params": ["{}"]}))


@pytest.fixture(scope="module")
def extras_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("extras")
    os.environ["ANALYTICS_DATA"] = str(root / "data")
    try:
        t, desc = make_dataset([*splits.YOLO, "effb0"], clocks=(1500, 2400), threads=(4,))
        t["rig"] = ["oct2026" if m == "effb0" else "nov2026" for m in t["base_model"]]
        paths = ri.Paths(root / "runs.jsonl", root / "mode_a", root / "mode_l", root / "extras", root / "t.csv", root / "q.csv",
                         root / "d.csv", root / "out", root / "out" / "fig")
        paths.mode_a_dir.mkdir()
        for target in ri.TARGETS:
            res = mode_a.run_split(t, [f for f in splits.make_splits(t, MASTER)["S-LOMO"]], target, MASTER, desc, grids=SMALL_GRIDS)
            res.to_parquet(paths.mode_a_dir / f"{target}.parquet")
            runlog.append({"run_id": target, "command": "mode-a", "split": "S-LOMO", "target": target, "seed": MASTER}, paths.runs_log)
        out = ex.run(t, desc, MASTER, paths, grids=SMALL_GRIDS, max_subsets=3)
        yield t, paths, out
    finally:
        os.environ.pop("ANALYTICS_DATA", None)


def test_final_fits_are_logged_with_sha256_and_the_auto_choice(extras_run):
    t, paths, out = extras_run
    recs = ri.read_records(paths.runs_log)
    fits = {r["target"]: r for r in recs if r.get("part") == "fit_final"}
    assert set(fits) == set(ri.TARGETS)
    for target, r in fits.items():
        assert len(r["sha256"]) == 64 and r["candidate"] in ex.ORDER and sum(r["auto_counts"].values()) == 6 + 1
        assert r["source_run_id"] == target and os.path.isfile(r["path"])
    assert out["fits"]["energy_mJ"]["candidate"] == fits["energy_mJ"]["candidate"]


def test_learning_curve_uses_the_energy_fit_and_is_saved(extras_run):
    t, paths, out = extras_run
    lc = pd.read_parquet(paths.extras_dir / f"{out['learning_curve']}.parquet")
    assert set(lc["k"]) == {2, 3, 4, 5} and set(lc["heldout"]) == set(splits.YOLO)
    rec = [r for r in ri.read_records(paths.runs_log) if r.get("part") == "learning_curve"][0]
    fit = [r for r in ri.read_records(paths.runs_log) if r.get("part") == "fit_final" and r["target"] == "energy_mJ"][0]
    assert rec["candidate"] == fit["candidate"] and rec["params"] == fit["params"]


def test_v2_only_run_trains_and_tests_on_nov2026_rows_only_over_yolo_folds(extras_run):
    t, paths, out = extras_run
    pooled = pd.read_parquet(paths.extras_dir / f"{out['v2_vs_pooled']['pooled']}.parquet")
    v2 = pd.read_parquet(paths.extras_dir / f"{out['v2_vs_pooled']['v2only']}.parquet")
    nov = set(t.loc[t["rig"] == "nov2026", "config_key"])
    assert set(v2["config_key"]) <= nov and set(v2["fold"]) == set(splits.YOLO) == set(pooled["fold"])
    assert (pooled["target"] == "energy_mJ").all()
    recs = {r["training"]: r for r in ri.read_records(paths.runs_log) if r.get("part") == "v2_vs_pooled"}
    assert recs["pooled"]["n_configs_train_table"] == len(t) and recs["v2only"]["n_configs_train_table"] == len(nov)


def test_extras_outputs_load_back_into_the_report_inputs(extras_run):
    t, paths, out = extras_run
    got = ri.load_extras(paths, ri.read_records(paths.runs_log), MASTER)
    assert set(got["fits"]) == set(ri.TARGETS) and len(got["learning_curve"].df) and len(got["pooled"].df) and len(got["v2only"].df)
