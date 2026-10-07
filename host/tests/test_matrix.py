# host/tests/test_matrix.py
import matrix


def test_full_matrix_counts_and_unique_ids():
    rows = matrix.cells()
    n_models = len(matrix.MODELS)
    assert len(rows) == n_models * (matrix.MAIN_REPEATS
                                    + matrix.SWEEP_REPEATS * len(matrix.SWEEP_MHZ))
    assert len({r["cell_id"] for r in rows}) == len(rows)
    for key in matrix.MODELS:
        mine = [r for r in rows if r["model_key"] == key]
        assert sum(r["cpu_mhz"] == 2400 for r in mine) == 10
        assert sum(r["cpu_mhz"] == 1500 for r in mine) == 3
        assert sum(r["cpu_mhz"] == 2000 for r in mine) == 3


def test_every_cell_is_pinned_monsoon_with_fan():
    for r in matrix.cells():
        a = r["args"]
        assert a[a.index("--cpu-mhz") + 1] == str(r["cpu_mhz"])
        assert "--fan-full" in a and a[a.index("--sampler") + 1] == "monsoon"
        if a[a.index("--task") + 1] == "train":
            assert "--no-bench" in a and "--serve" not in a


def test_rounds_spread_repeats_and_sweeps():
    rows = matrix.cells()
    rounds = [r["round"] for r in rows]
    assert rounds == sorted(rounds)                       # manifest is in round order
    sweep_rounds = sorted({r["round"] for r in rows if r["cpu_mhz"] != 2400})
    assert sweep_rounds == matrix.sweep_rounds() == [1, 5, 8]
    for rnd in range(matrix.MAIN_REPEATS):
        main = [r for r in rows if r["round"] == rnd and r["cpu_mhz"] == 2400]
        assert sorted(r["model_key"] for r in main) == sorted(matrix.MODELS)


def test_order_is_seeded_and_shuffled():
    a, b = matrix.cells(seed=7), matrix.cells(seed=7)
    assert [r["cell_id"] for r in a] == [r["cell_id"] for r in b]
    first = [r["model_key"] for r in a if r["round"] == 0]
    assert first != list(matrix.MODELS)
    assert [r["cell_id"] for r in matrix.cells(seed=8)] != [r["cell_id"] for r in a]


def test_smoke_and_only():
    smoke = matrix.cells(smoke=True)
    assert [r["cell_id"] for r in smoke] == ["smoke_yolo11n_2400_r00",
                                             "smoke_train_lm64_adam_2400_r00",
                                             "smoke_train_resnet18_sgd_2400_r00"]
    only = matrix.cells(only=["lm16"])
    assert {r["model_key"] for r in only} == {"lm16"} and len(only) == 16
    import pytest
    with pytest.raises(ValueError):
        matrix.cells(only=["nope"])
