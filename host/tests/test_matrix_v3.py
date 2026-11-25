"""v3 cell plan: counts, unique ids, rounds, hooks flags and arguments run.py accepts."""
import matrix_v3 as m3


def test_part_sizes_and_unique_ids():
    sizes = {p: len(m3.cells(p)) for p in m3.PARTS}
    assert sizes == {"calibration": 42, "additions": 21, "lockbox2": 15}
    ids = [r["cell_id"] for p in m3.PARTS for r in m3.cells(p)]
    assert len(ids) == len(set(ids))


def test_rounds_hold_every_configuration_once():
    rows = m3.cells("calibration")
    for r in range(m3.REPEATS):
        rnd = [x for x in rows if x["round"] == r]
        assert len(rnd) == len(m3.CALIBRATION)
        assert len({x["cell_id"].rsplit("_r", 1)[0] for x in rnd}) == len(m3.CALIBRATION)


def test_calibration_pairs_hooks_on_and_off_and_others_off():
    cal = m3.cells("calibration")
    assert {r["layer_hooks"] for r in cal} == {"on", "off"}
    assert sum(r["layer_hooks"] == "on" for r in cal) == 21
    for part in ("additions", "lockbox2"):
        assert {r["layer_hooks"] for r in m3.cells(part)} == {"off"}


def test_args_shape():
    r = next(x for x in m3.cells("lockbox2") if x["model_key"] == "resnet34")
    a = r["args"]
    assert a[a.index("--model") + 1] == "torchvision:resnet34" and "--check-image" not in a
    assert a[a.index("--layer-hooks") + 1] == "off" and "--no-bench" in a and "--bench" not in a
    assert a[a.index("--frames") + 1] == "94" and a[a.index("--sysfs-period-s") + 1] == "0.1"
    y = next(x for x in m3.cells("lockbox2") if x["model_key"] == "yolo12m")["args"]
    assert y[y.index("--model") + 1] == "yolo12m.pt" and y[y.index("--frames") + 1] == "20"


def test_deterministic():
    assert m3.cells("additions") == m3.cells("additions")
