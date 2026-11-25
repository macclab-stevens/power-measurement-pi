"""v4 cell plan: counts, unique ids, rounds, hooks off everywhere, arguments of variants."""
import matrix_v4 as m4


def test_part_sizes_and_unique_ids():
    assert len(m4.DEPLOY) == len(set(m4.DEPLOY))
    sizes = {p: len(m4.cells(p)) for p in m4.PARTS}
    assert sizes == {"deploy": 2 * len(m4.DEPLOY), "lockbox3": 18}
    ids = [r["cell_id"] for p in m4.PARTS for r in m4.cells(p)]
    assert len(ids) == len(set(ids))


def test_rounds_hold_every_configuration_once():
    rows = m4.cells("deploy")
    for r in range(m4.REPEATS["deploy"]):
        rnd = [x for x in rows if x["round"] == r]
        assert len({x["cell_id"].rsplit("_r", 1)[0] for x in rnd}) == len(m4.DEPLOY)


def test_hooks_off_everywhere():
    for p in m4.PARTS:
        for r in m4.cells(p):
            a = r["args"]
            assert r["layer_hooks"] == "off" and a[a.index("--layer-hooks") + 1] == "off" and "--no-bench" in a
            assert a[a.index("--perf-counters") + 1] == "on"


def test_variant_args():
    rows = {r["cell_id"]: r["args"] for r in m4.cells("deploy")}
    i8 = rows["v4_yolo11n_int8s_640_t4_1500_r00"]
    assert i8[i8.index("--quant") + 1] == "int8-static" and i8[i8.index("--quant-weights") + 1] == "models/yolo11n_int8s.pt"
    f16 = rows["v4_yolo11n_fp16_640_t4_2400_r00"]
    assert f16[f16.index("--quant") + 1] == "fp16"
    b8 = rows["v4_yolo11n_640_t4_b8_2400_r00"]
    assert b8[b8.index("--batch") + 1] == "8" and b8[b8.index("--frames") + 1] == "5"
    t1 = rows["v4_yolo11s_640_t1_2400_r00"]
    assert t1[t1.index("--threads") + 1] == "1"
    m = rows["v4_yolo11m_960_t4_2400_r00"]
    assert m[m.index("--frames") + 1] == "20"
    r18 = rows["v4_resnet18_320_t4_2400_r00"]
    assert r18[r18.index("--model") + 1] == "torchvision:resnet18" and r18[r18.index("--frames") + 1] == "91"


def test_lockbox3_models_never_collected():
    seen = set(m4.m3.SHORT.values()) | {"yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo12n",
                                         "yolo12s", "yolo12m"}
    assert not {r["model_key"] for r in m4.cells("lockbox3")} & seen


def test_deterministic():
    assert m4.cells("deploy") == m4.cells("deploy")
