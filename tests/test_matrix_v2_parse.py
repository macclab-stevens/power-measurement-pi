# tests/test_matrix_v2_parse.py -- every v2 manifest cell must parse with run.py's own parser.
# Lives in the Pi suite because importing run.py needs torch (the EliteBook host env has none);
# host/matrix_v2.py itself is pure stdlib and is imported straight from ../host.
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "host"))
import pytest

import matrix_v2
import run

ALL = dict(include_fp16=True, include_optional=True, include_int8d=True, include_prune=True)


def _cells(which, **kw):
    return matrix_v2.cells(which, **kw)


@pytest.mark.parametrize("which", [*matrix_v2.FAMILIES, "all", "pilot"])
def test_every_cell_parses_with_run_parse_args(which):
    rows = _cells(which, **ALL)
    assert rows
    for r in rows:
        a = run.parse_args(r["args"])
        assert a.sampler == "monsoon"
        if r["family"] == "ref":                                # v1 args: dataset_version 1
            assert (a.holdout, a.perf_counters, run.dataset_version(a)) == (None, "off", 1)
            assert a.fan_full and a.fan is None and a.governor == "pinned"
            continue
        assert a.perf_counters == "on"
        assert a.holdout == (r["holdout"] or "none")            # always set => dataset_version 2
        assert a.cpu_mhz == r["cpu_mhz"] and a.task == "image"
        assert a.governor == ("ondemand" if "_dvfs_" in r["cell_id"] else "pinned")


def test_parsed_values_of_special_cells():
    rows = {r["cell_id"]: run.parse_args(r["args"]) for r in _cells("all", **ALL)}
    a = rows["yolo11n_640_t4_b8_2400_r00"]
    assert (a.batch, a.frames, a.no_bench is False, a.bench) == (8, 5, True, 2.0)
    a = rows["yolo11n_serve5_640_t4_2400_r00"]
    assert (a.serve, a.serve_fps) == (100, 5.0)
    a = rows["yolo11n_stream_640_t4_2400_r00"]
    assert (a.frames_source, a.frames) == ("stream", 50)
    a = rows["yolo11m_soak30_640_t4_2400_r00"]
    assert (a.soak_minutes, a.fan, a.no_bench) == (30.0, "auto", True)
    a = rows["yolo11n_int8s_640_t4_1500_r00"]
    assert (a.quant, a.quant_weights, a.no_bench) == ("int8-static", "models/yolo11n_int8s.pt", True)
    a = rows["yolo11n_p50_640_t4_2400_r00"]
    assert a.prune == "models/yolo11n_p50.pt" and a.quant == "fp32"
    a = rows["yolo11n_crowded_640_t4_2400_r00"]
    assert (a.input_image, a.check_image) == ("inputs/crowded.jpg", "bus.jpg")
    a = rows["yolo11s_960_t4_2400_r00"]
    assert (a.holdout, a.imgsz) == ("primary-s", 960)


@pytest.mark.skipif(not hasattr(run.build_parser().parse_known_args([])[0], "concurrent"),
                    reason="--concurrent lands with Task 13")
def test_concurrency_cells_parse():
    for r in _cells("D", include_concurrent=True):
        run.parse_args(r["args"])
