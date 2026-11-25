import gzip
import json
import math
import subprocess

import pandas as pd
import pytest

import data
import runlog
import settings
from conftest import make_cell

CATALOGUE_COLS = [
    "cell_id", "source", "path", "task", "model_key", "base_model", "variant", "precision", "quant",
    "n_quantized_modules", "imgsz", "batch", "threads", "cpu_mhz", "governor", "fan_mode",
    "serve_fps", "soak_minutes", "concurrent_mode", "input_source", "family", "repeat", "rig",
    "dataset_version", "holdout", "has_counters", "mode_a_eligible", "mode_l_eligible", "layer_hooks",
]


def test_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    assert settings.data_root() == tmp_path
    d = settings.derived_dir()
    assert d == tmp_path / "derived" and d.is_dir()
    assert settings.seed() == 20261122 == settings.cfg()["seed"]
    monkeypatch.delenv("ANALYTICS_DATA")
    assert settings.data_root().name == "pi-power-data"


def test_vision_models():
    assert data.VISION_MODELS == ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m",
                                  "effb0", "mbv3", "resnet18",
                                  "yolo12n", "yolo12s", "swint", "vitb16", "convnextt", "maxvitt"]
    assert data.HOLDOUT_SOURCES == ("lockbox", "lockbox2", "lockbox3")


def test_cell_dirs_skips_missing_sources(tmp_path):
    make_cell(tmp_path, "v2", "b_cell")
    make_cell(tmp_path, "v1", "a_cell")
    dirs = data.cell_dirs(tmp_path)  # no lockbox folder
    assert [p.name for p in dirs] == ["a_cell", "b_cell"]  # v1 before v2
    assert data.cell_dirs(tmp_path / "nowhere") == []


def test_catalogue_v1_and_v2_defaults(tmp_path):
    make_cell(tmp_path, "v1", "mbv3_2400_r00", meta={"model": "torchvision:mobilenet_v3_small",
              "imgsz": 224}, index={"model_key": "mbv3"})
    make_cell(tmp_path, "v2", "yolo11n_640_t4_2400_r00",
              files={"counters.csv": "t_start_ns,t_end_ns\n0,1\n"})
    df = data.catalogue(tmp_path)
    assert list(df.columns) == CATALOGUE_COLS
    assert len(df) == 2
    v1 = df[df.source == "v1"].iloc[0]
    assert (v1.family, v1.governor, v1.fan_mode, v1.threads, v1.batch) == ("v1", "pinned", "full", 4, 1)
    assert (v1.quant, v1.precision, v1.n_quantized_modules, v1.dataset_version) == ("fp32", "fp32", 0, 1)
    assert (v1.rig, v1.has_counters, v1.mode_l_eligible) == ("oct2026", False, False)
    assert v1.model_key == v1.base_model == v1.variant == "mbv3"
    assert v1.cpu_mhz == 2400 and math.isnan(v1.serve_fps) and v1.concurrent_mode is None
    assert v1.mode_a_eligible
    v2 = df[df.source == "v2"].iloc[0]
    assert (v2.rig, v2.dataset_version, v2.family, v2.has_counters) == ("nov2026", 2, "A", True)
    assert v2.mode_l_eligible and v2.mode_a_eligible
    assert v2.path == str(tmp_path / "v2" / "cells" / "yolo11n_640_t4_2400_r00")


def test_counters_need_file_and_enabled_flag(tmp_path):
    pc_off = {"enabled": False, "interval_ms": 100}
    make_cell(tmp_path, "v2", "ref_off", files={"counters.csv": "t_start_ns\n"}, meta={"perf_counters": pc_off})
    make_cell(tmp_path, "v2", "ref_nofile")
    make_cell(tmp_path, "v2", "ok", files={"counters.csv": "t_start_ns\n"})
    df = data.catalogue(tmp_path).set_index("cell_id")
    assert not df.loc["ref_off", "has_counters"]
    assert not df.loc["ref_nofile", "has_counters"]
    assert df.loc["ok", "has_counters"]


@pytest.mark.parametrize("meta", [
    {"concurrent": {"mode": "model", "model": "yolo11n.pt"}},
    {"serve_fps": 5.0},
    {"soak_minutes": 30.0},
    {"governor": "ondemand"},
    {"fan_mode": "auto"},
    {"fan_mode": "off"},
    {"task": "lm"},
])
def test_not_mode_a_eligible(tmp_path, meta):
    make_cell(tmp_path, "v2", "x", meta=meta)
    row = data.catalogue(tmp_path).iloc[0]
    assert not row.mode_a_eligible


def test_non_vision_model_not_mode_a(tmp_path):
    make_cell(tmp_path, "v1", "lm16_2400_r00", meta={"task": "lm"}, index={"model_key": "lm16"})
    make_cell(tmp_path, "v1", "weird", index={"model_key": "other"})
    assert not data.catalogue(tmp_path).mode_a_eligible.any()


def test_concurrent_mode_and_serve_values(tmp_path):
    make_cell(tmp_path, "v2", "c", meta={"concurrent": {"mode": "spinner"}, "serve_fps": 15.0,
                                         "soak_minutes": 30.0})
    r = data.catalogue(tmp_path).iloc[0]
    assert (r.concurrent_mode, r.serve_fps, r.soak_minutes) == ("spinner", 15.0, 30.0)


def test_int8_cell(tmp_path):
    make_cell(tmp_path, "v2", "yolo11n_int8s_640_t4_1500_r00",
              meta={"variant": "yolo11n_int8s", "quant": "int8-static", "executed_dtype": "float32+quint8",
                    "quantized_modules": 57, "cpu_mhz_pinned": 1500},
              index={"model_key": "yolo11n", "cpu_mhz": 1500})
    r = data.catalogue(tmp_path).iloc[0]
    assert (r.base_model, r.precision, r.variant, r.n_quantized_modules) == ("yolo11n", "int8", "yolo11n_int8s", 57)
    assert r.mode_a_eligible and r.cpu_mhz == 1500


def test_torchvision_variant_of_an_fp32_cell_is_normalised_to_the_model_key_and_real_variants_are_kept(tmp_path):
    tv = "torchvision:mobilenet_v3_small"
    make_cell(tmp_path, "lockbox", "tv_equal", meta={"model": tv, "variant": tv, "quant": "fp32"}, index={"model_key": "mbv3"})
    make_cell(tmp_path, "lockbox", "tv_prefix", meta={"model": "mbv3.pt", "variant": "torchvision:other", "quant": "fp32"}, index={"model_key": "mbv3"})
    make_cell(tmp_path, "v2", "same_as_model", meta={"model": "foo", "variant": "foo", "quant": "fp32"}, index={"model_key": "mbv3"})
    make_cell(tmp_path, "v2", "yolo", meta={"model": "yolo11n.pt", "variant": "yolo11n", "quant": "fp32"}, index={"model_key": "yolo11n"})
    make_cell(tmp_path, "v2", "int8", meta={"model": "yolo11n.pt", "variant": "yolo11n_int8s", "quant": "int8-static"}, index={"model_key": "yolo11n"})
    make_cell(tmp_path, "v2", "fp16", meta={"model": "yolo11n.pt", "variant": "yolo11n_fp16", "quant": "fp16", "executed_dtype": "float16"},
              index={"model_key": "yolo11n"})
    make_cell(tmp_path, "v2", "tv_int8", meta={"model": tv, "variant": tv, "quant": "int8-static"}, index={"model_key": "mbv3"})
    v = data.catalogue(tmp_path).set_index("cell_id")["variant"]
    assert v["tv_equal"] == v["tv_prefix"] == v["same_as_model"] == "mbv3"
    assert v["yolo"] == "yolo11n" and v["int8"] == "yolo11n_int8s" and v["fp16"] == "yolo11n_fp16"
    assert v["tv_int8"] == tv                      # only fp32 cells are normalised


def test_quantized_modules_list_and_precision_fp16(tmp_path):
    make_cell(tmp_path, "v2", "q", meta={"quant": "int8-dynamic", "quantized_modules": ["a", "b"]})
    make_cell(tmp_path, "v2", "h", meta={"quant": "fp16", "executed_dtype": "float16"})
    make_cell(tmp_path, "v2", "h2", meta={"quant": "none", "executed_dtype": "float16"})
    df = data.catalogue(tmp_path).set_index("cell_id")
    assert df.loc["q", "n_quantized_modules"] == 2 and df.loc["q", "precision"] == "int8"
    assert df.loc["h", "precision"] == "fp16" and df.loc["h2", "precision"] == "fp16"


def test_cpu_mhz_prefers_pinned_else_index(tmp_path):
    make_cell(tmp_path, "v2", "a", meta={"cpu_mhz_pinned": 2000}, index={"cpu_mhz": 1500})
    make_cell(tmp_path, "v2", "b", meta={"cpu_mhz_pinned": None}, index={"cpu_mhz": 1500})
    df = data.catalogue(tmp_path).set_index("cell_id")
    assert df.loc["a", "cpu_mhz"] == 2000 and df.loc["b", "cpu_mhz"] == 1500


def test_cell_missing_from_index_is_error(tmp_path):
    cell = make_cell(tmp_path, "v2", "present")
    (tmp_path / "v2" / "cells" / "orphan").mkdir()
    (tmp_path / "v2" / "cells" / "orphan" / "run_meta.json").write_text(
        (cell / "run_meta.json").read_text())
    with pytest.raises(KeyError, match="orphan"):
        data.catalogue(tmp_path)


def test_readers_optional_files(tmp_path):
    cell = make_cell(tmp_path, "v1", "c", files={
        "events.csv.gz": "a,b\n1,2\n3,4\n", "extra.json": {"k": 1},
        "plain.csv": "x\n5\n"})
    assert data.read_csv_opt(cell, "layer_table.csv") is None
    assert data.read_json_opt(cell, "derived.json") is None
    assert data.read_meta(cell)["task"] == "image"
    ev = data.read_csv_opt(cell, "events.csv")  # name given without .gz
    assert list(ev.columns) == ["a", "b"] and len(ev) == 2
    assert len(data.read_csv_opt(cell, "events.csv.gz")) == 2
    assert data.read_csv_opt(cell, "plain.csv").x.tolist() == [5]
    assert data.read_json_opt(cell, "extra.json") == {"k": 1}


def test_read_trace_projection_and_strings(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cell = make_cell(tmp_path, "v1", "t")
    tbl = pa.table({"t_s": [0.0, 0.1], "P_W": [5.0, 6.0],
                    "phase": pa.array(["a", "b"]).dictionary_encode(), "iter": [1, 2]})
    pq.write_table(tbl, cell / "power_trace.parquet")
    df = data.read_trace(cell, ["t_s", "phase"])
    assert list(df.columns) == ["t_s", "phase"]
    assert df.phase.tolist() == ["a", "b"]
    assert not isinstance(df.phase.dtype, pd.CategoricalDtype)


def test_runlog_append(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path / "root"))
    (tmp_path / "root" / "v1").mkdir(parents=True)
    (tmp_path / "root" / "v1" / "dataset_index.csv").write_text("cell_id\nx\n")
    log = tmp_path / "runs.jsonl"
    runlog.append({"step": "qc", "n": 3}, path=log)
    runlog.append({"step": "qc2"}, path=log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    for k in ("run_id", "ts", "commit", "dirty", "data_hash", "machine", "step", "n"):
        assert k in rec
    assert len(rec["run_id"]) == 32 and rec["ts"].endswith("+00:00")
    assert isinstance(rec["dirty"], bool) and len(rec["data_hash"]) == 64
    assert rec["commit"] == subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=data.__file__.rsplit("/", 1)[0]).stdout.strip()
    assert json.loads(lines[1])["run_id"] != rec["run_id"]
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert runlog.file_hash(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_runlog_data_hash_changes_with_index(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    (tmp_path / "v2").mkdir()
    idx = tmp_path / "v2" / "dataset_index.csv"
    idx.write_text("a\n")
    h1 = runlog.data_hash()
    idx.write_text("b\n")
    assert runlog.data_hash() != h1


# ---- real data --------------------------------------------------------------------------------
def test_real_catalogue(real_data):
    df = data.catalogue(real_data)
    assert list(df.columns) == CATALOGUE_COLS
    assert (df.source == "v1").sum() == 224 and (df.source == "v2").sum() == 179
    ref = df[df.cell_id.str.startswith("v1ref_")]
    assert len(ref) == 18
    assert (ref.rig == "nov2026").all() and not ref.has_counters.any()
    assert not df[df.source == "v1"].has_counters.any()
    other_v2 = df[(df.source == "v2") & ~df.cell_id.str.startswith("v1ref_")]
    assert other_v2.has_counters.all() and len(other_v2) == 161
    assert (df[df.source == "v1"].rig == "oct2026").all()
    assert df.cell_id.is_unique
    assert set(df[df.mode_a_eligible].base_model) <= set(data.VISION_MODELS)
    assert not df[df.concurrent_mode.notna()].mode_a_eligible.any()
    assert not df[df.task != "image"].mode_a_eligible.any()
    assert df[df.cell_id == "yolo11n_int8s_640_t4_1500_r00"].mode_a_eligible.all()


def test_cli_catalogue_registered(tmp_path, monkeypatch, capsys):
    import cli
    make_cell(tmp_path, "v1", "a")
    make_cell(tmp_path, "v2", "b")
    monkeypatch.setenv("ANALYTICS_DATA", str(tmp_path))
    monkeypatch.setattr(cli, "REPORTS", tmp_path / "rep")
    assert cli.main(["catalogue"]) == 0
    assert (tmp_path / "rep" / "catalogue.csv").is_file()
    out = capsys.readouterr().out
    assert "v1" in out and "v2" in out
