# host/tests/test_package_dataset.py: Phase 4 packager, on synthetic cells in a temp dir
import csv
import gzip
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import package_dataset as pkg
from package_dataset import (INDEX_COLUMNS, GLOSSARY, PackageError, Source, error_line, package,
                             select)

PROBE_DOC = """| # | Probe | Command(s) | Question | Known so far |
|---|---|---|---|---|
| 1 | YOLO m-model latency | `./lab run x` | fits? | m-models took 80 s |
| 2 | FP16 viability | `./lab run y` | float16? | executes in float16, slower |
"""


def _trace(nrows=1000, cov=1.0, counters=True, nan_at=None, groups=170):
    """100 baseline_A rows, the rest `forward` except the last 100 baseline_B; `cov` of the forward
    rows have counter data."""
    phase = ["baseline_A"] * 100 + ["forward"] * (nrows - 200) + ["baseline_B"] * 100
    P = np.full(nrows, 7.5, dtype=np.float32)
    P[:100] = 4.8
    if nan_at is not None:
        P[nan_at] = np.nan
    cols = {"t_s": np.arange(nrows) / 5000.0, "t_pi_ns": np.arange(nrows, dtype=np.int64) * 200_000,
            "P_W": P, "phase": pa.array(phase, pa.large_string()),
            "step": pa.array(["forward" if p == "forward" else "" for p in phase], pa.large_string()),
            "depth": np.zeros(nrows, dtype=np.int16), "ctx_read": np.zeros(nrows, dtype=bool)}
    if counters:
        k = int((nrows - 200) * cov)
        ctr = [None] * 100 + [1e6 if i < k else None for i in range(nrows - 200)] + [None] * 100
        cols["ctr_instructions"] = pa.array(ctr, pa.float64())
        cols["sys_cpu0_kHz"] = pa.array([None, 2.4e6] * (nrows // 2), pa.float64())
    t = pa.table(cols)
    return t, groups


def make_cell(runs, cid, *, family="A", model_key="yolo11s", mhz=2400, repeat=0, dv=2,
              quant="fp32", governor="pinned", holdout="primary-s", cov=1.0, counters=True,
              nrows=1000, nan_at=None, residual=0.2, map_machine="aarch64", map_val=0.41,
              bench=False, soak=None, concurrent=None, serve_fps=None, files=None, threads=4,
              imgsz=640, batch=1, fan="full", with_verify=True, drop=()):
    d = runs / cid
    d.mkdir(parents=True)
    variant = model_key if quant == "fp32" else f"{model_key}_int8s"
    meta = {"task": "image", "model": f"{model_key}.pt", "modes": ["forward"], "imgsz": imgsz,
            "batch": batch, "frames_per_mode": 30, "bench_skipped": not bench,
            "bench_forced_off": None, "bench_forced_off_by_soak": False, "variant": variant,
            "quant": quant, "holdout": holdout, "dataset_version": dv, "governor": governor,
            "fan_mode": fan, "threads": threads, "serve_requests": 10, "serve_gap_ms": 200.0,
            "serve_fps": serve_fps, "soak_minutes": soak, "concurrent": concurrent,
            "inputs": {"source": "repeat", "files": files or [{"path": "bus.jpg", "sha256": "x"}]},
            "baseline_W": 4.86, "max_temp_C": 57.3, "throttled_start": 0, "throttled_end": 0,
            "kernel": "6.18.60+rpt-rpi-2712", "torch_version": "2.14.0",
            "recorder_status_start": {"vout_V": 5.4},
            "latency_ms_per_image": {"mean": 206.2},
            "perf_counters": {"enabled": dv == 2, "paranoid": 2, "scope": "user", "interval_ms": 100,
                              "events": ["cycles", "instructions"], "min_pct_running": 100.0}}
    (d / "run_meta.json").write_text(json.dumps(meta))
    (d / "alignment.json").write_text(json.dumps({
        "pass": True, "max_residual_ms": residual, "drift_ppm": -1.5, "slip_rate_hz": 5000.0,
        "slip_warning": False, "slip_samples_total": 0, "slips": [], "drift_warning": False}))
    (d / "derived.json").write_text(json.dumps({
        "energy_per_frame_mJ": 1954.5, "dyn_energy_per_frame_mJ": 946.4,
        "latency_ms_per_frame": 206.3, "batch": batch,
        "map_reference": {"map50_95": map_val, "map50": 0.56, "measured_on": {"machine": map_machine}}}))
    (d / "phase_energy.csv").write_text("phase,iter,step,n_samples,n_finite,mean_P_W,E_mJ\n"
                                        "forward,1,forward,500,500,7.5,750.0\n"
                                        "baseline_A,,,100,100,4.8,96.0\n")
    for f in ("layer_table.csv", "op_table.csv", "perop_power.csv", "context.csv", "sysfs.csv",
              "counters.csv"):
        if f != "counters.csv" or counters:
            (d / f).write_text("a,b\n1,2\n")
    (d / "events.csv").write_text("t_ns,kind\n" + "".join(f"{i},marker\n" for i in range(500)))
    t, groups = _trace(nrows, cov, counters, nan_at)
    pq.write_table(t, d / "power_trace.parquet", row_group_size=groups)       # snappy, odd groups
    (runs / f"{cid}.log").write_text("console\n")
    for f in drop:
        (d / f).unlink()
    row = {"cell_id": cid, "family": family, "model_key": model_key, "cpu_mhz": mhz,
           "repeat": repeat, "round": repeat, "holdout": holdout, "args": []}
    res = {"cell_id": cid, "model_key": model_key, "cpu_mhz": mhz, "repeat": repeat, "round": repeat,
           "attempt": 1, "started": "2026-11-21T12:51:26", "status": "ok", "stage": None,
           "error": None, "git_pi": "abc1234"}
    if with_verify:
        res["verify"] = {"mean_P_W": {"forward": 7.5}, "step_coverage": 0.99, "energy_total_J": 12.5,
                         "warnings": ["mAP reference not measured on the Pi (x@640)"]}
    return row, res


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def world(tmp_path):
    """3 manifest cells (a ok, b fail->ok, c ok->fail), 1 failed (d), 1 never run (e)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    ra, a = make_cell(runs, "a_s_640", family="A")
    rb, b = make_cell(runs, "b_int8", family="B", model_key="yolo11n", quant="int8-static",
                      holdout=None)
    rc, c = make_cell(runs, "c_dvfs", family="D", model_key="yolo11n", governor="ondemand",
                      holdout=None)
    rd = {"cell_id": "d_fail", "family": "C", "model_key": "yolo11n", "cpu_mhz": 2400, "repeat": 0,
          "round": 0, "holdout": None, "args": []}
    re_ = {"cell_id": "e_never", "family": "soak", "model_key": "yolo11n", "cpu_mhz": 2400,
           "repeat": 0, "round": 0, "holdout": None, "args": []}
    fail = lambda cid, stage, err: {"cell_id": cid, "status": "fail", "stage": stage, "error": err,
                                    "attempt": 1}
    man = [ra, rb, rc, rd, re_]
    # c_dvfs: ok first and failed last -> excluded; its files stay on disk
    res = [fail("b_int8", "align", "boom"), a, c, fail("c_dvfs", "verify", "ERROR: low"),
           fail("d_fail", "run", "l1\n12:00 ERROR   drift 0.7W\nFAILED (see run.log)"),
           fail("d_fail", "run", "second\nattempt failed"), b]
    write_jsonl(runs / "manifest.jsonl", man)
    write_jsonl(runs / "results.jsonl", res)
    return runs, man, res


def src(runs):
    return Source(runs, pkg.load_jsonl(runs / "manifest.jsonl"), pkg.load_jsonl(runs / "results.jsonl"))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---- selection ---------------------------------------------------------------------------------
def test_last_row_wins(world):
    runs, _, _ = world
    kept, excluded, orphans = select([src(runs)])
    assert [c.cell_id for c in kept] == ["a_s_640", "b_int8"]        # b: fail then ok -> kept
    assert {e["cell_id"] for e in excluded} == {"c_dvfs", "d_fail", "e_never"}   # c: ok then fail
    assert orphans == []
    assert next(c for c in kept if c.cell_id == "b_int8").attempts == 2


def test_excluded_csv(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True)
    rows = {r["cell_id"]: r for r in read_csv(out / "excluded.csv")}
    assert list(read_csv(out / "excluded.csv")[0]) == ["cell_id", "family", "last_stage", "last_error",
                                                        "attempts"]
    assert rows["d_fail"]["last_stage"] == "run" and rows["d_fail"]["attempts"] == "2"
    assert rows["d_fail"]["last_error"] == "attempt failed"
    assert rows["c_dvfs"]["last_stage"] == "verify" and rows["c_dvfs"]["attempts"] == "2"
    assert rows["e_never"] == {"cell_id": "e_never", "family": "soak", "last_stage": "not_run",
                               "last_error": "", "attempts": "0"}


def test_error_line():
    assert error_line("a\n12:00 ERROR   drift 0.7W > 0.5W\nACCEPTANCE FAILED") == "12:00 ERROR drift 0.7W > 0.5W"
    assert error_line("a\nValueError: bad\nz") == "ValueError: bad"
    assert error_line("first\nlast line  here\n\n") == "last line here"
    assert error_line(None) == "" and error_line("x" * 500) == "x" * 300


def test_truncated_last_results_line_is_ignored(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"cell_id": "a", "status": "ok"}\n{"cell_id": "b", "sta')
    assert [r["cell_id"] for r in pkg.load_jsonl(p)] == ["a"]
    p.write_text('{"cell_id": "a"\n{"cell_id": "b"}\n')
    with pytest.raises(PackageError):
        pkg.load_jsonl(p)


def test_orphans_and_duplicate_ids(world, tmp_path):
    runs, _, _ = world
    s = src(runs)
    s.results.append({"cell_id": "zzz", "status": "ok"})
    assert select([s])[2] == ["zzz"]
    with pytest.raises(PackageError, match="twice"):
        select([s, Source(tmp_path, [{"cell_id": "a_s_640", "family": "A"}], [])])


def test_extra_runs_second_folder(world, tmp_path):
    runs, _, _ = world
    runs2 = tmp_path / "runs2"
    runs2.mkdir()
    r, res = make_cell(runs2, "x_ref", family="ref", model_key="yolov8n", dv=1, holdout=None,
                       counters=False)
    write_jsonl(runs2 / "manifest.jsonl", [r])
    write_jsonl(runs2 / "results.jsonl", [res])
    out = tmp_path / "out"
    srcs = pkg.build_sources(runs, runs / "manifest.jsonl", None, [runs2])
    package(srcs, out, "n")
    assert (out / "cells" / "x_ref" / "power_trace.parquet").is_file()
    assert (out / "cells" / "a_s_640" / "power_trace.parquet").is_file()
    assert len(read_csv(out / "dataset_index.csv")) == 3


# ---- repack and verification -------------------------------------------------------------------
def test_package_copies_and_repacks(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    r = package([src(runs)], out, "n")
    assert r["kept"] == 2
    cell = out / "cells" / "a_s_640"
    for f in ("run_meta.json", "alignment.json", "phase_energy.csv", "derived.json", "layer_table.csv",
              "op_table.csv", "perop_power.csv", "context.csv", "sysfs.csv", "counters.csv",
              "power_trace.parquet", "events.csv.gz", "pi_console.log"):
        assert (cell / f).is_file(), f
    assert not (cell / "thermal_abort.json").exists() and not (cell / "events.csv").exists()
    assert not list((out / "cells").glob(".*partial"))
    assert gzip.open(cell / "events.csv.gz").read() == (runs / "a_s_640" / "events.csv").read_bytes()
    pf = pq.ParquetFile(cell / "power_trace.parquet")
    assert pf.metadata.row_group(0).column(0).compression == "ZSTD"
    assert pq.read_table(cell / "power_trace.parquet").equals(
        pq.read_table(runs / "a_s_640" / "power_trace.parquet"))
    for rr in (out / "readiness.md").read_text().splitlines():
        if "Trace repack verified" in rr:
            assert rr.startswith("- [x]")


def test_repack_multi_slice_and_nan(tmp_path):
    t, _ = _trace(1000, nan_at=500)
    src_p, dst = tmp_path / "s.parquet", tmp_path / "d.parquet"
    pq.write_table(t, src_p, row_group_size=170)
    pkg.repack_parquet(src_p, dst, rows=300)
    assert pq.ParquetFile(dst).metadata.num_row_groups == 4        # 300 + 300 + 300 + 100
    assert pkg.verify_parquet(src_p, dst, rows=300) == 1000
    assert pkg.verify_parquet(src_p, dst, rows=999) == 1000        # a different slicing also agrees


def test_repack_with_huge_row_group_fallback(tmp_path, monkeypatch):
    t, _ = _trace(1000, nan_at=3)
    src_p, dst = tmp_path / "s.parquet", tmp_path / "d.parquet"
    pq.write_table(t, src_p, row_group_size=1000)            # one big row group
    monkeypatch.setattr(pkg, "MAX_GROUP_ROWS", 100)          # -> iter_batches path
    pkg.repack_parquet(src_p, dst, rows=300)
    assert pkg.verify_parquet(src_p, dst, rows=300) == 1000
    assert pkg.trace_stats(src_p, ["forward"])["counter_coverage"] == 1.0


def _rewrite(path, fn, **kw):
    t = fn(pq.read_table(path))
    pq.write_table(t, path, compression=kw.get("compression", "zstd"))


def test_verify_parquet_catches_value_schema_rows_codec(tmp_path):
    t, _ = _trace(1000)
    s = tmp_path / "s.parquet"
    pq.write_table(t, s)

    def fresh(name):
        d = tmp_path / name
        pkg.repack_parquet(s, d)
        return d
    d = fresh("ok.parquet")
    assert pkg.verify_parquet(s, d) == 1000

    d = fresh("val.parquet")                        # one changed float, only in the last slice
    def bump(tab):
        P = tab["P_W"].to_numpy().copy()
        P[-1] += 0.001
        return tab.set_column(tab.schema.get_field_index("P_W"), "P_W", pa.array(P))
    _rewrite(d, bump)
    with pytest.raises(PackageError, match="values differ"):
        pkg.verify_parquet(s, d, rows=300)

    d = fresh("nan.parquet")                        # a NaN turned into a number
    t2, _ = _trace(1000, nan_at=10)
    pq.write_table(t2, s)
    pkg.repack_parquet(s, d)
    def unnan(tab):
        P = tab["P_W"].to_numpy().copy()
        P[10] = 0.0
        return tab.set_column(tab.schema.get_field_index("P_W"), "P_W", pa.array(P))
    _rewrite(d, unnan)
    with pytest.raises(PackageError, match="values differ"):
        pkg.verify_parquet(s, d)
    pq.write_table(t, s)

    d = fresh("schema.parquet")
    _rewrite(d, lambda tab: tab.drop(["depth"]))
    with pytest.raises(PackageError, match="schema"):
        pkg.verify_parquet(s, d)

    d = fresh("rows.parquet")
    _rewrite(d, lambda tab: tab.slice(0, 999))
    with pytest.raises(PackageError, match="rows"):
        pkg.verify_parquet(s, d)

    d = fresh("codec.parquet")
    _rewrite(d, lambda tab: tab, compression="snappy")
    with pytest.raises(PackageError, match="ZSTD"):
        pkg.verify_parquet(s, d)


def test_verify_gzip(tmp_path):
    s, d = tmp_path / "e.csv", tmp_path / "e.csv.gz"
    s.write_bytes(b"a,b\n" * 1000)
    pkg.gzip_events(s, d)
    assert pkg.verify_gzip(s, d) == 4000
    pkg.gzip_events(s, tmp_path / "e2.csv.gz")                          # reproducible bytes
    assert d.read_bytes() == (tmp_path / "e2.csv.gz").read_bytes()
    d.write_bytes(gzip.compress(b"a,b\n" * 999))
    with pytest.raises(PackageError, match="differs"):
        pkg.verify_gzip(s, d)
    d.write_bytes(gzip.compress(b"a,b\n" * 1000)[:-8])                  # truncated
    with pytest.raises(PackageError):
        pkg.verify_gzip(s, d)


def test_forced_parquet_mismatch_fails_loudly_and_leaves_nothing(world, tmp_path, monkeypatch):
    runs, _, _ = world
    real = pkg.repack_parquet

    def corrupt(src_p, dst, rows=pkg.ROW_SLICE):
        real(src_p, dst, rows)
        _rewrite(dst, lambda tab: tab.set_column(
            tab.schema.get_field_index("P_W"), "P_W", pa.array(np.zeros(tab.num_rows, np.float32))))
    monkeypatch.setattr(pkg, "repack_parquet", corrupt)
    out = tmp_path / "out"
    with pytest.raises(PackageError, match="values differ"):
        package([src(runs)], out, "n")
    assert not (out / "cells" / "a_s_640").exists()
    assert not list((out / "cells").glob(".*partial"))
    assert not (out / "dataset_index.csv").exists()


def test_forced_gzip_mismatch(world, tmp_path, monkeypatch):
    runs, _, _ = world
    monkeypatch.setattr(pkg, "gzip_events", lambda s, d: open(d, "wb").write(gzip.compress(b"other")))
    with pytest.raises(PackageError, match="differs"):
        package([src(runs)], tmp_path / "out", "n")


def test_missing_required_file_is_an_error(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    r, res = make_cell(runs, "a", drop=("derived.json",))
    write_jsonl(runs / "manifest.jsonl", [r])
    write_jsonl(runs / "results.jsonl", [res])
    with pytest.raises(PackageError, match="derived.json"):
        package([src(runs)], tmp_path / "out", "n", dry_run=True)


def test_thermal_abort_json_copied_when_present(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    r, res = make_cell(runs, "a")
    (runs / "a" / "thermal_abort.json").write_text("{}")
    write_jsonl(runs / "manifest.jsonl", [r])
    write_jsonl(runs / "results.jsonl", [res])
    package([src(runs)], tmp_path / "out", "n")
    assert (tmp_path / "out" / "cells" / "a" / "thermal_abort.json").is_file()


# ---- index -------------------------------------------------------------------------------------
def test_index_columns_and_values(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True)
    rows = read_csv(out / "dataset_index.csv")
    assert list(rows[0]) == INDEX_COLUMNS
    for c in ("cell_id", "model_key", "task", "model", "cpu_mhz", "repeat", "round", "modes",
              "frames_per_mode", "imgsz", "batch", "started", "git_commit", "n_samples", "trace_s",
              "baseline_W", "mean_P_W_forward", "energy_total_J", "step_coverage", "max_residual_ms",
              "drift_ppm", "slip_warning", "slip_samples_total", "n_slips", "drift_warning",
              "throttled_start", "throttled_end", "family", "dataset_version", "variant", "quant",
              "holdout", "threads", "governor", "fan_mode", "input_source", "serve_fps",
              "soak_minutes", "concurrent_mode", "energy_per_frame_mJ", "dyn_energy_per_frame_mJ",
              "latency_ms_per_frame", "latency_ms_per_image", "map50_95", "map50",
              "map_measured_on_machine", "counter_coverage", "max_temp_C", "baseline_drift_exempt",
              "bench"):
        assert c in INDEX_COLUMNS
    a, b = rows
    assert a["cell_id"] == "a_s_640" and b["cell_id"] == "b_int8"          # manifest order
    assert (a["family"], a["model_key"], a["holdout"], a["quant"], a["variant"]) == \
        ("A", "yolo11s", "primary-s", "fp32", "yolo11s")
    assert (b["family"], b["holdout"], b["quant"], b["variant"]) == \
        ("B", "none", "int8-static", "yolo11n_int8s")
    assert a["n_samples"] == "1000" and a["trace_s"] == "0.2"
    assert a["git_commit"] == "abc1234" and a["started"] == "2026-11-21T12:51:26"
    assert a["mean_P_W_forward"] == "7.5" and a["energy_total_J"] == "12.5" and a["step_coverage"] == "0.99"
    assert a["energy_per_frame_mJ"] == "1954.5" and a["latency_ms_per_frame"] == "206.3"
    assert a["latency_ms_per_image"] == "206.2" and a["map50_95"] == "0.41"
    assert a["map_measured_on_machine"] == "aarch64" and a["counter_coverage"] == "1.0"
    assert a["slip_warning"] == "False" and a["n_slips"] == "0" and a["bench"] == "False"
    assert a["max_residual_ms"] == "0.2" and a["drift_ppm"] == "-1.5"
    assert a["serve_fps"] == "" and a["soak_minutes"] == "" and a["concurrent_mode"] == ""
    assert a["baseline_drift_exempt"] == ""


def test_index_v1_shaped_cell_and_odd_cells(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = [make_cell(runs, "ref", family="ref", model_key="yolov8n", dv=1, holdout=None,
                       counters=False, with_verify=False),
             make_cell(runs, "partial", cov=0.9, bench=True, soak=30, serve_fps=5,
                       concurrent={"mode": "spinner"}, map_machine="x86_64",
                       files=[{"path": f"s/{i}.jpg"} for i in range(50)]),
             make_cell(runs, "one_img", files=[{"path": "inputs/empty.jpg"}])]
    write_jsonl(runs / "manifest.jsonl", [c[0] for c in cells])
    write_jsonl(runs / "results.jsonl", [c[1] for c in cells])
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True)
    ref, part, one = read_csv(out / "dataset_index.csv")
    assert ref["holdout"] == "" and ref["counter_coverage"] == "" and ref["dataset_version"] == "1"
    # no verify summary in results.jsonl -> energy and forward power come from phase_energy.csv
    assert ref["mean_P_W_forward"] == "7.5" and ref["energy_total_J"] == "0.846"
    assert part["counter_coverage"] == "0.9" and part["bench"] == "True"
    assert part["soak_minutes"] == "30" and part["serve_fps"] == "5" and part["concurrent_mode"] == "spinner"
    assert part["input_image"] == "50 files" and part["map_measured_on_machine"] == "x86_64"
    assert one["input_image"] == "empty.jpg"


def test_glossary_covers_every_index_column():
    assert set(GLOSSARY) == set(INDEX_COLUMNS)
    assert len(set(INDEX_COLUMNS)) == len(INDEX_COLUMNS)


# ---- coverage ----------------------------------------------------------------------------------
def test_coverage_counts(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = [make_cell(runs, "m1", model_key="yolo11s", imgsz=320),
             make_cell(runs, "m2", model_key="yolo11s", imgsz=640, repeat=1),
             make_cell(runs, "m3", model_key="yolo11n", holdout=None, governor="ondemand", fan="auto"),
             make_cell(runs, "m4", model_key="yolo11n", holdout=None, quant="int8-static", batch=4,
                       serve_fps=5, threads=1, mhz=1500)]
    write_jsonl(runs / "manifest.jsonl", [c[0] for c in cells])
    write_jsonl(runs / "results.jsonl", [c[1] for c in cells])
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True)
    cov = {(r["dimension"], r["value"]): int(r["n_cells"]) for r in read_csv(out / "coverage.csv")}
    assert cov[("model", "yolo11s")] == 2 and cov[("model", "yolo11n")] == 2
    assert cov[("imgsz", "320")] == 1 and cov[("imgsz", "640")] == 3
    assert cov[("batch", "1")] == 3 and cov[("batch", "4")] == 1
    assert cov[("threads", "4")] == 3 and cov[("threads", "1")] == 1
    assert cov[("quant", "fp32")] == 3 and cov[("quant", "int8-static")] == 1
    assert cov[("variant", "yolo11n_int8s")] == 1
    assert cov[("clock_mhz_or_dvfs", "dvfs")] == 1 and cov[("clock_mhz_or_dvfs", "2400")] == 2
    assert cov[("clock_mhz_or_dvfs", "1500")] == 1
    assert cov[("fan_mode", "auto")] == 1 and cov[("fan_mode", "full")] == 3
    assert cov[("serve", "5 requests/s")] == 1 and cov[("serve", "10 requests, 200 ms gaps")] == 3
    assert cov[("holdout", "primary-s")] == 2 and cov[("holdout", "none")] == 2
    assert cov[("concurrent", "none")] == 4 and cov[("input_image", "bus.jpg")] == 4
    for dim in pkg.COVERAGE_DIMENSIONS:                          # every dimension sums to the cells
        assert sum(v for (d, _), v in cov.items() if d == dim) == 4, dim


# ---- readiness, README, dry run ----------------------------------------------------------------
def _good_world(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = [make_cell(runs, "s1", model_key="yolo11s"),
             make_cell(runs, "q1", model_key="yolo11n", holdout=None, quant="int8-static"),
             make_cell(runs, "d1", model_key="yolo11n", holdout=None, governor="ondemand")]
    write_jsonl(runs / "manifest.jsonl", [c[0] for c in cells])
    write_jsonl(runs / "results.jsonl", [c[1] for c in cells])
    doc = tmp_path / "dataset-v2.md"
    doc.write_text(PROBE_DOC)
    return runs, doc


def test_readiness_all_green(tmp_path):
    runs, doc = _good_world(tmp_path)
    out = tmp_path / "out"
    r = package([src(runs)], out, "n", docs=doc)
    text = (out / "readiness.md").read_text()
    assert "[ ]" not in text, text
    assert r["todos"]


def test_readiness_flags_low_coverage_residual_and_missing_map(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = [make_cell(runs, "bad_cov", cov=0.9),
             make_cell(runs, "slow", residual=1.2, governor="ondemand"),
             make_cell(runs, "nomap", quant="int8-static", map_val=None, map_machine=None)]
    write_jsonl(runs / "manifest.jsonl", [c[0] for c in cells])
    write_jsonl(runs / "results.jsonl", [c[1] for c in cells])
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True, docs=tmp_path / "missing.md")
    lines = {ln.split("**")[1]: ln for ln in (out / "readiness.md").read_text().splitlines()
             if ln.startswith("- [")}
    k = [x for x in lines if x.startswith("Counter coverage")][0]
    assert lines[k].startswith("- [ ]") and "worst coverage 0.9000 (bad_cov)" in lines[k]
    k = [x for x in lines if x.startswith("Alignment")][0]
    assert lines[k].startswith("- [ ]") and "1.2000 ms (slow)" in lines[k]
    k = [x for x in lines if x.startswith("Every quantized")][0]
    assert lines[k].startswith("- [ ]") and "nomap" in lines[k]
    k = [x for x in lines if x.startswith("Probe")][0]
    assert lines[k].startswith("- [ ]") and "not found" in lines[k]
    k = [x for x in lines if x.startswith("Trace repack")][0]
    assert lines[k].startswith("- [ ]") and "dry run" in lines[k]


def test_readiness_flags_wrong_holdout_and_open_probe(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = [make_cell(runs, "w", model_key="yolo11m", holdout="primary-s")]    # m must be secondary-m
    write_jsonl(runs / "manifest.jsonl", [c[0] for c in cells])
    write_jsonl(runs / "results.jsonl", [c[1] for c in cells])
    doc = tmp_path / "d.md"
    doc.write_text(PROBE_DOC + "| 3 | x | y | z | not run |\n")
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True, docs=doc)
    text = (out / "readiness.md").read_text()
    assert "- [ ] **Holdout recorded for every cell.** 1 cells with a missing or wrong holdout, e.g. w" in text
    assert "2 of 3 probes" in text and "still open: 3 (x)" in text


def test_readiness_excluded_without_reason_is_open(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    package([src(runs)], out, "n", dry_run=True)
    text = (out / "readiness.md").read_text()
    assert "- [ ] **All cells ok, or explicitly excluded with a reason.** 2 kept + 3 excluded = 5 of 5" in text
    assert "1 never ran" in text and "not_run 1" in text


def test_dry_run_copies_no_trace(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    package([src(runs)], out, "monsoon_v2_x", dry_run=True)
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "coverage.csv", "dataset_index.csv", "excluded.csv", "manifest.jsonl",
        "readiness.md", "results.jsonl"]
    assert not list(out.rglob("*.parquet")) and not list(out.rglob("*.gz"))
    assert "TODO(controller): size on disk" in (out / "README.md").read_text()


def test_readme_contents(world, tmp_path):
    runs, _, _ = world
    out = tmp_path / "out"
    package([src(runs)], out, "monsoon_v2_x")
    text = (out / "README.md").read_text()
    for needle in ("monsoon_v2_x", "v1 versus v2: the counter split", "5.40 V", "6.18.60",
                   "perf_event_paranoid` = 2", "user-space", "repeat 0", "GIT_LFS_SKIP_SMUDGE=1",
                   "git lfs pull --include", "INT8 dynamic", "Pruning", "`primary-s`", "`secondary-m`",
                   "| `A` |", "1 / 1", "| `soak` |", "0 / 1", "TODO(controller)"):
        assert needle in text, needle
    assert "**2** of 5 planned" in text and "MB" in text
    for c in INDEX_COLUMNS:
        assert f"| `{c}` |" in text


def test_main_cli(world, tmp_path, capsys):
    runs, _, _ = world
    out = tmp_path / "out"
    rc = pkg.main(["--runs", str(runs), "--manifest", str(runs / "manifest.jsonl"), "--out", str(out),
                   "--name", "n", "--dry-run", "--docs", str(tmp_path / "none.md")])
    assert rc == 0
    cap = capsys.readouterr().out
    assert "2 cells kept, 3 excluded" in cap and "(dry run)" in cap
    rc = pkg.main(["--runs", str(runs), "--manifest", str(runs / "nope.jsonl"), "--out", str(out),
                   "--name", "n"])
    assert rc == 1
