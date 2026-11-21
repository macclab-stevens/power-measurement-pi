# host/tests/test_matrix_v2.py -- dataset v2 manifest families (pure stdlib; parse tests with
# run.parse_args live in tests/test_matrix_v2_parse.py because run.py imports torch)
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import experiment
import matrix
import matrix_v2 as m

ID_RE = re.compile(
    r"^(pilot_)?(v1ref_yolo(?:v8|11)n_(?:1500|2000|2400)_r\d\d$|(yolo(?:v8|11)[nsm])(_(?:int8d|int8s|fp16|p30|p50))?"
    r"(_(?:fanauto|fanoff|soak30|soak5|serve1|serve5|serve15|stream|empty|crowded|conc2|concspin))?"
    r"_(\d+)_t(\d)(_b\d+)?_(2400|2000|1500|dvfs)_r(\d\d)$)")
def v2_only(rows):
    return [r for r in rows if r["family"] != "ref"]


ALL_FLAGS = dict(include_fp16=True, include_optional=True, include_concurrent=True,
                 include_int8d=True, include_prune=True)


def arg(row, flag):
    a = row["args"]
    return a[a.index(flag) + 1]


def by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def test_family_counts():
    assert len(m.cells("A")) == 44
    assert len(m.cells("A-input")) == 6
    assert len(m.cells("B")) == 12
    assert len(m.cells("B", include_fp16=True)) == 15
    assert len(m.cells("B", include_int8d=True, include_prune=True)) == 27
    assert len(m.cells("B", include_fp16=True, include_int8d=True, include_prune=True)) == 30
    assert len(m.cells("C")) == 60
    assert len(m.cells("C", include_optional=True)) == 66
    assert len(m.cells("D")) == 34                                   # concurrency cells by default
    assert len(m.cells("D", include_concurrent=False)) == 28
    assert len(m.cells("soak")) == 2
    assert len(m.cells("ref")) == 18
    assert len(m.cells("all")) == 44 + 6 + 12 + 60 + 34 + 2 + 18
    assert len(m.cells("all", **ALL_FLAGS)) == 44 + 6 + 30 + 66 + 34 + 2 + 18


@pytest.mark.parametrize("which", [*m.FAMILIES, "all", "pilot"])
def test_ids_unique_and_well_formed(which):
    rows = m.cells(which, **ALL_FLAGS)
    ids = [r["cell_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for i in ids:
        assert ID_RE.match(i), i


def test_naming_examples_from_the_plan():
    ids = {r["cell_id"] for r in m.cells("all", **ALL_FLAGS)}
    for want in ("yolo11n_int8s_640_t4_2400_r00", "yolo11n_640_t4_dvfs_r00",
                 "yolo11n_fanoff_640_t4_2400_r01", "yolo11m_soak30_640_t4_2400_r00",
                 "yolo11s_480_t4_2400_r02", "yolo11n_640_t4_b8_2400_r00",
                 "yolo11n_empty_640_t4_2400_r00", "yolo11n_serve15_640_t4_2400_r02",
                 "yolo11n_p30_640_t4_2400_r00", "yolo11n_int8s_640_t4_1500_r00",
                 "yolov8n_int8d_640_t4_2400_r00", "yolo11n_concspin_640_t4_2400_r00",
                 "yolo11n_960_t1_2400_r00"):
        assert want in ids, want


def test_row_fields_and_arg_consistency():
    for r in v2_only(m.cells("all", **ALL_FLAGS)):
        assert {"cell_id", "family", "model_key", "cpu_mhz", "repeat", "round", "holdout",
                "args"} <= set(r)
        assert r["family"] in m.CONFIG_FAMILIES and r["round"] == r["repeat"]
        assert isinstance(r["cpu_mhz"], int) and arg(r, "--cpu-mhz") == str(r["cpu_mhz"])
        assert arg(r, "--model") == r["model_key"] + ".pt"
        assert arg(r, "--perf-counters") == "on"
        assert arg(r, "--sampler") == "monsoon" and arg(r, "--check-image") == "bus.jpg"
        assert arg(r, "--task") == "image" and arg(r, "--warmup") == "10"
        assert r["repeat"] == int(r["cell_id"][-2:])
        json.dumps(r)                                      # manifest rows are JSON lines


def test_holdout_tags():
    for r in v2_only(m.cells("all", **ALL_FLAGS)):
        mk = r["model_key"]
        want = "primary-s" if mk in ("yolo11s", "yolov8s") else (
            "secondary-m" if mk in ("yolo11m", "yolov8m") else None)
        assert r["holdout"] == want, mk
        assert arg(r, "--holdout") == (want or "none")


def test_dvfs_cells():
    rows = m.cells("D")
    dvfs = [r for r in rows if "_dvfs_" in r["cell_id"]]
    assert Counter(r["model_key"] for r in dvfs) == {"yolo11n": 5, "yolo11s": 5}
    for r in dvfs:
        assert arg(r, "--governor") == "ondemand" and arg(r, "--cpu-mhz") == "2400"
        assert r["cpu_mhz"] == 2400
    for r in rows:
        if r not in dvfs:
            assert "--governor" not in r["args"]
    assert all("dvfs" not in r["cell_id"] for r in m.cells("A") + m.cells("C"))


def test_bench_flags_per_family():
    # Ruling R31: benches only on repeat 0 of each A / A-input / C configuration
    for fam in ("A", "A-input", "C"):
        rows = m.cells(fam)
        assert any(r["repeat"] >= 1 for r in rows)
        for r in rows:
            if r["repeat"] == 0:
                assert arg(r, "--bench") == "2" and "--no-bench" not in r["args"], r["cell_id"]
            else:
                assert "--no-bench" in r["args"] and "--bench" not in r["args"], r["cell_id"]
    for fam in ("B", "D", "soak"):
        for r in m.cells(fam, **ALL_FLAGS):
            assert "--no-bench" in r["args"] and "--bench" not in r["args"]
    for r in m.cells("pilot"):
        if r["family"] in ("A", "A-input", "C"):
            assert r["repeat"] == 0 and arg(r, "--bench") == "2"


def test_bench_count_one_per_configuration():
    for fam in ("A", "A-input", "C"):
        rows = m.cells(fam)
        benched = [r for r in rows if "--bench" in r["args"]]
        assert len(benched) == len({r["cell_id"].rsplit("_r", 1)[0] for r in rows})


def test_family_a_grid():
    rows = m.cells("A")
    assert {r["model_key"] for r in rows} == {"yolov8s", "yolov8m", "yolo11s", "yolo11m"}
    for mk in {r["model_key"] for r in rows}:
        mine = by(rows, model_key=mk)
        assert Counter(r["cpu_mhz"] for r in mine) == {2400: 5, 2000: 3, 1500: 3}
    for r in rows:
        assert (arg(r, "--imgsz"), arg(r, "--threads")) == ("640", "4")
        assert "--batch" not in r["args"] and "--quant" not in r["args"]
        assert arg(r, "--fan") == "full"
        assert arg(r, "--frames") == ("20" if r["model_key"].endswith("m") else "30")
        assert (arg(r, "--serve"), arg(r, "--serve-gap-ms")) == ("10", "200")


def test_a_input():
    rows = m.cells("A-input")
    assert {r["model_key"] for r in rows} == {"yolo11n"}
    for cond in ("empty", "crowded"):
        mine = [r for r in rows if f"_{cond}_" in r["cell_id"]]
        assert len(mine) == 3
        assert all(arg(r, "--input-image") == f"inputs/{cond}.jpg" and r["cpu_mhz"] == 2400
                   for r in mine)


def test_b_variants_and_args():
    rows = m.cells("B", **ALL_FLAGS)
    s = [r for r in rows if "_int8s_" in r["cell_id"]]
    assert Counter((r["model_key"], r["cpu_mhz"]) for r in s) == {
        ("yolo11n", 2400): 3, ("yolo11s", 2400): 3, ("yolo11n", 2000): 3, ("yolo11n", 1500): 3}
    for r in s:
        assert arg(r, "--quant") == "int8-static"
        assert arg(r, "--quant-weights") == f"models/{r['model_key']}_int8s.pt"
    d = [r for r in rows if "_int8d_" in r["cell_id"]]
    assert Counter(r["model_key"] for r in d) == {"yolo11n": 3, "yolo11s": 3, "yolov8n": 3}
    assert all(arg(r, "--quant") == "int8-dynamic" for r in d)
    f = [r for r in rows if "_fp16_" in r["cell_id"]]
    assert len(f) == 3 and all(arg(r, "--quant") == "fp16" for r in f)
    p = [r for r in rows if "_p30_" in r["cell_id"] or "_p50_" in r["cell_id"]]
    assert len(p) == 6
    for r in p:
        pp = re.search(r"_p(30|50)_", r["cell_id"]).group(1)
        assert arg(r, "--prune") == f"models/yolo11n_p{pp}.pt" and "--quant" not in r["args"]


def test_b_default_is_int8_static_only():
    rows = m.cells("B")
    assert all("_int8s_" in r["cell_id"] for r in rows)
    assert not any(x in " ".join(r["args"]) for r in rows for x in ("int8-dynamic", "--prune", "fp16"))


def test_c_grid():
    rows = m.cells("C", include_optional=True)
    key = lambda r: (r["model_key"], arg(r, "--imgsz"), arg(r, "--threads"),  # noqa: E731
                     r["args"][r["args"].index("--batch") + 1] if "--batch" in r["args"] else "1")
    c = Counter(key(r) for r in rows)
    assert set(c.values()) == {3} and len(c) == 22
    for s in ("320", "480", "960"):
        for mk in ("yolo11n", "yolo11s", "yolo11m"):
            assert (mk, s, "4", "1") in c
    for b in ("2", "4", "8"):
        assert ("yolo11n", "640", "4", b) in c
    for t in ("1", "2"):
        for mk in ("yolo11n", "yolo11s", "yolo11m"):
            assert (mk, "640", t, "1") in c
        for s in ("320", "960"):
            assert ("yolo11n", s, t, "1") in c
    assert ("yolo11m", "640", "1", "1") not in {key(r) for r in m.cells("C")}   # optional only


def test_batch_frames_rule():
    rows = m.cells("C")
    frames = {arg(r, "--batch"): arg(r, "--frames") for r in rows if "--batch" in r["args"]}
    assert frames == {"2": "15", "4": "8", "8": "5"}          # max(5, ceil(30 / B))


def test_d_cells():
    rows = m.cells("D", include_concurrent=True)
    fa = [r for r in rows if "_fanauto_" in r["cell_id"]]
    fo = [r for r in rows if "_fanoff_" in r["cell_id"]]
    assert len(fa) == len(fo) == 3
    assert all(arg(r, "--fan") == "auto" and arg(r, "--cpu-mhz") == "2400" for r in fa)
    assert all(arg(r, "--fan") == "off" for r in fo)
    for fps in (1, 5, 15):
        mine = [r for r in rows if f"_serve{fps}_" in r["cell_id"]]
        assert len(mine) == 3
        for r in mine:
            assert arg(r, "--serve-fps") == str(fps) and arg(r, "--serve") == str(20 * fps)
            assert "--serve-gap-ms" not in r["args"]
    st = [r for r in rows if "_stream_" in r["cell_id"]]
    assert len(st) == 3
    assert all(arg(r, "--frames-source") == "stream" and arg(r, "--frames") == "50" for r in st)
    c2 = [r for r in rows if "_conc2_" in r["cell_id"]]
    cs = [r for r in rows if "_concspin_" in r["cell_id"]]
    assert len(c2) == len(cs) == 3
    assert all(arg(r, "--concurrent") == "yolo11n.pt" for r in c2)
    assert all(arg(r, "--concurrent") == "spinner" for r in cs)
    assert not any("--concurrent" in r["args"] for r in m.cells("D", include_concurrent=False))


def test_soak_cells():
    rows = m.cells("soak")
    assert {r["model_key"] for r in rows} == {"yolo11n", "yolo11m"}
    for r in rows:
        assert r["family"] == "soak" and r["cpu_mhz"] == 2400 and r["timeout_s"] == 3000
        assert arg(r, "--soak-minutes") == "30" and arg(r, "--fan") == "full"
    assert all("timeout_s" not in r for r in m.cells("all") if r["family"] != "soak")
    assert m.DEFAULT_TIMEOUT_S == experiment.RUN_TIMEOUT_S


def test_timeout_only_when_above_default():
    soak = lambda mins: m.Config("soak", "yolo11n", 1, soak_minutes=mins)  # noqa: E731
    assert m.timeout_s(soak(30)) == 50 * 60
    assert m.timeout_s(soak(20)) is None                   # 40 min is not above the default
    assert m.timeout_s(soak(5)) is None
    assert m.timeout_s(m.Config("A", "yolo11s", 1)) is None


def test_ref_family_is_v1_args_with_counters_off():
    rows = m.cells("ref")
    assert Counter((r["model_key"], r["cpu_mhz"]) for r in rows) == {
        (k, mhz): 3 for k in ("yolo11n", "yolov8n") for mhz in (1500, 2000, 2400)}
    for r in rows:
        v1 = matrix._cell(r["model_key"], r["cpu_mhz"], r["repeat"], r["round"])
        assert r["args"] == [*v1["args"], "--perf-counters", "off"]
        assert "--holdout" not in r["args"] and "--fan-full" in r["args"]
        assert r["cell_id"] == f"v1ref_{r['model_key']}_{r['cpu_mhz']}_r{r['repeat']:02d}"
        assert (r["family"], r["holdout"], r["round"]) == ("ref", None, r["repeat"])
        assert "timeout_s" not in r
    assert [r["round"] for r in rows] == sorted(r["round"] for r in rows)
    assert m.cells("ref", seed=3) == m.cells("ref", seed=3)
    assert {r["cell_id"] for r in m.cells("all") if r["family"] == "ref"} == \
        {r["cell_id"] for r in rows}

def test_fan_full_unless_overridden():
    for r in v2_only(m.cells("all", **ALL_FLAGS)):
        want = "auto" if "_fanauto_" in r["cell_id"] else (
            "off" if "_fanoff_" in r["cell_id"] else "full")
        assert arg(r, "--fan") == want, r["cell_id"]


def test_rounds_and_determinism():
    rows = m.cells("all", **ALL_FLAGS)
    rounds = [r["round"] for r in rows]
    assert rounds == sorted(rounds) and set(rounds) == {0, 1, 2, 3, 4}
    assert Counter(rounds)[4] == 4 + 2                   # repeat 4: 4 A cells @2400 + 2 DVFS
    assert m.cells("all", seed=7) == m.cells("all", seed=7)
    assert [r["cell_id"] for r in m.cells("A", seed=1)] != [r["cell_id"] for r in m.cells("A", seed=2)]
    # repeat r of one config is in round r, and repeats of one config are spread across rounds
    one = by(rows, model_key="yolo11s", cpu_mhz=2400, family="A")
    assert sorted(r["round"] for r in one) == [0, 1, 2, 3, 4]


def test_unknown_family():
    with pytest.raises(ValueError):
        m.cells("Z")


def test_pilot():
    rows = m.cells("pilot")
    assert len(rows) == 29 + 2                      # the two concurrency cells are on by default
    assert all(r["cell_id"].startswith("pilot_") for r in rows)
    fams = Counter(r["family"] for r in rows)
    assert fams == {"A": 6, "A-input": 2, "B": 3, "C": 8, "D": 9, "soak": 1, "ref": 2}
    dv = [r for r in rows if "_dvfs_" in r["cell_id"]]
    assert sorted(r["repeat"] for r in dv) == [0, 1]
    soak = by(rows, family="soak")[0]
    assert soak["cell_id"] == "pilot_yolo11n_soak5_640_t4_2400_r00"
    assert arg(soak, "--soak-minutes") == "5" and "timeout_s" not in soak
    assert arg(soak, "--fan") == "full"
    refs = by(rows, family="ref")
    assert sorted(r["cell_id"] for r in refs) == ["pilot_v1ref_yolo11n_2400_r00",
                                                  "pilot_v1ref_yolov8n_2400_r00"]
    # one per configuration group: no repeated id stem except the 2 DVFS repeats
    stems = Counter(r["cell_id"][:-4] for r in rows)
    assert [s for s, n in stems.items() if n > 1] == ["pilot_yolo11n_640_t4_dvfs"]
    assert len(m.cells("pilot", **ALL_FLAGS)) == 31 + 3 + 3   # + fp16 (1), int8d x3 (3 more with the p30/p50 pair)
    assert len(m.cells("pilot", include_fp16=True)) == 32
    assert len(m.cells("pilot", include_concurrent=False)) == 29


def test_cli_plan_v2(tmp_path):
    host = Path(__file__).resolve().parents[1]
    out = tmp_path / "man.jsonl"
    r = subprocess.run([sys.executable, str(host / "experiment.py"), "plan", "--v2", "B",
                        "--include-fp16", "--out", str(out)], capture_output=True, text=True,
                       cwd=host)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 15 and "wrote 15 cells" in r.stdout
    bad = subprocess.run([sys.executable, str(host / "experiment.py"), "plan", "--v2", "A",
                          "--smoke", "--out", str(tmp_path / "x.jsonl")],
                         capture_output=True, text=True, cwd=host)
    assert bad.returncode != 0


def test_cli_plan_v2_concurrency_cells_on_by_default_and_no_concurrent_drops_them(tmp_path):
    host = Path(__file__).resolve().parents[1]

    def plan(*extra):
        out = tmp_path / f"m{len(extra)}.jsonl"
        r = subprocess.run([sys.executable, str(host / "experiment.py"), "plan", "--v2", "D",
                            *extra, "--out", str(out)], capture_output=True, text=True, cwd=host)
        assert r.returncode == 0, r.stderr
        return [json.loads(x) for x in out.read_text().splitlines()]
    assert len(plan()) == 34
    rows = plan("--no-concurrent")
    assert len(rows) == 28 and not any("--concurrent" in r["args"] for r in rows)
