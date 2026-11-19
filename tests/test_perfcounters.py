# tests/test_perfcounters.py
import csv
import os
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from perfcounters import (DEFAULT_EVENTS, PerfStat, counters_meta, first_line_time_s,
                          min_pct_running, parse_interval_csv, read_paranoid, to_wide,
                          write_counters_csv)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "perf_interval_sample.txt")
T0 = 1_700_000_000_000_000_000
END1, END2, END3 = 100109271, 200292207, 300448125   # fixture interval ends (ns after perf zero)


def NOSLEEP(_s):
    pass


def _fixture_text():
    with open(FIXTURE) as f:
        return f.read()


class FakeProc:
    """Stands in for subprocess.Popen: records signals; communicate() can time out once."""

    def __init__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        self.signals, self.killed = [], False
        self.returncode = None
        self.timeouts = 0           # communicate() raises TimeoutExpired this many times
        self.exit_on_signal = -signal.SIGINT
        self.stderr_text = ""

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if self.returncode is None and self.timeouts == 0:
            self.returncode = self.exit_on_signal

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL

    def communicate(self, timeout=None):
        if self.timeouts > 0:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return ("", self.stderr_text)


def _popen_factory(procs):
    def popen(argv, **kw):
        p = FakeProc(argv, **kw)
        procs.append(p)
        return p
    return popen


# ---- parse_interval_csv ----------------------------------------------------

def test_parse_fixture_row_count_and_event_names():
    rows = parse_interval_csv(_fixture_text(), T0)
    assert len(rows) == 3 * len(DEFAULT_EVENTS)    # metric-only lines are not rows
    assert {r["event"] for r in rows} == set(DEFAULT_EVENTS)   # ':u' stripped
    assert all(":" not in r["event"] for r in rows)


def test_parse_counts_and_pct_running():
    rows = parse_interval_csv(_fixture_text(), T0)
    first = {r["event"]: r for r in rows if r["t_end_ns"] == T0 + END1}
    assert first["cycles"]["count"] == 239637235
    assert first["instructions"]["count"] == 403327456
    assert first["cycles"]["pct_running"] == pytest.approx(100.0)
    assert first["task-clock"]["pct_running"] == pytest.approx(100.0)
    assert first["page-faults"]["count"] == 0


def test_parse_timestamps_come_from_t0():
    rows = parse_interval_csv(_fixture_text(), T0)
    assert sorted({r["t_end_ns"] for r in rows}) == [T0 + END1, T0 + END2,
                                                     T0 + END3]


def test_parse_interval_s_is_time_since_previous_interval():
    rows = parse_interval_csv(_fixture_text(), T0)
    by_end = {r["t_end_ns"]: r["interval_s"] for r in rows}
    assert by_end[T0 + END1] == pytest.approx(0.100109271)
    assert by_end[T0 + END2] == pytest.approx(0.200292207 - 0.100109271)
    assert by_end[T0 + END3] == pytest.approx(0.300448125 - 0.200292207)


def test_parse_not_counted_and_not_supported_are_none():
    rows = parse_interval_csv(_fixture_text(), T0)
    third = {r["event"]: r for r in rows if r["t_end_ns"] == T0 + END3}
    assert third["LLC-load-misses"]["count"] is None
    assert third["LLC-load-misses"]["pct_running"] == 0.0
    assert third["branch-misses"]["count"] is None       # <not supported>
    assert third["cycles"]["count"] == 240092214


def test_parse_skips_comments_blank_and_metric_only_lines():
    text = ("# started on Thu\n\n"
            "0.1,5,,cycles:u,10,100.00,,\n"
            "0.1,,,,,,0.12,stalled cycles per insn\n"
            "garbage line\n")
    rows = parse_interval_csv(text, T0)
    assert [(r["event"], r["count"]) for r in rows] == [("cycles", 5)]


def test_parse_strips_k_modifier_and_missing_pct():
    rows = parse_interval_csv("0.2,7,,instructions:k,,,,\n", T0)
    assert rows[0]["event"] == "instructions"
    assert rows[0]["pct_running"] is None
    assert rows[0]["count"] == 7


def test_parse_empty_text():
    assert parse_interval_csv("", T0) == []


# ---- to_wide / write_counters_csv ------------------------------------------

def test_to_wide_one_row_per_interval_with_start_end():
    wide = to_wide(parse_interval_csv(_fixture_text(), T0))
    assert [w["t_end_ns"] for w in wide] == [T0 + END1, T0 + END2, T0 + END3]
    assert wide[0]["t_start_ns"] == T0
    assert wide[1]["t_start_ns"] == wide[0]["t_end_ns"]
    assert wide[2]["t_start_ns"] == wide[1]["t_end_ns"]


def test_to_wide_columns_values_and_name_normalisation():
    wide = to_wide(parse_interval_csv(_fixture_text(), T0))
    w = wide[0]
    assert w["ctr_cycles"] == 239637235
    assert w["pct_cycles"] == pytest.approx(100.0)
    assert w["ctr_LLC_load_misses"] == 14
    assert w["ctr_stalled_cycles_backend"] == 47355958
    assert wide[2]["ctr_LLC_load_misses"] is None
    assert wide[2]["pct_LLC_load_misses"] == 0.0


def test_to_wide_includes_requested_event_that_perf_never_printed():
    rows = parse_interval_csv("0.1,5,,cycles:u,10,100.00,,\n", T0)
    w = to_wide(rows)[0]
    assert w["ctr_instructions"] is None and w["pct_instructions"] is None


def test_to_wide_empty():
    assert to_wide([]) == []


def test_write_counters_csv_column_order(tmp_path):
    wide = to_wide(parse_interval_csv(_fixture_text(), T0))
    p = tmp_path / "counters.csv"
    write_counters_csv(str(p), wide)
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames
    norm = [e.replace("-", "_") for e in DEFAULT_EVENTS]
    assert header == (["t_start_ns", "t_end_ns"] + [f"ctr_{e}" for e in norm]
                      + [f"pct_{e}" for e in norm])
    assert len(rows) == 3
    assert rows[2]["ctr_LLC_load_misses"] == ""    # None -> empty cell
    assert rows[0]["ctr_cycles"] == "239637235"


def test_write_counters_csv_header_only_when_no_rows(tmp_path):
    p = tmp_path / "counters.csv"
    write_counters_csv(str(p), [])
    with open(p, newline="") as f:
        assert next(csv.reader(f))[:3] == ["t_start_ns", "t_end_ns", "ctr_cycles"]
        assert list(csv.reader(f)) == []


# ---- PerfStat --------------------------------------------------------------

def test_available_follows_which():
    assert PerfStat(1, which=lambda name: "/usr/bin/perf").available() is True
    assert PerfStat(1, which=lambda name: None).available() is False


def test_available_looks_up_perf():
    seen = []
    PerfStat(1, which=lambda name: seen.append(name) or "/x").available()
    assert seen == ["perf"]


def test_start_builds_exact_argv_and_records_t0_before_spawn(tmp_path):
    procs, order = [], []
    popen = _popen_factory(procs)

    def spying_popen(argv, **kw):
        order.append("popen")
        return popen(argv, **kw)

    def clock():
        order.append("clock")
        return 123456789

    raw = str(tmp_path / "counters.raw.txt")
    ps = PerfStat(4242, interval_ms=200, out_path=raw, sleep=NOSLEEP, popen=spying_popen, clock=clock,
                  which=lambda n: "/usr/bin/perf")
    assert ps.start() is True
    assert order == ["clock", "popen"]
    assert ps.t0_ns == 123456789
    assert procs[0].argv == ["perf", "stat", "-I", "200", "-x", ",", "-o", raw,
                             "-e", ",".join(DEFAULT_EVENTS), "-p", "4242"]


def test_start_default_interval_is_100ms(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/usr/bin/perf")
    ps.start()
    assert procs[0].argv[procs[0].argv.index("-I") + 1] == "100"


def test_start_when_perf_missing_records_error_and_does_not_spawn(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: None)
    assert ps.start() is False
    assert procs == []
    assert "not found" in ps.error


def test_start_oserror_is_recorded_not_raised(tmp_path):
    def boom(argv, **kw):
        raise OSError("exec failed")

    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=boom, which=lambda n: "/p")
    assert ps.start() is False
    assert "exec failed" in ps.error


def test_stop_sends_sigint_and_is_clean(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/p")
    ps.start()
    ps.stop()
    assert procs[0].signals == [signal.SIGINT]
    assert procs[0].killed is False
    assert ps.error is None


def test_stop_kills_after_timeout(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/p")
    ps.start()
    procs[0].timeouts = 1
    ps.stop()
    assert procs[0].signals == [signal.SIGINT]
    assert procs[0].killed is True
    assert "kill" in ps.error


def test_stop_reports_perf_that_died_before_stop(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/p")
    ps.start()
    procs[0].returncode = 255
    procs[0].stderr_text = "Error:\nThe sys_perf_event_open() syscall returned with 3 (No such process)"
    ps.stop()
    assert procs[0].signals == []
    assert "rc=255" in ps.error and "No such process" in ps.error


def test_stop_is_idempotent_and_safe_without_start(tmp_path):
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), which=lambda n: "/p")
    ps.stop()                         # never started
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "r.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/p")
    ps.start()
    ps.stop()
    ps.stop()
    assert procs[0].signals == [signal.SIGINT]


def test_wide_rows_parses_raw_file_from_t0(tmp_path):
    raw = tmp_path / "counters.raw.txt"
    procs = []
    ps = PerfStat(7, out_path=str(raw), sleep=NOSLEEP, popen=_popen_factory(procs), clock=lambda: T0,
                  which=lambda n: "/p")
    ps.start()
    raw.write_text(_fixture_text())
    ps.stop()
    wide = ps.wide_rows()
    assert len(wide) == 3
    assert wide[0]["t_start_ns"] == T0 and wide[0]["t_end_ns"] == T0 + END1


def test_wide_rows_missing_raw_file_is_empty_and_recorded(tmp_path):
    procs = []
    ps = PerfStat(7, out_path=str(tmp_path / "absent.txt"), sleep=NOSLEEP, popen=_popen_factory(procs),
                  which=lambda n: "/p")
    ps.start()
    ps.stop()
    assert ps.wide_rows() == []
    assert "absent.txt" in ps.error


def test_wide_rows_before_start_is_empty(tmp_path):
    assert PerfStat(7, out_path=str(tmp_path / "r.txt"), which=lambda n: "/p").wide_rows() == []


# ---- run_meta block --------------------------------------------------------

def test_read_paranoid(tmp_path):
    p = tmp_path / "perf_event_paranoid"
    p.write_text("2\n")
    assert read_paranoid(str(p)) == 2
    assert read_paranoid(str(tmp_path / "missing")) is None
    p.write_text("junk")
    assert read_paranoid(str(p)) is None


@pytest.mark.parametrize("paranoid,scope", [(2, "user"), (3, "user"), (1, "user+kernel"),
                                            (0, "user+kernel"), (-1, "user+kernel")])
def test_meta_scope_from_paranoid(paranoid, scope):
    m = counters_meta(enabled=True, available=True, interval_ms=100, events=DEFAULT_EVENTS,
                      t0_ns=5, error=None, paranoid=paranoid)
    assert m["scope"] == scope


def test_meta_keys_and_unknown_paranoid():
    m = counters_meta(enabled=False, available=False, interval_ms=200, events=("cycles",),
                      t0_ns=None, error="x", paranoid=None)
    assert set(m) == {"enabled", "available", "interval_ms", "events", "t0_ns", "offset_ns",
                      "offset_method", "error", "paranoid", "scope", "min_pct_running"}
    assert m["events"] == ["cycles"] and m["paranoid"] is None and m["scope"] is None


def test_perfstat_meta_reflects_state(tmp_path):
    procs = []
    ps = PerfStat(7, interval_ms=200, out_path=str(tmp_path / "r.txt"),
                  sleep=NOSLEEP, popen=_popen_factory(procs), clock=lambda: 99, which=lambda n: "/p")
    ps.start()
    ps.stop()
    m = ps.meta(enabled=True, paranoid=2)
    assert m == {"enabled": True, "available": True, "interval_ms": 200,
                 "events": list(DEFAULT_EVENTS), "t0_ns": 99, "offset_ns": 0,
                 "offset_method": "none", "error": None, "paranoid": 2, "scope": "user",
                 "min_pct_running": None}


# ---- event set (ruling R13) -------------------------------------------------

def test_default_events_are_the_non_multiplexing_set():
    assert DEFAULT_EVENTS == ("cycles", "instructions", "cache-references", "cache-misses",
                              "branch-misses", "stalled-cycles-backend", "LLC-load-misses",
                              "task-clock", "context-switches", "page-faults")


def test_min_pct_running_over_hardware_events_only():
    rows = [{"event": "cycles", "pct_running": 98.0},
            {"event": "instructions", "pct_running": 71.5},
            {"event": "task-clock", "pct_running": 10.0},        # software event ignored
            {"event": "cache-misses", "pct_running": None}]      # missing ignored
    assert min_pct_running(rows) == 71.5
    assert min_pct_running([]) is None
    assert min_pct_running([{"event": "page-faults", "pct_running": 100.0}]) is None


def test_min_pct_running_counts_not_counted_cells_as_zero():
    assert min_pct_running(parse_interval_csv(_fixture_text(), T0)) == 0.0


def test_meta_min_pct_running_comes_from_parsed_raw_file(tmp_path):
    raw = tmp_path / "raw.txt"
    ps = PerfStat(7, out_path=str(raw), sleep=NOSLEEP, popen=_popen_factory([]),
                  clock=lambda: T0, which=lambda n: "/p")
    ps.start()
    raw.write_text("0.1,5,,cycles:u,10,87.50,,\n0.1,6,,instructions:u,10,100.00,,\n")
    ps.stop()
    ps.wide_rows()
    assert ps.meta(True, 2)["min_pct_running"] == 87.5


# ---- start-offset correction (ruling R14) -----------------------------------

FIRST_LINE = "0.100109271,239637235,,cycles:u,100057703,100.00,2.395,GHz\n"


class FakeWorld:
    """Fake clock + sleep; the raw file appears once `appear_ns` of fake time has elapsed."""

    def __init__(self, path, appear_ns, text=FIRST_LINE, start_ns=T0):
        self.now, self.path, self.appear_ns, self.text, self.t_start = start_ns, path, appear_ns, text, start_ns

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += round(s * 1e9)
        if self.appear_ns is not None and self.now - self.t_start >= self.appear_ns:
            with open(self.path, "w") as f:
                f.write(self.text)


def _world_ps(tmp_path, appear_ns, **kw):
    raw = str(tmp_path / "counters.raw.txt")
    w = FakeWorld(raw, appear_ns, **kw)
    ps = PerfStat(7, out_path=raw, clock=w.clock, sleep=w.sleep, popen=_popen_factory([]),
                  which=lambda n: "/p")
    return ps, w


def test_offset_is_first_seen_minus_t0_minus_first_time(tmp_path):
    ps, w = _world_ps(tmp_path, appear_ns=135_000_000)
    assert ps.start() is True
    # the file is written on the 68th 2 ms sleep (136 ms); the next poll sees it
    assert w.now == T0 + 136_000_000
    assert ps.offset_ns == 136_000_000 - 100_109_271
    assert ps.offset_method == "first-line"
    assert ps.t0_ns == T0


def test_offset_shifts_all_timestamps(tmp_path):
    ps, w = _world_ps(tmp_path, appear_ns=135_000_000)
    ps.start()
    with open(ps.out_path, "w") as f:
        f.write(_fixture_text())
    ps.stop()
    wide = ps.wide_rows()
    base = T0 + ps.offset_ns
    assert wide[0]["t_start_ns"] == base
    assert wide[0]["t_end_ns"] == base + END1
    assert wide[2]["t_end_ns"] == base + END3
    m = ps.meta(True, 2)
    assert m["t0_ns"] == T0 and m["offset_ns"] == ps.offset_ns and m["offset_method"] == "first-line"


def test_offset_is_clamped_at_zero(tmp_path):
    # first line already present with a time larger than the elapsed wall time
    ps, w = _world_ps(tmp_path, appear_ns=0, text="9.0,5,,cycles:u,10,100.00,,\n")
    ps.start()
    assert ps.offset_ns == 0 and ps.offset_method == "first-line"


def test_offset_ignores_incomplete_last_line(tmp_path):
    half = "0.100109271,239637235,,cyc"      # perf mid-write: no newline yet
    ps, w = _world_ps(tmp_path, appear_ns=0, text=half)
    ps.start()
    assert ps.offset_method == "none" and ps.offset_ns == 0


def test_offset_gives_up_after_two_seconds(tmp_path):
    ps, w = _world_ps(tmp_path, appear_ns=None)
    assert ps.start() is True
    assert w.now - T0 == 2_000_000_000
    assert ps.offset_ns == 0 and ps.offset_method == "none"


def test_offset_poll_stops_when_perf_dies(tmp_path):
    procs = []
    raw = str(tmp_path / "counters.raw.txt")
    w = FakeWorld(raw, None)

    def dying_popen(argv, **kw):
        p = FakeProc(argv, **kw)
        p.returncode = 1
        procs.append(p)
        return p

    ps = PerfStat(7, out_path=raw, clock=w.clock, sleep=w.sleep, popen=dying_popen,
                  which=lambda n: "/p")
    ps.start()
    assert w.now == T0           # no sleeping once perf has exited
    assert ps.offset_method == "none"


def test_start_removes_stale_raw_file(tmp_path):
    raw = tmp_path / "counters.raw.txt"
    raw.write_text(FIRST_LINE)
    ps = PerfStat(7, out_path=str(raw), sleep=NOSLEEP, popen=_popen_factory([]),
                  clock=lambda: T0, which=lambda n: "/p")
    ps.start()
    assert ps.offset_method == "none"       # the stale line was not mistaken for perf output


def test_first_line_time_s():
    assert first_line_time_s("") is None
    assert first_line_time_s("# started on x\n\n") is None
    assert first_line_time_s(FIRST_LINE) == pytest.approx(0.100109271)
    assert first_line_time_s(FIRST_LINE + "0.1001") == pytest.approx(0.100109271)
    assert first_line_time_s("0.100109271,239637235,,cyc") is None
