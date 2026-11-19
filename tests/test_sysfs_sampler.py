# tests/test_sysfs_sampler.py
import csv
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sysfs_sampler import ROW_KEYS, SysfsSampler, ThermalAbort, thermal_exit_code

CPU = "sys/devices/system/cpu"
THERMAL = "sys/class/thermal"


def _put(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def _tree(root, khz=(2400000, 2400000, 1500000, 2000000), milli_c=48750, fan=2):
    for i, k in enumerate(khz):
        _put(root, f"{CPU}/cpu{i}/cpufreq/scaling_cur_freq", f"{k}\n")
    _put(root, f"{THERMAL}/thermal_zone0/temp", f"{milli_c}\n")
    _put(root, f"{THERMAL}/cooling_device0/cur_state", f"{fan}\n")


def _sampler(root, interrupts=None, **kw):
    ticks = iter(range(1000, 10**6, 7))
    return SysfsSampler(root=str(root), clock=lambda: next(ticks),
                        interrupt=((lambda: interrupts.append(1)) if interrupts is not None else (lambda: None)),
                        **kw)


def _set_temp(root, milli_c):
    _put(root, f"{THERMAL}/thermal_zone0/temp", f"{milli_c}\n")


def test_row_keys_exact():
    assert ROW_KEYS == ["t_real_ns", "cpu0_kHz", "cpu1_kHz", "cpu2_kHz", "cpu3_kHz",
                        "temp_C", "fan_state"]


def test_read_once_parses_fake_tree(tmp_path):
    _tree(tmp_path)
    s = _sampler(tmp_path)
    row = s.read_once()
    assert row == {"t_real_ns": 1000, "cpu0_kHz": 2400000, "cpu1_kHz": 2400000,
                   "cpu2_kHz": 1500000, "cpu3_kHz": 2000000, "temp_C": 48.75, "fan_state": 2}
    assert isinstance(row["cpu0_kHz"], int) and isinstance(row["fan_state"], int)
    assert isinstance(row["temp_C"], float)
    assert list(row) == ROW_KEYS
    assert s.rows() == [row]


def test_missing_files_give_none_and_never_raise(tmp_path):
    s = _sampler(tmp_path)            # empty root: nothing exists
    row = s.read_once()
    assert row["t_real_ns"] == 1000
    assert all(row[k] is None for k in ROW_KEYS[1:])
    assert s.max_temp_C() is None
    assert s.thermal_abort is None


def test_unparseable_and_partial_values_are_none(tmp_path):
    _tree(tmp_path)
    _put(tmp_path, f"{CPU}/cpu1/cpufreq/scaling_cur_freq", "garbage\n")
    os.remove(tmp_path / CPU / "cpu3/cpufreq/scaling_cur_freq")
    _put(tmp_path, f"{THERMAL}/cooling_device0/cur_state", "")
    row = _sampler(tmp_path).read_once()
    assert row["cpu0_kHz"] == 2400000 and row["cpu2_kHz"] == 1500000
    assert row["cpu1_kHz"] is None and row["cpu3_kHz"] is None
    assert row["fan_state"] is None
    assert row["temp_C"] == 48.75


def test_max_temp_ignores_none(tmp_path):
    _tree(tmp_path, milli_c=40000)
    s = _sampler(tmp_path)
    s.read_once()
    _set_temp(tmp_path, 61500)
    s.read_once()
    os.remove(tmp_path / THERMAL / "thermal_zone0/temp")
    s.read_once()
    _set_temp(tmp_path, 50000)
    s.read_once()
    assert s.max_temp_C() == 61.5


def test_below_threshold_sets_nothing(tmp_path):
    _tree(tmp_path, milli_c=84999)
    hits = []
    s = _sampler(tmp_path, interrupts=hits)
    s.read_once()
    assert s.thermal_abort is None and hits == []


def test_abort_then_hard_stop_transitions(tmp_path):
    _tree(tmp_path, milli_c=84000)
    hits = []
    s = _sampler(tmp_path, interrupts=hits)
    s.read_once()
    assert s.thermal_abort is None
    _set_temp(tmp_path, 85000)
    row_abort = s.read_once()
    assert s.thermal_abort == {"t_real_ns": row_abort["t_real_ns"], "temp_C": 85.0,
                               "level": "abort"}
    assert hits == []                                   # abort alone never interrupts
    _set_temp(tmp_path, 86000)
    s.read_once()
    first = dict(s.thermal_abort)
    assert first["level"] == "abort" and first["temp_C"] == 85.0   # first crossing is kept
    _set_temp(tmp_path, 90000)
    row_hard = s.read_once()
    assert s.thermal_abort == {"t_real_ns": row_hard["t_real_ns"], "temp_C": 90.0,
                               "level": "hard_stop"}
    assert len(hits) == 1
    _set_temp(tmp_path, 95000)
    s.read_once()
    assert len(hits) == 1                               # interrupt only once
    assert s.thermal_abort["level"] == "hard_stop" and s.thermal_abort["temp_C"] == 90.0


def test_direct_jump_to_hard_stop_interrupts_once(tmp_path):
    _tree(tmp_path, milli_c=91000)
    hits = []
    s = _sampler(tmp_path, interrupts=hits)
    s.read_once()
    s.read_once()
    assert s.thermal_abort["level"] == "hard_stop" and s.thermal_abort["temp_C"] == 91.0
    assert len(hits) == 1


def test_custom_limits(tmp_path):
    _tree(tmp_path, milli_c=50000)
    hits = []
    s = _sampler(tmp_path, interrupts=hits, abort_c=45.0, hard_stop_c=60.0)
    s.read_once()
    assert s.thermal_abort["level"] == "abort" and hits == []


def test_write_csv_header_and_order(tmp_path):
    _tree(tmp_path)
    s = _sampler(tmp_path)
    s.read_once()
    os.remove(tmp_path / THERMAL / "cooling_device0/cur_state")
    s.read_once()
    out = tmp_path / "sysfs.csv"
    s.write_csv(str(out))
    with open(out, newline="") as f:
        lines = list(csv.reader(f))
    assert lines[0] == ROW_KEYS
    assert lines[1] == ["1000", "2400000", "2400000", "1500000", "2000000", "48.75", "2"]
    assert lines[2][-1] == ""                           # None -> empty cell
    assert len(lines) == 3


def test_thread_samples_and_stops(tmp_path):
    _tree(tmp_path)
    s = SysfsSampler(period_s=0.01, root=str(tmp_path), interrupt=lambda: None)
    s.start()
    deadline = time.time() + 5
    while len(s.rows()) < 3 and time.time() < deadline:
        time.sleep(0.01)
    s.stop()
    n = len(s.rows())
    assert n >= 3
    time.sleep(0.05)
    assert len(s.rows()) == n                           # stopped: no more rows
    ts = [r["t_real_ns"] for r in s.rows()]
    assert ts == sorted(ts)
    s.stop()                                            # idempotent


def test_start_stop_with_no_start_is_safe(tmp_path):
    s = _sampler(tmp_path)
    s.stop()


def test_thread_survives_unexpected_error(tmp_path):
    _tree(tmp_path)
    calls = {"n": 0}

    def flaky_clock():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return calls["n"]

    s = SysfsSampler(period_s=0.01, root=str(tmp_path), clock=flaky_clock,
                     interrupt=lambda: None)
    s.start()
    deadline = time.time() + 5
    while len(s.rows()) < 2 and time.time() < deadline:
        time.sleep(0.01)
    s.stop()
    assert len(s.rows()) >= 2


def test_thermal_exit_code():
    assert thermal_exit_code(None) is None
    assert thermal_exit_code({"t_real_ns": 1, "temp_C": 85.2, "level": "abort"}) == 3
    assert thermal_exit_code({"t_real_ns": 1, "temp_C": 90.1, "level": "hard_stop"}) == 4


def test_thermal_abort_exception_carries_details():
    d = {"t_real_ns": 5, "temp_C": 85.5, "level": "abort"}
    e = ThermalAbort(d)
    assert e.details == d and "85.5" in str(e)
    with pytest.raises(ThermalAbort):
        raise e
