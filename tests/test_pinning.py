# tests/test_pinning.py
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pinning
from pinning import (ALLOWED_MHZ, apply, apply_mode, clock_errors, clock_range_errors,
                     new_undervoltage, pin, status, throttle_errors, unpin, wait_fan_full)

HELPER_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "pi-power-pin")
H = "/usr/local/sbin/pi-power-pin"


def _runner(rc=0, out="governor=performance cur_kHz=2400000", err=""):
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
    return run, calls


def test_apply_calls_whitelisted_helper_non_interactively():
    run, calls = _runner()
    status = apply(2400, True, run=run)
    assert calls == [["sudo", "-n", "/usr/local/sbin/pi-power-pin", "fan", "full"],
                     ["sudo", "-n", "/usr/local/sbin/pi-power-pin", "pin", "2400"]]
    assert "performance" in status


def test_apply_rejects_unsupported_clock_before_calling_sudo():
    run, calls = _runner()
    with pytest.raises(ValueError):
        apply(2450, False, run=run)
    assert calls == [] and 1500 in ALLOWED_MHZ and 2400 in ALLOWED_MHZ


def test_apply_surfaces_missing_sudoers_rule():
    run, _ = _runner(rc=1, err="sudo: a password is required")
    with pytest.raises(RuntimeError, match="sudoers"):
        apply(2400, False, run=run)


def test_apply_noop_when_nothing_requested():
    run, calls = _runner()
    assert apply(None, False, run=run) == "" and calls == []


def test_wait_fan_full_polls_until_max_state():
    states = iter([(1, 4), (3, 4), (4, 4)])
    assert wait_fan_full(read=lambda: next(states), sleep=lambda s: None) == (4, 4)


def test_wait_fan_full_times_out():
    with pytest.raises(RuntimeError, match="fan"):
        wait_fan_full(read=lambda: (2, 4), timeout_s=0.0, sleep=lambda s: None)


def test_clock_errors_flags_drift_throttle_and_missing_readings():
    ok = [{"t_ms": 0, "arm_MHz": 2400, "throttle_bits": 0},
          {"t_ms": 5000, "arm_MHz": 2399, "throttle_bits": 0x50000}]
    assert clock_errors(ok, 2400) == []        # latched-only bits (0x50000) are not current
    bad = [{"t_ms": 0, "arm_MHz": 1800, "throttle_bits": 0x4}]
    errs = clock_errors(bad, 2400)
    assert any("1800" in e for e in errs) and any("0x4" in e for e in errs)
    assert clock_errors([{"t_ms": 0, "error": "x"}], 2400) == ["no arm clock readings in context"]


def test_throttle_errors_any_row():
    rows = [{"t_ms": 0, "throttle_bits": 0}, {"t_ms": 5000, "throttle_bits": 0x1},
            {"t_ms": 9000, "error": "x"}, {"t_ms": 10000, "throttle_bits": 0x50000}]
    errs = throttle_errors(rows)
    assert len(errs) == 1 and "0x1" in errs[0]
    assert throttle_errors([]) == []


def test_new_sticky_undervoltage_detected():
    assert new_undervoltage(0x0, 0x10000) is True
    assert new_undervoltage(0x10000, 0x10000) is False   # already latched before the run
    assert new_undervoltage(0x0, 0x40000) is False       # other sticky bits do not count
    assert new_undervoltage(None, 0x10000) is False


def test_status_never_raises():
    run, calls = _runner(out="governor=performance\n")
    assert status(run=run) == "governor=performance"
    run, _ = _runner(rc=1, err="boom")
    assert status(run=run) == "unavailable: boom"
    def missing(cmd, **kw):
        raise FileNotFoundError("sudo")
    assert status(run=missing).startswith("unavailable:")


def test_read_fan_checks_type(tmp_path, monkeypatch):
    for name, val in (("type", "pwm-fan"), ("cur_state", "4"), ("max_state", "4")):
        (tmp_path / name).write_text(val + "\n")
    monkeypatch.setattr(pinning, "FAN", str(tmp_path))
    assert pinning._read_fan() == (4, 4)
    (tmp_path / "type").write_text("cpu-thermal\n")
    with pytest.raises(RuntimeError, match="pwm-fan"):
        pinning._read_fan()


def _argvs(calls):
    return [c[3:] for c in calls]


@pytest.mark.parametrize("fan,governor,mhz,expected", [
    ("full", "pinned", 2400, [["fan", "full"], ["pin", "2400"]]),
    ("auto", "pinned", 2000, [["fan", "auto"], ["pin", "2000"]]),
    ("off", "pinned", 1500, [["fan", "off"], ["pin", "1500"]]),
    ("none", "pinned", 2400, [["pin", "2400"]]),
    ("auto", "ondemand", None, [["fan", "auto"], ["unpin"]]),
    ("off", "ondemand", None, [["fan", "off"], ["unpin"]]),
    ("none", "ondemand", None, [["unpin"]]),
    ("full", "pinned", None, [["fan", "full"]]),
    ("none", "pinned", None, []),
])
def test_apply_mode_command_paths(fan, governor, mhz, expected):
    run, calls = _runner()
    apply_mode(mhz, fan, governor, run=run)
    assert all(c[:3] == ["sudo", "-n", H] for c in calls)
    assert _argvs(calls) == expected


def test_apply_mode_returns_last_status_and_empty_when_noop():
    run, _ = _runner(out="governor=ondemand\n")
    assert apply_mode(None, "none", "ondemand", run=run) == "governor=ondemand"
    run, calls = _runner()
    assert apply_mode(None, "none", "pinned", run=run) == "" and calls == []


@pytest.mark.parametrize("args", [
    (2450, "none", "pinned"),
    (2400, "bogus", "pinned"),
    (2400, "none", "performance"),
])
def test_apply_mode_validates_before_calling_sudo(args):
    run, calls = _runner()
    with pytest.raises(ValueError):
        apply_mode(*args, run=run)
    assert calls == []


def test_apply_mode_surfaces_missing_sudoers_rule():
    run, _ = _runner(rc=1, err="sudo: a password is required")
    with pytest.raises(RuntimeError, match="sudoers"):
        apply_mode(None, "auto", "ondemand", run=run)


@pytest.mark.parametrize("mhz", [None, 1500, 2000, 2400])
@pytest.mark.parametrize("fan_full", [True, False])
def test_apply_equivalent_to_apply_mode(mhz, fan_full):
    run_a, calls_a = _runner()
    run_b, calls_b = _runner()
    ra = apply(mhz, fan_full, run=run_a)
    rb = apply_mode(mhz, "full" if fan_full else "none", "pinned", run=run_b)
    assert calls_a == calls_b and ra == rb


def test_pin_and_unpin_wrappers():
    run, calls = _runner()
    pin(2400, run=run)
    unpin(run=run)
    assert _argvs(calls) == [["pin", "2400"], ["unpin"]]
    with pytest.raises(ValueError):
        pin(2450, run=run)
    with pytest.raises(ValueError):
        pin(None, run=run)
    assert len(calls) == 2


def _row(freqs=(2400000,) * 4, **kw):
    r = {"t_real_ns": 1, "temp_C": 50.0, "fan_state": 0}
    r.update({f"cpu{i}_kHz": f for i, f in enumerate(freqs)})
    r.update(kw)
    return r


def test_clock_range_errors_ok_inside_range_and_tolerance():
    rows = [_row((1500000,) * 4), _row((2400000, 2000000, 1975000, 2425000)),
            _row((1475000, None, None, None))]
    assert clock_range_errors(rows) == []


def test_clock_range_errors_one_error_per_offending_row():
    rows = [_row(), _row((2400000, 1200000, 2400000, 2400000)),
            _row((2600000,) * 4), _row((1000000, 3000000, 2400000, 2400000))]
    errs = clock_range_errors(rows)
    assert len(errs) == 3
    assert "cpu1_kHz=1200000" in errs[0]
    assert "cpu0_kHz=2600000" in errs[1]
    assert "cpu0_kHz=1000000" in errs[2] and "cpu1_kHz=3000000" in errs[2]


def test_clock_range_errors_custom_bounds_and_tolerance():
    rows = [_row((1800000,) * 4)]
    assert clock_range_errors(rows, lo_mhz=1800, hi_mhz=1800, tol_mhz=0) == []
    assert len(clock_range_errors(rows, lo_mhz=1900, hi_mhz=2400, tol_mhz=25)) == 1


def test_clock_range_errors_no_readings():
    assert clock_range_errors([]) == ["no cpu frequency readings"]
    rows = [_row((None,) * 4), {"t_real_ns": 2}]
    assert clock_range_errors(rows) == ["no cpu frequency readings"]


def _helper_branch(name):
    """Text of the `name)` branch of the helper's top-level case statement."""
    text = open(HELPER_PATH).read()
    start = text.index(f"\n    {name})")
    end = text.index("\n        ;;", start)
    return text[start:end]


def test_helper_fan_off_trips_and_fan_auto_trips():
    text = open(HELPER_PATH).read()
    assert 'OFF_TRIPS="85000 86000 87000 88000"' in text
    assert 'STOCK_TRIPS="50000 60000 67500 75000"' in text
    fan = _helper_branch("fan")
    assert "auto" in fan and "off" in fan and "full" in fan
    assert 'set_trips "$OFF_TRIPS"' in fan and 'set_trips "$STOCK_TRIPS"' in fan


def test_helper_unpin_leaves_fan_alone():
    unpin_branch = _helper_branch("unpin")
    assert "set_freq ondemand 1500000 2400000" in unpin_branch
    assert "set_trips" not in unpin_branch
    assert "set_trips" in _helper_branch("restore")   # restore still resets the fan


def test_helper_usage_and_header_mention_new_commands():
    text = open(HELPER_PATH).read()
    usage = next(l for l in text.splitlines() if l.startswith("usage()"))
    assert "unpin" in usage and "fan auto" in usage and "fan off" in usage
    header = text.split("set -eu")[0]
    assert "unpin" in header and "fan auto" in header and "fan off" in header
