# tests/test_pinning.py
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pinning
from pinning import (ALLOWED_MHZ, apply, clock_errors, new_undervoltage, status,
                     throttle_errors, wait_fan_full)


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
