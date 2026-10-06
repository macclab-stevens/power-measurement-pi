# tests/test_pinning.py
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pinning import ALLOWED_MHZ, apply, clock_errors, wait_fan_full


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
