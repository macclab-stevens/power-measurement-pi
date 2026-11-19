# tests/test_run_v2_flags.py -- dataset v2 workload flags in src/run.py (pure helpers + argparse)
import contextlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import torch

import run


def _parse(*argv):
    return run.parse_args(list(argv))


def _err(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        run.parse_args(list(argv))
    assert e.value.code == 2
    return capsys.readouterr().err


# ---- argparse -------------------------------------------------------------------------------
def test_v1_defaults_unchanged():
    a = _parse()
    assert (a.threads, a.batch, a.governor, a.fan, a.fan_full) == (4, None, "pinned", None, False)
    assert (a.holdout, a.serve_fps, a.soak_minutes, a.input_image) == ("none", None, None, None)
    assert (a.frames_source, a.stream_dir) == ("repeat", "inputs/stream50")


def test_fan_full_conflicts_with_other_fan_modes(capsys):
    assert "--fan-full" in _err(capsys, "--fan-full", "--fan", "off")
    assert "--fan-full" in _err(capsys, "--fan-full", "--fan", "auto")
    a = _parse("--fan-full", "--fan", "full")   # same value is not a conflict
    assert run.resolve_fan_mode(a.fan, a.fan_full) == "full"


def test_governor_ondemand_needs_cpu_mhz(capsys):
    assert "--cpu-mhz" in _err(capsys, "--governor", "ondemand")
    assert "--cpu-mhz" in _err(capsys, "--governor", "ondemand", "--cpu-mhz", "1000")
    assert _parse("--governor", "ondemand", "--cpu-mhz", "2400").governor == "ondemand"


def test_other_flag_conflicts(capsys):
    assert "--serve" in _err(capsys, "--serve-fps", "5")
    assert "image" in _err(capsys, "--task", "lm", "--soak-minutes", "30")
    assert "stream" in _err(capsys, "--frames-source", "stream", "--input-image", "x.jpg")
    assert "--holdout" in _err(capsys, "--holdout", "bogus")


@pytest.mark.parametrize("fan,fan_full,want", [
    (None, False, "none"), (None, True, "full"), ("auto", False, "auto"),
    ("off", False, "off"), ("full", False, "full")])
def test_resolve_fan_mode(fan, fan_full, want):
    assert run.resolve_fan_mode(fan, fan_full) == want


@pytest.mark.parametrize("task,batch,want", [
    ("train", None, 8), ("image", None, 1), ("lm", None, 1), ("image", 8, 8), ("train", 4, 4)])
def test_resolve_batch(task, batch, want):
    assert run.resolve_batch(task, batch) == want


# ---- dataset_version ------------------------------------------------------------------------
def test_dataset_version_v1_command_lines_stay_v1():
    assert run.dataset_version(_parse()) == 1
    # v1 flags, plus threads/batch/imgsz, plus perf counters (default on) do not make v2
    a = _parse("--threads", "2", "--batch", "8", "--imgsz", "480", "--fan-full",
               "--cpu-mhz", "2400", "--serve", "5", "--perf-counters", "on")
    assert run.dataset_version(a) == 1
    assert run.dataset_version(_parse("--perf-counters", "off")) == 1


@pytest.mark.parametrize("argv", [
    ["--holdout", "primary-s"], ["--holdout", "secondary-m"],
    ["--governor", "ondemand", "--cpu-mhz", "2400"], ["--fan", "auto"], ["--fan", "off"],
    ["--serve", "5", "--serve-fps", "5"], ["--soak-minutes", "30"],
    ["--frames-source", "stream"], ["--input-image", "bus.jpg"]])
def test_dataset_version_v2_flags(argv):
    assert run.dataset_version(_parse(*argv)) == 2


def test_dataset_version_counts_flags_added_by_later_tasks():
    a = _parse()
    for name, value in [("quant", "int8s"), ("prune", 0.3), ("concurrent", "conc2")]:
        assert run.dataset_version(a) == 1
        b = SimpleNamespace(**vars(a), **{name: value})
        assert run.dataset_version(b) == 2
    assert run.dataset_version(SimpleNamespace(holdout=None)) == 1   # bare namespace: v1


# ---- serve scheduling -----------------------------------------------------------------------
def test_serve_wait_on_time_idles_for_remaining():
    idle, missed = run.serve_wait(100.0, 1, 5.0, now=100.05)   # due 100.2
    assert idle == pytest.approx(0.15) and missed is False


def test_serve_wait_late_starts_immediately_and_counts_missed():
    idle, missed = run.serve_wait(100.0, 3, 2.0, now=101.8)    # due 101.5
    assert idle == 0.0 and missed is True


def test_serve_wait_exactly_due_is_not_missed():
    idle, missed = run.serve_wait(0.0, 2, 4.0, now=0.5)
    assert idle == 0.0 and missed is False


def test_serve_wait_schedule_does_not_drift():
    # k/F is relative to t0, so one slow request does not push later deadlines back
    assert run.serve_wait(10.0, 4, 1.0, now=12.0)[0] == pytest.approx(2.0)


# ---- latency --------------------------------------------------------------------------------
def test_latency_stats():
    s = run.latency_stats([10.0, 20.0, 30.0, 40.0])
    assert (s["mean"], s["median"], s["n"]) == (25.0, 25.0, 4)
    assert s["p95"] == pytest.approx(38.5)
    assert run.latency_stats([7.0])["p95"] == 7.0
    assert run.latency_stats([]) is None


# ---- measured loop (frame 0 profiled, soak, thermal) -----------------------------------------
class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _FakeTask:
    def __init__(self, clock, dt):
        self.clock, self.dt, self.calls = clock, dt, 0

    @contextlib.contextmanager
    def step(self, name):
        yield

    def run_one(self, mode):
        self.calls += 1
        torch.ones(2) + 1   # something for the profiler to see
        self.clock.t += self.dt


class _FakeProf:
    def __init__(self):
        self.frames = []

    def set_frame(self, f):
        self.frames.append(f)


def test_measured_loop_frames_excludes_frame0_from_latency():
    clk = _FakeClock()
    task, prof, checks = _FakeTask(clk, 0.25), _FakeProf(), []
    done, lat, ops = run.measured_loop(task, prof, "forward", 4, None,
                                       lambda: checks.append(1), clock=clk)
    assert done == 4 and task.calls == 4 and prof.frames == [0, 1, 2, 3]
    assert lat == [250.0, 250.0, 250.0] and len(checks) == 4
    assert ops and all(r["mode"] == "forward" for r in ops)


def test_measured_loop_zero_frames_does_nothing_as_in_v1():
    clk = _FakeClock()
    task = _FakeTask(clk, 0.1)
    assert run.measured_loop(task, _FakeProf(), "forward", 0, None, lambda: None, clock=clk)[:2] \
        == (0, [])
    assert task.calls == 0


def test_measured_loop_soak_runs_until_deadline_and_ignores_frames():
    clk = _FakeClock()
    task, prof, checks = _FakeTask(clk, 1.0), _FakeProf(), []
    # 0.1 min = 6 s: frame 0 (1 s) then iterations until clock >= 6 s
    done, lat, ops = run.measured_loop(task, prof, "forward", 3, 0.1, lambda: checks.append(1),
                                       clock=clk)
    assert done == 6 and prof.frames == list(range(6)) and len(lat) == 5
    assert len(checks) == 6 and ops


def test_measured_loop_thermal_abort_propagates():
    clk = _FakeClock()

    def boom():
        raise run.ThermalAbort({"level": "abort", "temp_C": 86.0})
    with pytest.raises(run.ThermalAbort):
        run.measured_loop(_FakeTask(clk, 0.1), _FakeProf(), "forward", 3, None, boom, clock=clk)


# ---- threads --------------------------------------------------------------------------------
def test_apply_threads_sets_torch_opencv_and_omp_env(monkeypatch):
    import cv2
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    t0, c0 = torch.get_num_threads(), cv2.getNumThreads()
    try:
        d = run.apply_threads(2)
        assert d["torch"] == 2 == torch.get_num_threads()
        assert d["omp_env"] == "2" and os.environ["OMP_NUM_THREADS"] == "2"
        assert d["opencv"] == cv2.getNumThreads() and isinstance(d["opencv"], int)
    finally:
        torch.set_num_threads(t0)
        cv2.setNumThreads(c0)
