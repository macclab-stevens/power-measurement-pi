# host/tests/test_experiment.py
import json
import shlex

import pytest

import experiment as ex


def _cell(cid, rnd=0, mhz=2400):
    return {"cell_id": cid, "model_key": cid.split("_")[0], "cpu_mhz": mhz, "repeat": 0,
            "round": rnd, "args": ["--task", "lm", "--model", "minigpt-explicit",
                                   "--cpu-mhz", str(mhz)]}


class FakeShell:
    def __init__(self, run_rc=None, pi_up=True, temps=("temp=42.0'C",), ping=False):
        self.run_rc = dict(run_rc or {})    # cell_id -> [rc per attempt]
        self.pi_up_ = pi_up
        self.temps = list(temps)
        self.ping_ = ping
        self.cmds = []

    def ssh(self, cmd, timeout):
        self.cmds.append(("ssh", cmd))
        if cmd == "true":
            return (0, "") if self._up() else (255, "down")
        if cmd.startswith("vcgencmd"):
            t = self.temps.pop(0) if len(self.temps) > 1 else self.temps[0]
            return 0, "bash: warning: setlocale: LC_ALL\n" + t
        if "src/run.py" in cmd:
            cid = cmd.split("--out runs/matrix/")[1].split()[0]
            rcs = self.run_rc.get(cid, [0])
            return (rcs.pop(0) if len(rcs) > 1 else rcs[0]), "out"
        return 0, ""

    def _up(self):
        if isinstance(self.pi_up_, list):
            return self.pi_up_.pop(0) if len(self.pi_up_) > 1 else self.pi_up_[0]
        return self.pi_up_

    def run(self, cmd, timeout, cwd=None):
        self.cmds.append(("run", cmd))
        if cmd[0] == "rsync" and cmd[-1].endswith("/"):
            from pathlib import Path
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, ""

    def popen(self, cmd, out_path):
        class P:
            def terminate(self):
                pass
        class F:
            def close(self):
                pass
        return P(), F()

    def ping(self):
        return self.ping_


def _orch(tmp_path, sh, verify_ok=True, **kw):
    calls = {"align": [], "verify": []}

    def align_fn(rec, pi, out):
        calls["align"].append(str(pi))
        return {"max_residual_ms": 0.2, "slip_warning": False, "n_edges": 12}

    def verify_fn(d, mhz):
        calls["verify"].append((str(d), mhz))
        return {"ok": verify_ok, "errors": [] if verify_ok else ["bad"], "warnings": []}

    o = ex.Orchestrator(tmp_path / "runs", tmp_path / "runs" / "results.jsonl", sh=sh,
                        status_fn=lambda: {"recording": True, "run_dir": "data/x"},
                        align_fn=align_fn, verify_fn=verify_fn, sleep=lambda s: None,
                        log=lambda m: None, **kw)
    return o, calls


def test_pi_command_quotes_args_and_logs_to_file():
    c = _cell("lm16_2400_r00")
    c["args"].append("a b")
    cmd = ex.pi_command(c)
    assert "src/run.py" in cmd and shlex.quote("a b") in cmd
    assert cmd.endswith("--out runs/matrix/lm16_2400_r00 > runs/matrix/lm16_2400_r00.log 2>&1")
    assert "| head" not in cmd


def test_parse_temp_ignores_noise():
    assert ex.parse_temp("bash: warning: setlocale\ntemp=48.3'C\n") == 48.3
    assert ex.parse_temp("garbage") is None


def test_ok_cells_written_and_skipped_on_resume(tmp_path):
    sh = FakeShell()
    o, calls = _orch(tmp_path, sh)
    man = [_cell("a_2400_r00"), _cell("b_2400_r00", rnd=1)]
    s = o.run(man)
    assert s == {"cells": 2, "ok": 2, "aborted": None}
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert calls["verify"][0][1] == 2400
    o2, calls2 = _orch(tmp_path, FakeShell())
    o2.run(man)
    assert calls2["align"] == [] and len(ex.load_jsonl(tmp_path / "runs" / "results.jsonl")) == 2


def test_failed_cell_retried_once_at_end_of_round(tmp_path):
    sh = FakeShell(run_rc={"a_2400_r00": [1, 1]})
    o, _ = _orch(tmp_path, sh)
    man = [_cell("a_2400_r00"), _cell("b_2400_r00"), _cell("c_2400_r00", rnd=1)]
    s = o.run(man)
    order = [r["cell_id"] for r in ex.load_jsonl(tmp_path / "runs" / "results.jsonl")]
    assert order == ["a_2400_r00", "b_2400_r00", "a_2400_r00", "c_2400_r00"]
    assert s["ok"] == 2
    last = ex.cell_state(ex.load_jsonl(tmp_path / "runs" / "results.jsonl"))["a_2400_r00"]
    assert last["attempts"] == 2 and last["last"]["stage"] == "run"


def test_retry_succeeds(tmp_path):
    sh = FakeShell(run_rc={"a_2400_r00": [1, 0]})
    o, _ = _orch(tmp_path, sh)
    assert o.run([_cell("a_2400_r00")])["ok"] == 1


def test_verify_failure_fails_cell(tmp_path):
    o, _ = _orch(tmp_path, FakeShell(), verify_ok=False)
    s = o.run([_cell("a_2400_r00")], max_attempts=1)
    r = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")[0]
    assert s["ok"] == 0 and r["stage"] == "verify" and "bad" in r["error"]


def test_thermal_gate_waits_then_passes(tmp_path):
    sh = FakeShell(temps=["temp=55.0'C", "temp=52.0'C", "temp=49.0'C"])
    o, _ = _orch(tmp_path, sh)
    assert o.thermal_gate() is None
    assert sum(1 for k, c in sh.cmds if k == "ssh" and c.startswith("vcgencmd")) == 3


def test_pi_down_without_power_cycle_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    o, _ = _orch(tmp_path, FakeShell(pi_up=False))
    s = o.run([_cell("a_2400_r00"), _cell("b_2400_r00")])
    assert s["aborted"].startswith("pi_down")
    assert len(ex.load_jsonl(tmp_path / "runs" / "results.jsonl")) == 1


def test_power_cycle_refused_when_current_is_not_halted(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    sh = FakeShell(pi_up=False)
    o, _ = _orch(tmp_path, sh, auto_power_cycle=3, current_fn=lambda d: (0.9, 1.5))
    err = o.recover_pi()
    assert "halt not confirmed" in err and o.cycles_left == 3
    assert not any(k == "run" and "vout_off.py" in " ".join(c) for k, c in sh.cmds)


def test_power_cycle_refused_when_ping_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    o, _ = _orch(tmp_path, FakeShell(pi_up=False, ping=True), auto_power_cycle=3,
                 current_fn=lambda d: (0.28, 0.3))
    assert "ping" in o.recover_pi()


def test_power_cycle_when_halted(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    sh = FakeShell(pi_up=[False, False, True])
    (tmp_path / "runs").mkdir()
    o, _ = _orch(tmp_path, sh, auto_power_cycle=1, current_fn=lambda d: (0.28, 0.31))
    assert o.recover_pi() is None and o.cycles_left == 0
    runs = [" ".join(map(str, c)) for k, c in sh.cmds if k == "run"]
    assert "stop monsoon-recorder" in runs[0]
    assert "vout_off.py --confirm-pi-halted" in runs[1]
    assert "recorder.py --vout 5.25 --current-limit 6" in runs[2]
    assert ex.load_jsonl(tmp_path / "runs" / "power_cycles.jsonl")[0]["mean_A"] == 0.28


def test_mid_run_crash_is_retryable_not_abort(tmp_path):
    # run fails and the Pi is unreachable right after; next check finds it back up
    sh = FakeShell(run_rc={"a_2400_r00": [255, 0]}, pi_up=[True, False, True])
    o, _ = _orch(tmp_path, sh)
    s = o.run([_cell("a_2400_r00")])
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert rows[0]["stage"] == "pi_crash" and s["aborted"] is None and s["ok"] == 1


def test_status_report(tmp_path):
    man = [_cell("a_2400_r00"), _cell("b_2400_r00"), _cell("c_2400_r00")]
    res = [{"cell_id": "a_2400_r00", "status": "ok", "duration_s": 240,
            "align": {"slip_warning": True}},
           {"cell_id": "b_2400_r00", "status": "fail", "stage": "align"}]
    txt = ex.status_report(man, res)
    assert "ok 1" in txt and "failed 1" in txt and "not run 1" in txt
    assert "failed at align: b_2400_r00" in txt and "slip_warning: 1" in txt


# ---- PR #1 review fixes ---------------------------------------------------------------------
@pytest.mark.parametrize("cur", [(float("nan"), float("nan")), (0.2, float("nan")),
                                 (float("nan"), 0.3), (-0.1, 0.2), (0.2, float("inf"))])
def test_power_cycle_fails_closed_on_invalid_current(tmp_path, monkeypatch, cur):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    sh = FakeShell(pi_up=False)
    o, _ = _orch(tmp_path, sh, auto_power_cycle=3, current_fn=lambda d: cur)
    err = o.recover_pi()
    assert err and "not power-cycling" in err and o.cycles_left == 3
    assert not any(k == "run" for k, _ in sh.cmds)


def test_halted_current_ignores_nan_and_fails_on_empty(tmp_path, monkeypatch):
    import numpy as np
    import store
    data = np.full((25000, 2), 0.28)
    data[:100] = np.nan
    monkeypatch.setattr(store, "open_samples", lambda d: data)
    m, mx = ex.halted_current("x")
    assert abs(m - 0.28) < 1e-9 and abs(mx - 0.28) < 1e-9
    monkeypatch.setattr(store, "open_samples", lambda d: np.full((25000, 2), np.nan))
    m, mx = ex.halted_current("x")
    assert m != m and mx != mx          # NaN: recover_pi then refuses to cycle


def test_recorder_failure_during_recovery_is_a_recorder_result(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "PI_DOWN_S", 0)
    calls = {"n": 0}

    def status():
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("connection refused")
        return {"recording": True, "run_dir": "data/x"}

    sh = FakeShell(pi_up=False)
    o, _ = _orch(tmp_path, sh, auto_power_cycle=3)
    o.status_fn = status
    s = o.run([_cell("a_2400_r00"), _cell("b_2400_r00")])
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert rows[0]["stage"] == "recorder" and "connection refused" in rows[0]["error"]
    assert s["aborted"].startswith("recorder")


def test_recorder_failure_after_recovery_is_a_recorder_result(tmp_path):
    calls = {"n": 0}

    def status():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("gone")
        return {"recording": True, "run_dir": "data/x"}

    o, _ = _orch(tmp_path, FakeShell())
    o.status_fn = status
    r = o.run_cell(_cell("a_2400_r00"), 1)
    assert r["stage"] == "recorder" and "gone" in r["error"]


def test_exit_code_requires_every_cell_ok():
    assert ex.exit_code({"cells": 3, "ok": 3, "aborted": None}) == 0
    assert ex.exit_code({"cells": 3, "ok": 2, "aborted": None}) != 0
    assert ex.exit_code({"cells": 3, "ok": 0, "aborted": None}) != 0
    assert ex.exit_code({"cells": 3, "ok": 3, "aborted": "pi_down: x"}) != 0
