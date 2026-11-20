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
    def __init__(self, run_rc=None, pi_up=True, temps=("temp=42.0'C",), ping=False,
                 pi_free_kb=100 * 1024 ** 2):
        self.run_rc = dict(run_rc or {})    # cell_id -> [rc per attempt]
        self.pi_up_ = pi_up
        self.temps = list(temps)
        self.ping_ = ping
        self.pi_free_kb = pi_free_kb
        self.cmds = []

    def ssh(self, cmd, timeout):
        self.cmds.append(("ssh", cmd))
        if cmd.startswith("df -Pk"):
            return 0, ("bash: warning: setlocale: LC_ALL\nFilesystem 1024-blocks Used Available "
                       f"Capacity Mounted on\n/dev/sda2 500000000 1 {self.pi_free_kb} 1% /\n")
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


def _rising_status(extra=None, start=1000):
    """A /status that behaves like a live recorder: n_total grows with every read."""
    n = {"v": start}

    def status():
        n["v"] += 10000
        return {"recording": True, "run_dir": "data/x", "n_total": n["v"], **(extra or {})}
    return status


def _orch(tmp_path, sh, verify_ok=True, **kw):
    calls = {"align": [], "verify": []}
    kw.setdefault("local_free_fn", lambda p: 100.0)

    def align_fn(rec, pi, out):
        calls["align"].append(str(pi))
        return {"max_residual_ms": 0.2, "slip_warning": False, "n_edges": 12}

    def verify_fn(d, mhz):
        calls["verify"].append((str(d), mhz))
        return {"ok": verify_ok, "errors": [] if verify_ok else ["bad"], "warnings": []}

    o = ex.Orchestrator(tmp_path / "runs", tmp_path / "runs" / "results.jsonl", sh=sh,
                        status_fn=_rising_status(),
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
    assert "recorder.py --vout 5.35 --current-limit 6" in runs[2]
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
        if calls["n"] > 2:
            raise OSError("connection refused")
        return {"recording": True, "run_dir": "data/x", "n_total": calls["n"] * 10000}

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
        if calls["n"] == 3:
            raise OSError("gone")
        return {"recording": True, "run_dir": "data/x", "n_total": calls["n"] * 10000}

    o, _ = _orch(tmp_path, FakeShell())
    o.status_fn = status
    r = o.run_cell(_cell("a_2400_r00"), 1)
    assert r["stage"] == "recorder" and "gone" in r["error"]


def test_exit_code_requires_every_cell_ok():
    assert ex.exit_code({"cells": 3, "ok": 3, "aborted": None}) == 0
    assert ex.exit_code({"cells": 3, "ok": 2, "aborted": None}) != 0
    assert ex.exit_code({"cells": 3, "ok": 0, "aborted": None}) != 0
    assert ex.exit_code({"cells": 3, "ok": 3, "aborted": "pi_down: x"}) != 0


def test_parse_git_head_ignores_ssh_warnings():
    warn = "bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)"
    assert ex.parse_git_head(f"d2d6971\n{warn}\n") == "d2d6971"
    assert ex.parse_git_head(f"{warn}\nd2d6971\n") == "d2d6971"
    assert ex.parse_git_head(warn) is None


# ---- pi_subdir / run_timeout_s (used by lab.py) ---------------------------------------------
def test_pi_command_for_the_matrix_is_unchanged():
    c = _cell("lm16_2400_r00")
    assert ex.pi_command(c) == (
        "cd Desktop/power-measurement-pi && mkdir -p runs/matrix && rm -rf runs/matrix/lm16_2400_r00"
        " && { [ ! -f runs/matrix/lm16_2400_r00.log ] || mv runs/matrix/lm16_2400_r00.log "
        "runs/matrix/lm16_2400_r00.log.$(date +%s); } && ~/.local/bin/uv run --frozen python "
        "src/run.py --task lm --model minigpt-explicit --cpu-mhz 2400 "
        "--out runs/matrix/lm16_2400_r00 > runs/matrix/lm16_2400_r00.log 2>&1")


def test_pi_command_custom_subdir():
    cmd = ex.pi_command(_cell("x1"), "runs/lab")
    assert "mkdir -p runs/lab && rm -rf runs/lab/x1 &&" in cmd
    assert "mv runs/lab/x1.log runs/lab/x1.log.$(date +%s)" in cmd
    assert cmd.endswith("--out runs/lab/x1 > runs/lab/x1.log 2>&1") and "runs/matrix" not in cmd


class _SubdirShell(FakeShell):
    def __init__(self):
        super().__init__()
        self.timeouts = []

    def ssh(self, cmd, timeout):
        if "src/run.py" in cmd:
            self.cmds.append(("ssh", cmd))
            self.timeouts.append(timeout)
            return 0, "out"
        return super().ssh(cmd, timeout)


def test_orchestrator_custom_subdir_and_timeout(tmp_path, monkeypatch):
    sh = _SubdirShell()
    o, _ = _orch(tmp_path, sh, pi_subdir="runs/lab", run_timeout_s=123)
    assert o.run_cell(_cell("x1"), 1)["status"] == "ok"
    run = [c for k, c in sh.cmds if k == "ssh" and "src/run.py" in c]
    assert len(run) == 1 and "--out runs/lab/x1 " in run[0] and sh.timeouts == [123]
    srcs = [c[2] for k, c in sh.cmds if k == "run" and c[0] == "rsync"]
    assert srcs == [f"{ex.PI}:{ex.PI_DIR}/runs/lab/x1.log", f"{ex.PI}:{ex.PI_DIR}/runs/lab/x1/"]


def test_orchestrator_default_timeout_read_at_call_time(tmp_path, monkeypatch):
    sh = _SubdirShell()
    o, _ = _orch(tmp_path, sh, pi_subdir="runs/lab")
    monkeypatch.setattr(ex, "RUN_TIMEOUT_S", 77)
    o.run_cell(_cell("x1"), 1)
    assert sh.timeouts == [77]


# ---- Task 11: timeouts, thermal abort, disk and recorder pre-checks -------------------------------
class _RcShell(_SubdirShell):
    """Like _SubdirShell, but run.py answers with a chosen rc and thermal_abort.json is rsynced."""

    def __init__(self, rc, abort_json=None, **kw):
        super().__init__()
        self.rc, self.abort_json = rc, abort_json

    def ssh(self, cmd, timeout):
        if "src/run.py" in cmd:
            self.cmds.append(("ssh", cmd))
            self.timeouts.append(timeout)
            return self.rc, "out"
        return super().ssh(cmd, timeout)

    def run(self, cmd, timeout, cwd=None):
        rc = super().run(cmd, timeout, cwd)
        if cmd[0] == "rsync" and cmd[-1].endswith("/") and self.abort_json is not None:
            from pathlib import Path
            (Path(cmd[-1]) / "thermal_abort.json").write_text(self.abort_json)
        return rc


def test_cell_timeout_overrides_orchestrator_and_default(tmp_path):
    sh = _SubdirShell()
    o, _ = _orch(tmp_path, sh, run_timeout_s=123)
    o.run_cell({**_cell("a"), "timeout_s": 7200}, 1)
    o.run_cell(_cell("b"), 1)
    assert sh.timeouts == [7200, 123]
    sh2 = _SubdirShell()
    o2, _ = _orch(tmp_path / "two", sh2)
    o2.run_cell({**_cell("c"), "timeout_s": 5400}, 1)
    o2.run_cell(_cell("d"), 1)
    assert sh2.timeouts == [5400, ex.RUN_TIMEOUT_S]


@pytest.mark.parametrize("rc", [3, 4])
def test_thermal_exit_code_is_thermal_abort_with_json_and_not_retried(tmp_path, rc):
    payload = {"t_real_ns": 1, "temp_C": 85.4, "level": "abort", "phase": "measure"}
    sh = _RcShell(rc, json.dumps(payload))
    o, calls = _orch(tmp_path, sh)
    man = [_cell("a_2400_r00"), _cell("b_2400_r00")]
    s = o.run(man, max_attempts=2)
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert [(r["cell_id"], r["stage"], r["attempt"]) for r in rows] == [
        ("a_2400_r00", "thermal_abort", 1), ("b_2400_r00", "thermal_abort", 1)]
    assert json.loads(rows[0]["error"]) == payload and rows[0]["run_rc"] == rc
    assert s["aborted"] is None and calls["align"] == []      # matrix continued; nothing aligned
    assert ex.cell_state(rows)["a_2400_r00"]["attempts"] == 1
    # the start-of-cell thermal gate still ran before the second cell
    assert sum(1 for k, c in sh.cmds if k == "ssh" and c.startswith("vcgencmd")) == 2


def test_thermal_abort_without_json_still_reports_stage(tmp_path):
    o, _ = _orch(tmp_path, _RcShell(3))
    r = o.run_cell(_cell("a"), 1)
    assert r["stage"] == "thermal_abort" and "thermal_abort.json" in r["error"]


def test_other_nonzero_rc_is_still_run_stage(tmp_path):
    r = _orch(tmp_path, _RcShell(2))[0].run_cell(_cell("a"), 1)
    assert r["stage"] == "run"


def test_soak_cells_are_not_gated_mid_run(tmp_path):
    # temperature is read only at the start of a cell, whatever the family
    sh = _SubdirShell()
    o, _ = _orch(tmp_path, sh)
    o.run_cell({**_cell("a"), "family": "soak"}, 1)
    assert sum(1 for k, c in sh.cmds if k == "ssh" and c.startswith("vcgencmd")) == 1


def test_disk_check_thresholds(tmp_path):
    gb = 1024 ** 2          # KiB per GiB
    o, _ = _orch(tmp_path, FakeShell(pi_free_kb=5 * gb), local_free_fn=lambda p: 10.0)
    assert o.disk_check(_cell("a")) is None
    soak = {**_cell("a"), "family": "soak"}
    assert "EliteBook" in o.disk_check(soak) and "need 12" in o.disk_check(soak)
    o, _ = _orch(tmp_path, FakeShell(pi_free_kb=5 * gb), local_free_fn=lambda p: 12.0)
    assert "Pi" in o.disk_check(soak) and "need 7" in o.disk_check(soak)
    o, _ = _orch(tmp_path, FakeShell(pi_free_kb=7 * gb), local_free_fn=lambda p: 12.0)
    assert o.disk_check(soak) is None
    o, _ = _orch(tmp_path, FakeShell(pi_free_kb=5 * gb - 1), local_free_fn=lambda p: 50.0)
    assert "Pi" in o.disk_check(_cell("a"))
    o, _ = _orch(tmp_path, FakeShell(), local_free_fn=lambda p: 9.9)
    assert "need 10" in o.disk_check(_cell("a"))


def test_disk_failure_aborts_matrix_before_run(tmp_path):
    sh = FakeShell(pi_free_kb=1024)
    o, calls = _orch(tmp_path, sh)
    s = o.run([_cell("a_2400_r00"), _cell("b_2400_r00")])
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert len(rows) == 1 and rows[0]["stage"] == "disk" and s["aborted"].startswith("disk")
    assert not any("src/run.py" in c for k, c in sh.cmds if k == "ssh")
    assert not any("vcgencmd" in c for k, c in sh.cmds if k == "ssh")


def test_disk_check_fails_closed_when_df_unreadable(tmp_path):
    class Garbled(FakeShell):
        def ssh(self, cmd, timeout):
            return (0, "garbage") if cmd.startswith("df") else super().ssh(cmd, timeout)
    r = _orch(tmp_path, Garbled())[0].run_cell(_cell("a"), 1)
    assert r["stage"] == "disk" and "could not read" in r["error"]


def test_parse_df_avail_gb():
    out = "bash: warning: setlocale\nFilesystem 1024-blocks Used Available Capacity Mounted on\n" \
          "/dev/sda2 100 1 2097152 3% /home\n"
    assert ex.parse_df_avail_gb(out) == 2.0
    assert ex.parse_df_avail_gb("garbage") is None


def test_local_free_gb_walks_up_to_existing_parent(tmp_path):
    assert ex.local_free_gb(tmp_path / "not" / "yet") > 0


CEIL = 2 ** 32 - 1


def test_recorder_stalled_pure():
    assert ex.recorder_stalled({"n_total": 100}, {"n_total": 200}) is None
    msg = ex.recorder_stalled({"n_total": CEIL}, {"n_total": CEIL})
    assert "not sampling" in msg and "README section 8" in msg
    assert "not sampling" in ex.recorder_stalled({"n_total": 5}, {"n_total": 5})
    assert "not sampling" in ex.recorder_stalled({"n_total": 9}, {"n_total": 5})
    limit = CEIL - 5000 * 3600
    assert ex.recorder_stalled({"n_total": limit - 20}, {"n_total": limit - 1}) is None
    near = ex.recorder_stalled({"n_total": limit - 1}, {"n_total": limit})
    assert "ceiling" in near and "README section 8" in near
    assert "n_total" in ex.recorder_stalled({}, {"n_total": 5})
    assert "n_total" in ex.recorder_stalled({"n_total": 5}, {"n_total": None})


def _flat_status(n):
    return lambda: {"recording": True, "run_dir": "data/x", "n_total": n}


def test_stalled_recorder_fails_cell_and_aborts_matrix_before_anything_else(tmp_path):
    sh = FakeShell()
    o, _ = _orch(tmp_path, sh)
    o.status_fn = _flat_status(CEIL)
    s = o.run([_cell("a_2400_r00"), _cell("b_2400_r00")])
    rows = ex.load_jsonl(tmp_path / "runs" / "results.jsonl")
    assert len(rows) == 1 and rows[0]["stage"] == "recorder" and "README section 8" in rows[0]["error"]
    assert s["aborted"].startswith("recorder") and sh.cmds == []


def test_status_is_read_twice_with_the_injected_sleep(tmp_path):
    sleeps, reads = [], []
    sh = FakeShell()
    o, _ = _orch(tmp_path, sh)
    base = _rising_status()
    o.status_fn = lambda: (reads.append(1), base())[1]
    o.sleep = sleeps.append
    assert o.run_cell(_cell("a"), 1)["status"] == "ok"
    assert sleeps[0] == 2 and len(reads) >= 2


def test_second_status_read_failure_is_a_recorder_result(tmp_path):
    calls = {"n": 0}

    def status():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("reset")
        return {"recording": True, "run_dir": "data/x", "n_total": 10}
    o, _ = _orch(tmp_path, FakeShell())
    o.status_fn = status
    r = o.run_cell(_cell("a"), 1)
    assert r["stage"] == "recorder" and "reset" in r["error"]


def test_default_local_free_fn_is_resolved_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "local_free_gb", lambda p: 1.0)
    o = ex.Orchestrator(tmp_path, tmp_path / "r.jsonl", sh=FakeShell(), log=lambda m: None)
    assert "EliteBook" in o.disk_check(_cell("a"))
