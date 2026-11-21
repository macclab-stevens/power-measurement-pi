# host/tests/test_lab.py - offline tests for the one-command launcher (no real shell, ever)
import datetime
import json

import pytest

import experiment as ex
import lab
import matrix

SETLOCALE = "bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)"
HEAD = "a878d93" + "0" * 33
NOW = datetime.datetime(2026, 10, 16, 14, 2, 33)
STATUS = {"recording": True, "serial": 36517, "vout_V": 5.4, "sample_hz": 4992.4,
          "n_total": int(4992.4 * 86400 * 2), "run_dir": "data/20261014T140000"}


@pytest.fixture(autouse=True)
def no_real_shell(monkeypatch, tmp_path):
    """A test that forgets to pass its fake shell must fail, never reach the lab."""
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("a test tried to construct a real Shell")
    monkeypatch.setattr(ex, "Shell", Boom)
    monkeypatch.setattr(lab, "LabShell", Boom)
    monkeypatch.setattr(ex, "local_free_gb", lambda p: 100.0)   # never depend on the real disk
    monkeypatch.setattr(lab, "LOCAL_LAB", tmp_path / "runs" / "lab")
    monkeypatch.setattr(lab, "MATRIX_RESULTS", tmp_path / "runs" / "matrix" / "results.jsonl")


def live_status():
    """/status of a recorder that is sampling: n_total grows with every read (experiment.py
    refuses to start a cell on a recorder whose n_total stands still)."""
    n = {"v": STATUS["n_total"]}

    def status():
        n["v"] += 10000
        return dict(STATUS, n_total=n["v"])
    return status


def _run_py_cmd(c):
    """The cell's run.py command (not experiment.py's pgrep for a stale run.py)."""
    return c.startswith("cd ") and "src/run.py" in c


def default_rules():
    """(predicate, answer): predicate is a substring or a callable on the command text; answer is
    (rc, out), a callable(cmd) -> (rc, out), or a list of those used one after another."""
    return [
        (lambda c: c == "true", (0, "")),
        ("vcgencmd", (0, "temp=42.0'C\nthrottled=0x0")),
        ("df -Pk", (0, "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                       "/dev/sda2 500000000 1000 400000000 1% /home")),
        ("pi-power-pin status", (0, "governor=performance min_kHz=2400000 max_kHz=2400000 "
                                    "cur_kHz=2400000 fan_state=4/4 fan_rpm=7000 temp_mC=42000")),
        ("pi-power-pin restore", (0, "governor=ondemand")),
        ("grep -c unpin", (0, "4")),
        ("operstate", (0, "down")),
        ("command -v perf", (0, "/usr/bin/perf")),
        ("rev-parse HEAD", (0, HEAD)),
        ("[s]rc/run.py", (1, "")),
        ("pytho[n] src/run.py", (1, "")),      # experiment.py's busy check (before "src/run.py")
        ("[p]ytest", (1, "")),
        ("is-active exp-matrix", (3, "inactive")),
        ("[e]xperiment.py run", (1, "")),
        ("test -e", (1, "")),
        ("test -f", (0, "")),
        (_run_py_cmd, (0, "[ok] dataset written")),
        ("pytest -q", (0, "...\n3 passed in 4.21s")),
    ]


class ScriptedShell:
    def __init__(self, rules=(), ping=False):
        self.rules = [*rules, *default_rules()]
        self.ping_ = ping
        self.cmds = []          # (kind, command text, timeout)

    def _answer(self, text):
        for pred, ans in self.rules:
            if pred(text) if callable(pred) else pred in text:
                if isinstance(ans, list):
                    ans = ans.pop(0) if len(ans) > 1 else ans[0]
                return ans(text) if callable(ans) else ans
        return 0, ""

    def ssh(self, cmd, timeout):
        self.cmds.append(("ssh", cmd, timeout))
        rc, out = self._answer(cmd)
        return rc, f"{SETLOCALE}\n{out}\n{SETLOCALE}\n"

    def run(self, cmd, timeout, cwd=None):
        text = " ".join(map(str, cmd))
        self.cmds.append(("run", text, timeout))
        if cmd[0] == "rsync":
            if cmd[-1].endswith("/"):
                from pathlib import Path
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            return 0, ""
        return self._answer(text)

    def popen(self, cmd, out_path):
        self.cmds.append(("popen", " ".join(cmd), None))

        class P:
            def terminate(self):
                pass

        class F:
            def close(self):
                pass
        return P(), F()

    def ping(self):
        self.cmds.append(("ping", "", None))
        return self.ping_

    def sent(self, sub):
        return [c for _, c, _ in self.cmds if sub in c]


def _rc(argv, **kw):
    try:
        return lab.main(argv, **kw)
    except SystemExit as e:
        return e.code


def _jobs(argv):
    return lab.plan_jobs(lab.build_parser().parse_args(["run", *argv]), NOW)


def _fakes(calls, verify_ok=True):
    def align_fn(rec, pi, out):
        calls.setdefault("align", []).append(str(pi))
        (out / "run_meta.json").write_text(json.dumps({
            "task": "image", "model": "yolo11n.pt", "frames_per_mode": 60, "modes": ["forward"],
            "cpu_mhz_pinned": 2400, "fan_full": True, "baseline_W": 4.91,
            "baseline_A_W": 4.90, "baseline_B_W": 4.92, "train_loss": None}))
        (out / "phase_energy.csv").write_text(
            "phase,iter,step,n_samples,n_finite,mean_P_W,ctx_read_frac,t_start_s,t_end_s,"
            "dur_ms,E_mJ\nserve,0,,1000,1000,7.8,0,0,0.2,200.0,1560.0\n")
        return {"max_residual_ms": 0.17, "rms_residual_ms": 0.05, "drift_ppm": 1.2,
                "slip_warning": True, "slip_samples_total": 23, "n_edges": 12}

    def verify_fn(d, mhz):
        calls.setdefault("verify", []).append((str(d), mhz))
        return {"ok": verify_ok, "errors": [] if verify_ok else ["fan_full not set"],
                "warnings": ["slip_warning: 23 samples lost"], "mean_P_W": {"forward": 9.58},
                "step_P_W": {"preprocess": 8.02, "forward": 9.91, "postprocess": 8.15},
                "baseline_P_W": 4.9, "step_coverage": 0.98, "energy_total_J": 1234.5}
    return align_fn, verify_fn


def _run(argv, sh, calls=None, verify_ok=True):
    calls = {} if calls is None else calls
    align_fn, verify_fn = _fakes(calls, verify_ok)
    return _rc(argv, sh=sh, status_fn=live_status(), current_fn=lambda d: (0.29, 0.31),
               align_fn=align_fn, verify_fn=verify_fn, now=lambda: NOW, sleep=lambda s: None)


POWER = ("systemctl --user stop", "start monsoon", "vout_off", "recorder.py", "pkill")


def _no_power_commands(sh):
    for kind, c, _ in sh.cmds:
        assert not any(p in c for p in POWER), c


# ---- presets and arguments ------------------------------------------------------------------
@pytest.mark.parametrize("key", list(matrix.MODELS))
def test_preset_without_overrides_is_exactly_the_matrix_cell(key):
    (job,) = _jobs([key])
    assert job["args"] == matrix._cell(key, 2400, 0, 0)["args"]
    assert job["preset"] == key and job["cpu_mhz"] == 2400


def test_overrides_replace_values_in_place():
    (job,) = _jobs(["yolo11n", "--frames", "5", "--threads", "2", "--cpu-mhz", "1500"])
    a = job["args"]
    assert a.count("--frames") == 1 and a[a.index("--frames") + 1] == "5"
    assert a[a.index("--threads") + 1] == "2" and a[-2:] == ["--cpu-mhz", "1500"]
    assert "--sampler" in a and "--fan-full" in a and "--unpinned" not in a


def test_bench_zero_and_positive_bench():
    (job,) = _jobs(["yolo11n", "--bench", "0"])
    assert "--no-bench" in job["args"] and "--bench" not in job["args"]
    (job,) = _jobs(["train_lm64_adam", "--bench", "1.5"])
    a = job["args"]
    assert "--no-bench" not in a and a[a.index("--bench") + 1] == "1.5"


def test_no_check_image_drops_flag():
    (job,) = _jobs(["yolo11n", "--no-check-image"])
    assert "--check-image" not in job["args"] and "bus.jpg" not in job["args"]


def test_overrides_apply_to_every_preset_and_names_differ():
    jobs = _jobs(["yolo11n", "train_lm64_adam", "train_resnet18_sgd", "--frames", "3"])
    assert [j["preset"] for j in jobs] == ["yolo11n", "train_lm64_adam", "train_resnet18_sgd"]
    assert all(j["args"][j["args"].index("--frames") + 1] == "3" for j in jobs)
    assert len({j["name"] for j in jobs}) == 3


def test_custom_requires_task_and_model(capsys):
    assert _rc(["run", "--task", "image"]) == 2
    assert "--model" in capsys.readouterr().err
    assert _rc(["run", "yolo11n", "--model", "yolov8n.pt"]) == 2
    (job,) = _jobs(["--task", "train", "--model", "torchvision:resnet50", "--batch", "4"])
    assert job["args"][:4] == ["--task", "train", "--model", "torchvision:resnet50"]
    assert "--no-bench" in job["args"] and job["preset"] is None


def test_default_name_sanitized():
    (job,) = _jobs(["--task", "lm", "--model", "hf:org/tiny:v1", "--cpu-mhz", "2000"])
    assert job["name"] == "hf-org-tiny-v1_2000_20261016-140233"
    assert _jobs(["yolo11n"])[0]["name"] == "yolo11n_2400_20261016-140233"


@pytest.mark.parametrize("argv", [["yolo11n", "--cpu-mhz", "2500"],
                                  ["yolo11n", "--cpu-mhz", "1450"],
                                  ["yolo11n", "--name", "../x"], ["yolo11n", "--name", "a/b"],
                                  ["yolo11n", "--name", "-x"],
                                  ["yolo11n", "lm64", "--name", "two"],
                                  ["yolo11"], ["--task", "train", "--model", "minigpt-explicit",
                                               "--batch", "1"]])
def test_bad_run_arguments_are_usage_errors(argv):
    assert _rc(["run", *argv, "--dry-run"]) == 2


# ---- run --------------------------------------------------------------------------------------
def test_dry_run_makes_no_calls(capsys):
    sh = ScriptedShell()
    assert _run(["run", "yolo11n", "lm64", "--dry-run"], sh) == 0
    out = capsys.readouterr().out
    assert sh.cmds == []
    assert "--out runs/lab/yolo11n_2400_20261016-140233" in out and "lm64_2400_" in out
    assert "src/run.py --task image --model yolo11n.pt" in out and "| head" not in out


def test_run_happy_path(capsys):
    sh = ScriptedShell()
    calls = {}
    assert _run(["run", "yolo11n", "--timeout", "5"], sh, calls) == 0
    out = capsys.readouterr().out
    name = "yolo11n_2400_20261016-140233"
    rows = ex.load_jsonl(lab.LOCAL_LAB / "results.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["cell_id"] == name
    assert rows[0]["lab"]["preset"] == "yolo11n"
    assert rows[0]["lab"]["args"] == matrix._cell("yolo11n", 2400, 0, 0)["args"]
    assert calls["verify"] == [(str(lab.LOCAL_LAB / name), 2400)]
    (run,) = [(c, t) for k, c, t in sh.cmds if k == "ssh" and _run_py_cmd(c)]
    assert f"--out runs/lab/{name} > runs/lab/{name}.log" in run[0] and run[1] == 300
    assert (f"watch live: ssh maccpi@192.168.0.243 tail -f "
            f"Desktop/power-measurement-pi/runs/lab/{name}.log") in out
    for s in ("RUN OK", "forward", "9.58", "+4.67", "serve", "7.80", "max residual",
              "slip", "power_trace.parquet", "read_parquet"):
        assert s in out, s
    assert sh.sent("pi-power-pin restore")
    _no_power_commands(sh)


def test_run_failure_prints_stage_help_and_exits_1(capsys):
    sh = ScriptedShell([(_run_py_cmd, (1, "ACCEPTANCE: under-voltage"))])
    assert _run(["run", "yolo11n"], sh) == 1
    out = capsys.readouterr().out
    assert "RUN FAILED" in out and "'run'" in out and "What to do" in out
    assert ex.load_jsonl(lab.LOCAL_LAB / "results.jsonl")[0]["stage"] == "run"
    _no_power_commands(sh)


def test_pi_crash_points_to_readme_and_never_switches_power(capsys):
    up = [(0, ""), (0, ""), (255, "down")]    # pre-flight and pre-run checks fine, then gone
    sh = ScriptedShell([(_run_py_cmd, (255, "")), (lambda c: c == "true", up)])
    assert _run(["run", "yolo11n"], sh) == 1
    out = capsys.readouterr().out
    assert "pi_crash" in out and "README" in out and "rsts" in out
    assert "never switches power" in out and ".dmesg.log" in out
    _no_power_commands(sh)


def test_restore_skipped_while_run_py_still_running(capsys):
    sh = ScriptedShell([("[s]rc/run.py", [(1, ""), (0, "4242 python src/run.py --task image")])])
    assert _run(["run", "yolo11n"], sh) == 0
    assert not sh.sent("pi-power-pin restore")
    assert "still running" in capsys.readouterr().out


def test_keep_pinned_skips_restore():
    sh = ScriptedShell()
    assert _run(["run", "yolo11n", "--keep-pinned"], sh) == 0
    assert not sh.sent("pi-power-pin restore")


@pytest.mark.parametrize("where", ["local_dir", "local_log", "pi"])
def test_existing_name_refused(where, capsys):
    rules = []
    if where == "pi":
        rules = [("test -e", (0, ""))]
    else:
        lab.LOCAL_LAB.mkdir(parents=True)
        p = lab.LOCAL_LAB / ("taken" if where == "local_dir" else "taken.log")
        p.mkdir() if where == "local_dir" else p.write_text("x")
    sh = ScriptedShell(rules)
    assert _run(["run", "yolo11n", "--name", "taken"], sh) == 1
    assert "choose another --name" in capsys.readouterr().out
    assert not sh.sent("src/run.py ")


def test_preflight_failure_stops_before_the_run(capsys):
    sh = ScriptedShell([("is-active exp-matrix", (0, "active"))])
    assert _run(["run", "yolo11n"], sh) == 1
    assert not sh.sent("src/run.py ") and "FAIL" in capsys.readouterr().out


def test_no_thermal_gate_skips_the_wait():
    sh = ScriptedShell([("vcgencmd", (0, "temp=70.0'C\nthrottled=0x0"))])
    assert _run(["run", "yolo11n", "--no-thermal-gate"], sh) == 0
    assert len(sh.sent("vcgencmd")) == 1           # only the pre-flight reading


def test_ctrl_c_reports_remote_state_and_exits_130(capsys):
    sh = ScriptedShell([("[s]rc/run.py", [(1, ""), (0, "4242 python src/run.py")])])

    def align_fn(rec, pi, out):
        raise KeyboardInterrupt
    rc = _rc(["run", "yolo11n"], sh=sh, status_fn=live_status(),
             current_fn=lambda d: (0.9, 1.0), align_fn=align_fn, verify_fn=lambda d, m: {},
             now=lambda: NOW, sleep=lambda s: None)
    out = capsys.readouterr().out
    assert rc == 130 and "4242" in out and "pkill -INT -f src/run.py" in out
    assert not sh.sent("pkill") and not sh.sent("pi-power-pin restore")


def test_safe_print_swallows_broken_pipe(monkeypatch):
    class Closed:
        def write(self, s):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError
    monkeypatch.setattr("sys.stdout", Closed())
    lab.safe_print("hello")
    lab.safe_print("again")


# ---- check ------------------------------------------------------------------------------------
def _check(sh, status=None, status_fn=None, **kw):
    reads = {"n": 0}

    def st():
        if isinstance(status, Exception):
            raise status
        reads["n"] += 1             # a sampling recorder: n_total grows with every read
        s = dict(STATUS, **(status or {}))
        return dict(s, n_total=s["n_total"] + 10000 * reads["n"])
    checks, _ = lab.preflight(sh, status_fn or st, lambda d: (0.29, 0.31), sleep=lambda s: None,
                              **kw)
    return {c.name: c for c in checks}


def test_check_all_ok():
    c = _check(ScriptedShell())
    assert all(x.level == "ok" for x in c.values()), [(x.name, x.level) for x in c.values()]


def test_check_git_unknown_is_a_failure_and_mismatch_can_be_allowed():
    c = _check(ScriptedShell([(lambda t: t.startswith("git -C Desktop"), (128, "fatal"))]))
    assert c["code version"].level == "fail"
    other = "b" * 40
    sh = ScriptedShell([(lambda t: t.startswith("git -C Desktop"), (0, other))])
    assert _check(sh)["code version"].level == "fail"
    sh = ScriptedShell([(lambda t: t.startswith("git -C Desktop"), (0, other))])
    assert _check(sh, allow_git_mismatch=True)["code version"].level == "warn"


@pytest.mark.parametrize("bits,level", [("0x0", "ok"), ("0x50000", "warn"), ("0x4", "fail"),
                                        ("0x50005", "fail")])
def test_check_throttled(bits, level):
    c = _check(ScriptedShell([("vcgencmd", (0, f"temp=42.0'C\nthrottled={bits}"))]))
    assert c["pi throttling"].level == level


def test_sticky_undervoltage_bit_is_not_old_news():
    c = _check(ScriptedShell([("vcgencmd", (0, "temp=42.0'C\nthrottled=0x50000"))]))
    msg = c["pi throttling"].detail
    assert "old news" not in msg
    assert "reboot the Pi before a long matrix" in msg and "uv_alarm" in msg and "dmesg" in msg


def test_check_pin_helper_missing():
    c = _check(ScriptedShell([("pi-power-pin status", (1, "sudo: a password is required"))]))
    assert c["pinning helper"].level == "fail" and "4.5" in c["pinning helper"].hint


@pytest.mark.parametrize("rule", [("[s]rc/run.py", (0, "4242 python src/run.py")),
                                  ("[p]ytest", (0, "77 pytest -q")),
                                  ("is-active exp-matrix", (0, "active")),
                                  ("[e]xperiment.py run", (0, "9 python experiment.py run"))])
def test_check_busy_fails(rule):
    sh = ScriptedShell([rule])
    assert _check(sh)["idle"].level == "fail"
    assert not any(_run_py_cmd(c) for k, c, _ in sh.cmds if k == "ssh")


def test_check_recorder_unreachable_hints_the_elitebook():
    c = _check(ScriptedShell(), OSError("connection refused"))
    assert c["recorder"].level == "fail" and "192.168.0.11" in c["recorder"].hint
    assert c["recorder stall"].level == "skip"


def test_check_recorder_slow_or_old():
    assert _check(ScriptedShell(), {"sample_hz": 4000.0})["recorder"].level == "fail"
    c = _check(ScriptedShell(), {"n_total": int(4992.4 * 86400 * 8.6)})
    assert c["recorder age"].level == "warn" and "README 8" in c["recorder age"].hint


def test_check_pi_down_reports_current_and_skips_the_rest():
    sh = ScriptedShell([(lambda t: t == "true", (255, "timeout"))])
    c = _check(sh)
    assert c["pi reachable"].level == "fail"
    text = c["pi reachable"].detail + c["pi reachable"].hint
    assert "0.29" in text and "never switches power" in text and "README" in text
    assert c["pi temperature"].level == "skip" and c["idle"].level != "fail"
    assert {c[n].level for n in ("helper version", "pi wifi", "perf")} == {"skip"}
    _no_power_commands(sh)


def test_check_recorder_stall_is_an_error_row():
    frozen = lambda: dict(STATUS)                                    # noqa: E731
    c = _check(ScriptedShell(), status_fn=frozen)
    assert c["recorder stall"].level == "fail" and "not sampling" in c["recorder stall"].detail
    assert "README 8" in c["recorder stall"].hint
    ceiling = {"n_total": ex.SAMPLE_CEILING - 5000}
    c = _check(ScriptedShell(), ceiling)
    assert c["recorder stall"].level == "fail" and "ceiling" in c["recorder stall"].detail
    assert _check(ScriptedShell())["recorder stall"].level == "ok"


def test_check_recorder_stall_waits_between_the_two_reads_and_survives_a_dying_recorder():
    waits, reads = [], []

    def st():
        reads.append(len(waits))
        if len(reads) == 2:
            raise OSError("connection refused")
        return dict(STATUS)
    checks, _ = lab.preflight(ScriptedShell(), st, lambda d: (0.29, 0.31), sleep=waits.append)
    row = {c.name: c for c in checks}["recorder stall"]
    assert waits == [ex.STALL_READ_GAP_S] and reads == [0, 1]
    assert row.level == "fail" and "192.168.0.11" in row.hint


def test_check_pi_wifi_up_warns_and_absent_is_fine():
    c = _check(ScriptedShell([("operstate", (0, "up"))]))
    assert c["pi wifi"].level == "warn" and "power noise" in c["pi wifi"].detail
    assert "sudo nmcli con down stevens-iot" in c["pi wifi"].hint and "radio wifi off" not in c["pi wifi"].hint
    gone = ScriptedShell([("operstate", (1, "cat: /sys/class/net/wlan0/operstate: No such file"))])
    assert _check(gone)["pi wifi"].level == "ok"
    assert _check(ScriptedShell([("operstate", (255, "ssh: timeout"))]))["pi wifi"].level == "warn"
    assert _check(ScriptedShell())["pi wifi"].level == "ok"


def test_check_perf_missing_warns():
    c = _check(ScriptedShell([("command -v perf", (1, ""))]))
    assert c["perf"].level == "warn" and "counters" in c["perf"].detail
    assert _check(ScriptedShell())["perf"].detail == "/usr/bin/perf"


def test_check_old_pin_helper_warns_with_the_reinstall_hint():
    sh = ScriptedShell([("grep -c unpin", (1, "0"))])
    c = _check(sh)
    assert c["helper version"].level == "warn" and "reinstall" in c["helper version"].hint
    assert sh.sent("grep -c unpin /usr/local/sbin/pi-power-pin")
    assert c["pinning helper"].level == "ok"
    assert _check(ScriptedShell())["helper version"].level == "ok"


def test_check_models_and_check_image():
    sh = ScriptedShell([("test -f Desktop/power-measurement-pi/yolo11n.pt", (1, "")),
                        ("test -f Desktop/power-measurement-pi/bus.jpg", (1, ""))])
    c = _check(sh, models=["yolo11n.pt"], check_images=["bus.jpg"])
    assert c["model weights"].level == "warn" and "internet" in c["model weights"].detail
    assert c["check image"].level == "fail"


def test_check_command_exit_code(capsys):
    assert _rc(["check"], sh=ScriptedShell(), status_fn=live_status(),
               current_fn=lambda d: (0.9, 1.0), sleep=lambda s: None) == 0
    sh = ScriptedShell([("vcgencmd", (0, "temp=42.0'C\nthrottled=0x1"))])
    assert _rc(["check"], sh=sh, status_fn=live_status(), current_fn=lambda d: (0.9, 1.0),
               sleep=lambda s: None) == 1
    assert "NOT ready" in capsys.readouterr().out


# ---- test -------------------------------------------------------------------------------------
def test_pr1_command_is_exact():
    sh = ScriptedShell()
    assert _rc(["test", "pr1"], sh=sh) == 0
    (cmd,) = sh.sent("pytest -q")
    assert cmd == ("cd Desktop/power-measurement-pi && ~/.local/bin/uv run --frozen --with pytest "
                   "pytest -q "
                   "tests/test_monsoon_sampler.py::test_snapshot_env_is_bracketed_by_ctx_events "
                   "tests/test_tasks_steps.py::test_topk_bounded_by_number_of_classes "
                   "tests/test_tasks_steps.py::test_live_frames_use_shared_preprocess")
    assert list((lab.LOCAL_LAB / "tests").glob("*_pi.log"))


def test_known_failure_deselected_when_its_data_is_missing(capsys):
    sh = ScriptedShell([("fix_abab_smoke", (1, ""))])
    assert _rc(["test"], sh=sh) == 0
    (cmd,) = sh.sent("pytest -q")
    assert cmd.endswith("pytest -q tests/ --deselect "
                        "tests/test_phases.py::test_no_idle_contamination")
    assert "expected" in capsys.readouterr().out
    sh = ScriptedShell()
    assert _rc(["test", "all", "-k", "topk or snap"], sh=sh) == 0
    assert sh.sent("pytest -q")[0].endswith("pytest -q tests/ -k 'topk or snap'")


@pytest.mark.parametrize("arg", ["../x", "/etc/passwd", "tests/../../x"])
def test_test_refuses_paths_outside_the_project(arg):
    sh = ScriptedShell()
    assert _rc(["test", arg], sh=sh) == 2 and sh.cmds == []


def test_test_refuses_while_busy_and_reports_failures():
    sh = ScriptedShell([("[s]rc/run.py", (0, "4242 python src/run.py"))])
    assert _rc(["test", "pr1"], sh=sh) == 1 and not sh.sent("pytest -q")
    sh = ScriptedShell([("pytest -q", (1, "F..\n1 failed, 2 passed in 3.1s"))])
    assert _rc(["test", "pr1"], sh=sh) == 1


# ---- presets / help ---------------------------------------------------------------------------
def test_presets_lists_every_key_with_durations(capsys):
    lab.MATRIX_RESULTS.parent.mkdir(parents=True)
    lab.MATRIX_RESULTS.write_text(
        json.dumps({"model_key": "yolo11n", "cpu_mhz": 2400, "status": "ok", "duration_s": 150})
        + "\n" + json.dumps({"model_key": "yolo11n", "cpu_mhz": 2400, "status": "ok",
                             "duration_s": 90}) + "\n")
    assert _rc(["presets"]) == 0
    out = capsys.readouterr().out
    assert all(k in out for k in matrix.MODELS) and "2.0 min" in out


def test_presets_has_a_dataset_v2_section(capsys):
    assert _rc(["presets"]) == 0
    out = capsys.readouterr().out
    head, v2 = out.split("Dataset v2\n")
    assert all(k in head for k in matrix.MODELS) and "yolo11n_int8s" not in head
    assert all(k in v2 for k in ("yolo11n_int8s_640_t4", "yolo11n_fanoff_640_t4",
                                 "yolo11n_640_t4_dvfs", "yolo11m_soak30_640_t4",
                                 "yolo11n_concspin_640_t4", "yolo11n_fp16_640_t4"))
    assert "--yes-supervised" in v2


def test_v2_presets_are_one_per_group_with_the_v1_names_untouched():
    names = list(lab.V2_PRESETS)
    assert len(names) == len(set(names)) and not set(names) & set(matrix.MODELS)
    assert "yolo11n_640_t4_dvfs" in names and "yolo11n_640_t4_2400" not in names
    assert not any(n[-4:].isdigit() and "_r" in n for n in names)
    assert lab.V2_PRESETS["yolo11m_soak30_640_t4"]["timeout_s"] == 50 * 60
    assert lab.V2_PRESETS["yolo11n_fanoff_640_t4"]["family"] == "D"


def test_every_v2_preset_builds_with_its_matrix_v2_args():
    for name, v in lab.V2_PRESETS.items():
        (job,) = _jobs([name, "--include-cut"])
        assert job["args"] == v["args"], name
        assert "--fan-full" not in job["args"] and "--holdout" in job["args"]


def test_build_args_fan_off_preset_exact_argv():
    (job,) = _jobs(["yolo11n_fanoff_640_t4"])
    assert job["args"] == [
        "--task", "image", "--model", "yolo11n.pt", "--check-image", "bus.jpg",
        "--sampler", "monsoon", "--perf-counters", "on", "--holdout", "none",
        "--threads", "4", "--imgsz", "640", "--cpu-mhz", "2400", "--fan", "off",
        "--frames", "30", "--warmup", "10", "--no-bench", "--serve", "10",
        "--serve-gap-ms", "200"]
    assert job["cpu_mhz"] == 2400 and job["family"] == "D" and job["timeout_s"] is None


def test_build_args_dvfs_preset_exact_argv_and_overrides():
    (job,) = _jobs(["yolo11n_640_t4_dvfs"])
    assert job["args"] == [
        "--task", "image", "--model", "yolo11n.pt", "--check-image", "bus.jpg",
        "--sampler", "monsoon", "--perf-counters", "on", "--holdout", "none",
        "--threads", "4", "--imgsz", "640", "--cpu-mhz", "2400", "--fan", "full",
        "--governor", "ondemand", "--frames", "30", "--warmup", "10", "--no-bench",
        "--serve", "10", "--serve-gap-ms", "200"]
    (job,) = _jobs(["yolo11n_int8s_640_t4", "--frames", "5", "--threads", "2", "--cpu-mhz", "1500"])
    a = lab.args_to_dict(job["args"])
    assert a["--frames"] == "5" and a["--threads"] == "2" and a["--cpu-mhz"] == "1500"
    assert a["--quant"] == "int8-static" and a["--quant-weights"] == "models/yolo11n_int8s.pt"
    assert list(a)[:3] == ["--task", "--model", "--check-image"] and job["args"].count("--threads") == 1


def test_v2_flags_survive_build_args():
    for name, flag in (("yolo11n_conc2_640_t4", "--concurrent"), ("yolo11n_serve5_640_t4",
                       "--serve-fps"), ("yolo11n_stream_640_t4", "--frames-source"),
                       ("yolo11n_empty_640_t4", "--input-image"),
                       ("yolo11n_p30_640_t4", "--prune"), ("yolo11n_soak30_640_t4",
                       "--soak-minutes")):
        assert flag in _jobs([name, "--include-cut"])[0]["args"], name


def test_unknown_preset_suggests_v2_names():
    with pytest.raises(lab.UsageError, match="yolo11n_int8s_640_t4"):
        _jobs(["yolo11n_int8s_640_t"])


# ---- cut presets (final review M4) ----------------------------------------------------------------
CUT = ["yolo11n_int8d_640_t4", "yolo11s_int8d_640_t4", "yolov8n_int8d_640_t4",
       "yolo11n_p30_640_t4", "yolo11n_p50_640_t4"]


def test_cut_presets_are_marked_and_hidden_from_the_list(capsys):
    assert {k for k, v in lab.V2_PRESETS.items() if v["cut"]} == set(CUT)
    assert _rc(["presets"]) == 0
    out = capsys.readouterr().out
    assert not any(k in out for k in CUT) and "yolo11n_int8s_640_t4" in out
    assert "--include-cut" in out
    assert _rc(["presets", "--include-cut"]) == 0
    out = capsys.readouterr().out
    assert all(k in out for k in CUT)


@pytest.mark.parametrize("name", CUT)
def test_cut_presets_are_refused_without_include_cut(name, capsys):
    sh = ScriptedShell()
    assert _run(["run", name], sh) == 2
    err = capsys.readouterr().err
    assert "--include-cut" in err and "cut from the dataset" in err and sh.cmds == []
    assert _rc(["run", name, "--include-cut", "--dry-run"]) == 0


def test_unknown_preset_does_not_suggest_cut_presets():
    with pytest.raises(lab.UsageError) as e:
        _jobs(["yolo11n_int8d_640_t"])
    assert "int8d" not in str(e.value).split("did you mean")[-1]


# ---- supervised fan-off -------------------------------------------------------------------------
def test_fan_off_preset_needs_yes_supervised(capsys):
    sh = ScriptedShell()
    assert _run(["run", "yolo11n_fanoff_640_t4"], sh) == 2
    err = capsys.readouterr().err
    assert "watched in person" in err and "--yes-supervised" in err
    assert sh.cmds == []


def test_fan_off_preset_with_yes_supervised_warns_in_one_line_and_runs(capsys):
    sh = ScriptedShell()
    assert _run(["run", "yolo11n_fanoff_640_t4", "--yes-supervised"], sh) == 0
    out = capsys.readouterr().out
    warns = [ln for ln in out.splitlines() if "watched in person" in ln]
    assert len(warns) == 1 and "yolo11n_fanoff_640_t4" in warns[0]
    (run,) = [c for k, c, _ in sh.cmds if k == "ssh" and _run_py_cmd(c)]
    assert "--fan off" in run and "--fan-full" not in run
    assert "fan off" in out                  # the summary names the fan mode of the cell


def test_dry_run_of_a_fan_off_preset_needs_no_flag(capsys):
    assert _run(["run", "yolo11n_fanoff_640_t4", "--dry-run"], ScriptedShell()) == 0
    assert "watched in person" in capsys.readouterr().out


def test_other_presets_need_no_yes_supervised():
    assert _rc(["run", "yolo11n_640_t4_dvfs", "yolo11n", "--dry-run"]) == 0


def test_soak_preset_gets_family_and_a_longer_timeout():
    sh = ScriptedShell()
    assert _run(["run", "yolo11n_soak30_640_t4"], sh) == 0
    (run,) = [(c, t) for k, c, t in sh.cmds if k == "ssh" and _run_py_cmd(c)]
    assert run[1] == 50 * 60


# ---- stages -------------------------------------------------------------------------------------
def test_disk_and_thermal_abort_stages_have_help_and_disk_stops_the_next_preset():
    for stage in ("disk", "thermal_abort"):
        what, cause, todo = lab.STAGE_HELP[stage]
        assert what and cause and todo
    assert "85 C" in lab.STAGE_HELP["thermal_abort"][0] and "exit 3" in lab.STAGE_HELP["thermal_abort"][0]
    assert "disk" in lab.STOP_STAGES and "thermal_abort" not in lab.STOP_STAGES


def test_pi_busy_timeout_and_undervoltage_stages_have_help():
    for stage in ("pi_busy", "timeout", "undervoltage"):
        what, cause, todo = lab.STAGE_HELP[stage]
        assert what and cause and todo
    assert "pi_busy" in lab.STOP_STAGES
    assert "timeout" not in lab.STOP_STAGES and "undervoltage" not in lab.STOP_STAGES


def test_run_while_run_py_is_active_stops_with_pi_busy(capsys):
    sh = ScriptedShell([("pytho[n] src/run.py", (0, "4242 python src/run.py --task image"))])
    assert _run(["run", "yolo11n", "lm64"], sh) == 1
    out = capsys.readouterr().out
    assert "RUN FAILED at 'pi_busy'" in out and "4242" in out
    assert "Not starting the remaining presets after a 'pi_busy' failure" in out
    assert not [c for k, c, _ in sh.cmds if k == "ssh" and _run_py_cmd(c)]


def test_timeout_failure_is_explained(capsys):
    sh = ScriptedShell([(_run_py_cmd, (124, "timeout")),
                        ("pytho[n] src/run.py", [(1, ""), (0, "9 python src/run.py"), (1, "")])])
    assert _run(["run", "yolo11n"], sh) == 1
    out = capsys.readouterr().out
    assert "RUN FAILED at 'timeout'" in out and "SIGINT sent and exited" in out
    assert sh.sent("pkill -INT -f 'pytho[n] src/run.py'")


def test_thermal_abort_failure_is_explained(capsys):
    sh = ScriptedShell([(_run_py_cmd, (3, "thermal abort"))])
    assert _run(["run", "yolo11n_fanoff_640_t4", "--yes-supervised"], sh) == 1
    out = capsys.readouterr().out
    assert "RUN FAILED at 'thermal_abort'" in out and "85 C" in out and "fan mode" in out


@pytest.mark.parametrize("cmd", [[], ["check"], ["run"], ["test"], ["presets"]])
def test_help_works(cmd, capsys):
    assert _rc([*cmd, "--help"]) == 0
    assert "Examples" in capsys.readouterr().out
