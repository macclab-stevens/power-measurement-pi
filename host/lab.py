#!/usr/bin/env python3
"""lab.py - measure one ML workload on the Raspberry Pi 5 with the Monsoon, from the EliteBook.

Commands:
  check     Is everything ready? (recorder, Pi, temperature, throttling, pinning helper,
            code versions, nothing else running)
  run       Measure one or more workloads: run them on the Pi, copy the data back, align
            it, check it, and print a plain-English summary of each.
  test      Run the unit tests on the Pi (and/or here on the EliteBook).
  presets   List the ready-made workloads (the 14 models of the experiment matrix).

Run `lab.py <command> --help` for every option of a command. Start it with `./lab ...` from
the project folder, or `~/.local/bin/uv run python lab.py ...` from host/. It must run on the
EliteBook (192.168.0.11), where the recorder runs.

This script never stops the recorder and never switches the Pi's power. If the Pi locks up it
tells you what to do (README sections 8 and 9). For the full 224-cell matrix use experiment.py.

Exit codes: 0 all ok, 1 failure (the failed stage or check is printed), 2 bad command line,
130 interrupted (Ctrl-C).

Examples:
  ./lab check
  ./lab presets
  ./lab run yolo11n                                   # YOLO11n inference, 2400 MHz, fan full
  ./lab run yolo11n train_lm64_adam train_resnet18_sgd   # the 3-cell smoke test
  ./lab run train_resnet18_sgd --cpu-mhz 1500
  ./lab run lm64 --frames 200 --name lm64_quick
  ./lab run --task image --model torchvision:resnet50 --imgsz 224 --frames 50
  ./lab test pr1                                      # the 3 tests from the PR #1 review
  ./lab test --host                                   # whole Pi suite + host suite
"""
from __future__ import annotations

import argparse
import csv
import datetime
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import experiment as ex
import matrix

REPO = ex.HERE.parent
LAB_SUBDIR = "runs/lab"                       # same relative folder on both machines
LOCAL_LAB = REPO / "runs" / "lab"
PI_LAB = f"{ex.PI_DIR}/{LAB_SUBDIR}"
MATRIX_RESULTS = REPO / "runs" / "matrix" / "results.jsonl"
PIN = "sudo -n /usr/local/sbin/pi-power-pin"
MHZ_CHOICES = list(range(1500, 2401, 100))
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
RECORDER_MIN_HZ = 4900                        # run.py's MONSOON_MIN_HZ
RECORDER_AGE_WARN_DAYS = 8.5                  # the Monsoon library stops at ~9.9 days
PGREP_RUN = "pgrep -af '[s]rc/run.py'"        # [s]: never matches pgrep's own shell
PGREP_PYTEST = "pgrep -af '[p]ytest'"
PR1 = ["tests/test_monsoon_sampler.py::test_snapshot_env_is_bracketed_by_ctx_events",
       "tests/test_tasks_steps.py::test_topk_bounded_by_number_of_classes",
       "tests/test_tasks_steps.py::test_live_frames_use_shared_preprocess"]
KNOWN_FAIL = "tests/test_phases.py::test_no_idle_contamination"
KNOWN_FAIL_DATA = "runs/fix_abab_smoke/samples.csv"
STOP_STAGES = ("recorder", "pi_down", "pi_crash", "thermal")   # don't start the next preset
NEVER = "This script never switches power."
ELITEBOOK_HINT = "run this on the EliteBook (192.168.0.11), where the recorder runs (README 4.7)"
PI_DOWN_HINT = ("Wait 3 minutes (it sometimes reboots itself), then `./lab check` again. If it "
                "stays down and the Monsoon shows ~0.3 A, follow README section 8 steps 2, 3 and 5 "
                f"(lockups: README 9). {NEVER}")
STOP_RUN = f"ssh {ex.PI} pkill -INT -f src/run.py"


class UsageError(Exception):
    pass


def safe_print(*a, **kw) -> None:
    """print that survives a closed pipe (`./lab run ... | head`): further output is dropped,
    the launcher and the run on the Pi carry on."""
    try:
        print(*a, **kw, flush=True)
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass


class LabShell(ex.Shell):
    """experiment.Shell with stdin closed for every child: a backgrounded launcher is never
    stopped by SIGTTIN, and keystrokes never reach run.py on the Pi."""

    def run(self, cmd: list[str], timeout: float, cwd=None) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            return 124, f"timeout after {timeout}s: {e}"
        return p.returncode, (p.stdout + p.stderr)

    def popen(self, cmd: list[str], out_path):
        f = open(out_path, "w")
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f,
                                stderr=subprocess.STDOUT), f


# ---- parsers (ssh merges stderr, so the Pi's setlocale warning can come before or after) -----
def _lines(out: str) -> list[str]:
    return [s.strip() for s in out.splitlines()
            if s.strip() and not s.lstrip().startswith("bash: warning: setlocale")]


def parse_throttled(out: str) -> int | None:
    """`throttled=0x50000` -> 0x50000."""
    for line in _lines(out):
        if line.startswith("throttled="):
            try:
                return int(line.split("=", 1)[1], 16)
            except ValueError:
                return None
    return None


def parse_pytest_summary(out: str) -> str | None:
    """The final `3 passed, 1 deselected in 4.2s` line, without the ==== decoration."""
    pat = re.compile(r"(\d+ (passed|failed|errors?|skipped|deselected)|no tests ran)\b.*"
                     r"\bin [\d.]+s")
    for line in reversed(_lines(out)):
        if pat.search(line):
            return line.strip("= ")
    return None


def _short(h: str | None) -> str:
    return h[:7] if h else "unknown"


# ---- pre-flight --------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    level: str              # ok | warn | fail | skip
    detail: str
    hint: str = ""


def pi_down_detail(sh, current_fn, run_dir) -> str:
    ping = "yes" if sh.ping() else "no"
    try:
        mean_a, max_a = current_fn(run_dir)
        cur = f"Monsoon current last 5 s: mean {mean_a:.2f} A, max {max_a:.2f} A"
        cur += (", which looks like a lockup/halt" if ex.halt_confirmed(mean_a, max_a)
                else ", not halted-level (busy or rebooting?)")
    except Exception as e:  # noqa: BLE001 - diagnosis only
        cur = f"Monsoon current unreadable ({type(e).__name__})"
    return f"no ssh answer (ping: {ping}). {cur}"


def idle_check(sh, pi_up: bool) -> Check:
    """Nothing else measuring: run.py / pytest on the Pi, the matrix here. pgrep rc 1 = none."""
    busy, errors = [], []
    probes = [("run.py on the Pi", lambda: sh.ssh(PGREP_RUN, 15)),
              ("pytest on the Pi", lambda: sh.ssh(PGREP_PYTEST, 15))] if pi_up else []
    probes.append(("experiment.py run here",
                   lambda: sh.run(["pgrep", "-af", "[e]xperiment.py run"], 10)))
    for what, probe in probes:
        rc, out = probe()
        if rc == 0:
            pids = ", ".join(s.split()[0] for s in _lines(out)) or "?"
            busy.append(f"{what} (pid {pids})")
        elif rc != 1:
            errors.append(f"{what}: pgrep rc={rc}")
    if sh.run(["systemctl", "--user", "is-active", "exp-matrix"], 10)[0] == 0:
        busy.append("the matrix (exp-matrix service)")
    skipped = "" if pi_up else " (Pi part skipped)"
    if busy:
        return Check("idle", "fail", "busy: " + "; ".join(busy),
                     "wait for it to finish; this script never stops other runs")
    if errors:
        return Check("idle", "fail", "could not check: " + "; ".join(errors))
    return Check("idle", "ok", "no matrix, no run.py or pytest on the Pi" + skipped)


def preflight(sh, status_fn, current_fn, *, allow_git_mismatch=False, models=(),
              check_images=()) -> tuple[list[Check], dict]:
    """Every check, in order. Returns (checks, facts); facts holds git heads and temperature."""
    checks: list[Check] = []
    facts: dict = {"git_host": None, "git_pi": None, "temp": None}
    add = lambda *a: checks.append(Check(*a))  # noqa: E731

    # 1-2. recorder: only GET /status
    run_dir = None
    try:
        st = status_fn()
    except (OSError, ValueError) as e:
        st = None
        add("recorder", "fail", f"unreachable: {e}", ELITEBOOK_HINT)
        add("recorder age", "skip", "recorder unreachable")
    if st is not None:
        run_dir = st.get("run_dir")
        hz = float(st.get("sample_hz") or 0)
        if not st.get("recording"):
            add("recorder", "fail", "running but not recording", "README 4.7 (start the recorder)")
        elif hz < RECORDER_MIN_HZ:
            add("recorder", "fail", f"only {hz:.0f} Hz (needs {RECORDER_MIN_HZ})",
                "just started? wait 10 s and re-check; otherwise README 9")
        else:
            add("recorder", "ok", f"recording, {hz:.0f} Hz, serial {st.get('serial')}, {run_dir}")
        if hz > 0 and st.get("n_total") is not None:
            days = st["n_total"] / hz / 86400
            if days > RECORDER_AGE_WARN_DAYS:
                add("recorder age", "warn", f"sampling for {days:.1f} days",
                    "the Monsoon library stops at ~9.9 days; restart it the safe way, README 8; "
                    "this script won't")
            else:
                add("recorder age", "ok", f"sampling for {days:.1f} days of ~9.9")
        else:
            add("recorder age", "skip", "unknown (no sample rate yet)")

    # 3. Pi reachable; if not, diagnose and skip the other Pi checks
    pi_up = sh.ssh("true", 15)[0] == 0
    if pi_up:
        add("pi reachable", "ok", ex.PI)
    else:
        add("pi reachable", "fail", pi_down_detail(sh, current_fn, run_dir), PI_DOWN_HINT)
        for name in ("pi temperature", "pi throttling", "pinning helper", "code version"):
            add(name, "skip", "Pi unreachable")

    if pi_up:
        # 4. temperature and throttling
        out = sh.ssh("vcgencmd measure_temp; vcgencmd get_throttled", 30)[1]
        temp, bits = ex.parse_temp(out), parse_throttled(out)
        facts["temp"] = temp
        if temp is None:
            add("pi temperature", "warn", "could not read vcgencmd measure_temp")
        elif temp >= ex.THERMAL_MAX_C:
            add("pi temperature", "warn", f"{temp:.1f} C, warm: a run waits until it is below "
                f"{ex.THERMAL_MAX_C:.0f} C")
        else:
            add("pi temperature", "ok", f"{temp:.1f} C (gate {ex.THERMAL_MAX_C:.0f} C)")
        if bits is None:
            add("pi throttling", "warn", "could not read vcgencmd get_throttled")
        elif bits & 0xF:
            add("pi throttling", "fail", f"throttled={bits:#x}: under-voltage or throttling now",
                "check the power cable and the fan (README 9), wait, re-check")
        elif bits & 0xF0000:
            add("pi throttling", "warn", f"throttled={bits:#x}: happened earlier since boot "
                "(old news)")
        else:
            add("pi throttling", "ok", "throttled=0x0")
        # 5. pinning helper
        rc, out = sh.ssh(f"{PIN} status", 30)
        gov = [s for s in _lines(out) if "governor=" in s]
        if rc == 0 and gov:
            add("pinning helper", "ok", gov[0])
        else:
            add("pinning helper", "fail", f"missing or not allowed (rc={rc})",
                "install /usr/local/sbin/pi-power-pin and its sudoers entry (README 4.5)")

    # 6. code versions: the full hash on both sides
    rc, out = sh.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], 10)
    host = ex.parse_git_head(out) if rc == 0 else None
    if pi_up:
        rc, out = sh.ssh(f"git -C {ex.PI_DIR} rev-parse HEAD", 30)
        pi = ex.parse_git_head(out) if rc == 0 else None
        facts.update(git_host=_short(host), git_pi=_short(pi))
        if host and host == pi:
            add("code version", "ok", f"{_short(host)} on both machines")
        else:
            add("code version", "warn" if allow_git_mismatch else "fail",
                f"EliteBook {_short(host)} vs Pi {_short(pi)}",
                "update both (README 12.2), or --allow-git-mismatch")

    # 7. nothing else measuring
    checks.append(idle_check(sh, pi_up))

    # 8. model weights and the detection check image
    if pi_up:
        pts = sorted({m for m in models if m.endswith(".pt")})
        missing = [m for m in pts if sh.ssh(f"test -f {ex.PI_DIR}/{shlex.quote(m)}", 15)[0]]
        if missing:
            add("model weights", "warn", f"{', '.join(missing)} not on the Pi: will be downloaded "
                "(needs internet)")
        elif pts:
            add("model weights", "ok", ", ".join(pts))
        for img in sorted({i for i in check_images if i}):
            if sh.ssh(f"test -f {ex.PI_DIR}/{shlex.quote(img)}", 15)[0]:
                add("check image", "fail", f"{img} not found in ~/{ex.PI_DIR}",
                    "copy it there, or --no-check-image")
            else:
                add("check image", "ok", img)
    return checks, facts


def format_checks(checks: list[Check]) -> str:
    tag = {"ok": "[ok]  ", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[skip]"}
    lines = ["Pre-flight checks"]
    for c in checks:
        lines.append(f"  {tag[c.level]} {c.name:<15} {c.detail}")
        if c.hint and c.level in ("warn", "fail"):
            lines.append(f"{'':25}-> {c.hint}")
    nf = sum(c.level == "fail" for c in checks)
    nw = sum(c.level == "warn" for c in checks)
    lines.append(f"NOT ready: {nf} check(s) failed (see hints above)." if nf
                 else f"Ready ({nw} warning(s)).")
    return "\n".join(lines)


# ---- run.py arguments ----------------------------------------------------------------------------
BOOL_FLAGS = {"--no-bench", "--fan-full", "--live-camera"}
VALUE_OVERRIDES = {"frames": "--frames", "warmup": "--warmup", "imgsz": "--imgsz", "seq": "--seq",
                   "batch": "--batch", "optimizer": "--optimizer", "serve": "--serve",
                   "serve_gap_ms": "--serve-gap-ms", "conf": "--conf",
                   "check_image": "--check-image"}


def args_to_dict(tokens: list[str]) -> dict:
    """Ordered {flag: value}; boolean flags map to None."""
    d, i = {}, 0
    while i < len(tokens):
        if tokens[i] in BOOL_FLAGS:
            d[tokens[i]], i = None, i + 1
        else:
            d[tokens[i]], i = tokens[i + 1], i + 2
    return d


def dict_to_args(d: dict) -> list[str]:
    return [t for k, v in d.items() for t in ([k] if v is None else [k, v])]


def build_args(base: list[str], a) -> list[str]:
    """base (a matrix preset or --task/--model) + only the overrides the user typed, then the
    matrix's common flags and --cpu-mhz, in the matrix's order (so no overrides == matrix cell)."""
    d = args_to_dict(base)
    for attr, flag in VALUE_OVERRIDES.items():
        if getattr(a, attr) is not None:
            d[flag] = str(getattr(a, attr))
    if a.bench is not None:
        if float(a.bench) == 0:
            d.pop("--bench", None)
            d["--no-bench"] = None
        else:
            d.pop("--no-bench", None)
            d["--bench"] = a.bench
    if a.no_check_image:
        d.pop("--check-image", None)
    if a.live_camera:
        d["--live-camera"] = None
    common = args_to_dict(matrix.COMMON)
    if a.threads is not None:
        common["--threads"] = str(a.threads)
    for k in [*common, "--cpu-mhz"]:
        d.pop(k, None)
    return [*dict_to_args(d), *dict_to_args(common), "--cpu-mhz", str(a.cpu_mhz)]


def sanitize(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-._")[:60] or "custom"


def plan_jobs(a, now: datetime.datetime) -> list[dict]:
    """One job per preset (or one custom job): run.py args, run name, model, check image."""
    presets, custom = a.presets, a.task is not None or a.model is not None
    if presets and custom:
        raise UsageError("give preset name(s) or --task and --model, not both")
    if not presets and not (a.task and a.model):
        raise UsageError("give a preset (see `./lab presets`) or both --task and --model")
    for p in presets:
        if p not in matrix.MODELS:
            close = difflib.get_close_matches(p, matrix.MODELS, n=3)
            raise UsageError(f"unknown preset {p!r}"
                             + (f"; did you mean {', '.join(close)}?" if close
                                else "; `./lab presets` lists them"))
    if len(set(presets)) != len(presets):
        raise UsageError("give each preset once (for repeats use experiment.py, README 5.4)")
    if a.name is not None:
        if len(presets) > 1:
            raise UsageError("--name works with a single preset only; several presets get "
                             "their own default names")
        if not NAME_RE.match(a.name):
            raise UsageError(f"--name {a.name!r}: use letters, digits, '.', '_' and '-' only, "
                             "starting with a letter or digit (at most 100 characters)")
    if a.check_image is not None and a.no_check_image:
        raise UsageError("--check-image and --no-check-image contradict each other")
    stamp = now.strftime("%Y%m%d-%H%M%S")
    jobs = []
    for key in presets or [None]:
        base = list(matrix.MODELS[key]) if key else ["--task", a.task, "--model", a.model,
                                                     *(["--no-bench"] if a.task == "train" else [])]
        args = build_args(base, a)
        d = args_to_dict(args)
        if d["--task"] == "train" and int(d.get("--batch", 8)) < 2:
            raise UsageError("--task train needs --batch 2 or more (BatchNorm in training mode)")
        jobs.append({"preset": key, "args": args, "cpu_mhz": a.cpu_mhz, "model": d["--model"],
                     "task": d["--task"], "check_image": d.get("--check-image"),
                     "name": a.name or f"{key or sanitize(a.model)}_{a.cpu_mhz}_{stamp}"})
    return jobs


# ---- summaries -----------------------------------------------------------------------------------
def _read_json(p) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def phase_means(csv_path) -> dict:
    """phase -> mean power (W) over all its rows of phase_energy.csv (E_mJ / dur_ms)."""
    sums: dict = {}
    try:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                s = sums.setdefault(r["phase"], [0.0, 0.0])
                s[0] += float(r["E_mJ"] or 0)
                s[1] += float(r["dur_ms"] or 0)
    except (OSError, ValueError, KeyError):
        return {}
    return {k: e / d for k, (e, d) in sums.items() if d > 0}


def _dur(s) -> str:
    if s is None:
        return "?"
    m, sec = divmod(int(round(s)), 60)
    return f"{m} min {sec} s" if m else f"{sec} s"


def _f(v, fmt=".2f") -> str:
    return "?" if v is None else format(v, fmt)


def format_summary(job: dict, res: dict, ver: dict, facts: dict) -> str:
    name = job["name"]
    d = LOCAL_LAB / name
    meta = _read_json(d / "run_meta.json")
    phases = phase_means(d / "phase_energy.csv")
    ali = res.get("align") or {}
    idle = meta.get("baseline_W") or ver.get("baseline_P_W")
    train = job["task"] == "train"
    frames = meta.get("frames_per_mode", args_to_dict(job["args"]).get("--frames"))
    bar = "=" * 76
    L = [bar, f" RUN OK  {name}    {job['task']}, "
         + (f"preset {job['preset']}" if job["preset"] else "custom"), bar,
         f" Workload     {job['model']}, {frames} {'optimizer steps' if train else 'iterations'}"
         " per mode",
         f" Conditions   CPU pinned {meta.get('cpu_mhz_pinned', job['cpu_mhz'])} MHz, fan "
         f"{'full' if meta.get('fan_full', True) else 'NOT full'}, Pi "
         f"{_f(facts.get('temp'), '.1f')} C at start, git {res.get('git_host')} (EliteBook) / "
         f"{res.get('git_pi')} (Pi)",
         f" Duration     {_dur(res.get('duration_s'))} total (run.py {_dur(res.get('run_s'))})",
         "",
         f" Mean power (W)          {'above idle':>12}",
         f"   idle (baseline)  {_f(idle):>7}            (A {_f(meta.get('baseline_A_W'))} / "
         f"B {_f(meta.get('baseline_B_W'))})"]
    rows = dict(ver.get("mean_P_W") or {})
    if "serve" in phases:
        rows["serve"] = phases["serve"]
    for k, v in rows.items():
        extra = "   requests with idle gaps" if k == "serve" else ""
        above = f"{v - idle:+.2f}" if idle is not None else "?"
        L.append(f"   {k:<16} {v:>7.2f}   {above:>12}{extra}")
    steps = ver.get("step_P_W") or {}
    if steps:
        L.append(" Per step (W)  " + "   ".join(f"{k} {v:.2f}" for k, v in steps.items()))
    loss = meta.get("train_loss")
    if loss:
        L.append(f" Training loss {loss[0]:.3f} -> {loss[-1]:.3f} over {len(loss)} steps")
    if ver.get("energy_total_J") is not None:
        L.append(f" Energy        {ver['energy_total_J']:.1f} J in the aligned trace")
    L.append(f" Alignment     {ali.get('n_edges')} edges, max residual "
             f"{_f(ali.get('max_residual_ms'))} ms (limit 1.0 ms), drift "
             f"{_f(ali.get('drift_ppm'), '.1f')} ppm")
    if ali.get("slip_warning"):
        n = ali.get("slip_samples_total") or 0
        L.append(f" Data quality  slip warning: the Monsoon lost {n} samples (~{n / 5:.1f} ms). "
                 "Labels after that point are\n               shifted by a few ms; averages are "
                 "unaffected (README 9).")
    if ver.get("step_coverage") is not None:
        L.append(f"               step coverage {ver['step_coverage']:.0%} of the measured modes")
    for w in ver.get("warnings") or []:
        if not w.startswith("slip_warning"):        # explained above
            L.append(f" Warning       {w}")
    L += [f" Files         {d}/",
          "   power_trace.parquet   every 5 kHz sample with phase/step/layer labels",
          "   phase_energy.csv      energy and mean power per phase / iteration / step",
          "   run_meta.json, alignment.json",
          f"   ../{name}.log         Pi console output",
          f"   ../{name}.dmesg.log   Pi kernel messages during the run",
          f" On the Pi     ~/{PI_LAB}/{name}/",
          f" Load it       ~/.local/bin/uv run --project {ex.HERE} python -c "
          "\"import pandas as pd; "
          f"print(pd.read_parquet('{d}/power_trace.parquet').groupby('phase').P_W.mean())\"",
          bar]
    return "\n".join(L)


_CRASH_DO = ("Wait 3 minutes; it often reboots itself. Then `./lab check`. If it stays down and "
             "the Monsoon shows ~0.3 A,\n                follow README section 8 steps 2, 3 and 5. "
             + NEVER)
STAGE_HELP = {   # stage -> (what happened, likely cause, what to do)
    "recorder": ("The Monsoon recorder was unreachable or not recording.",
                 "The recorder service stopped, or this is not the EliteBook.",
                 f"`curl -s localhost:8051/status`; {ELITEBOOK_HINT}."),
    "pi_down": ("The Pi did not answer ssh for 3 minutes before the run.",
                "Pi lockup or reboot (README 9).", _CRASH_DO),
    "thermal": ("The Pi did not cool below 50 C within 10 minutes.",
                "A previous run or test heated it, or the fan is not at full speed.",
                "Wait and run again, or use --no-thermal-gate (the start temperature then "
                "differs from the matrix)."),
    "pi_crash": ("The Pi stopped answering during the run.",
                 "Pi lockup (README 9: board/firmware, seen with mobilenet_v3_small).", _CRASH_DO),
    "run": ("run.py on the Pi ended with an error.",
            "See its last lines below. ACCEPTANCE: under-voltage -> power cable (README 9); "
            "baseline drift -> something else ran on the Pi; a task check -> model or weights.",
            "Fix the cause and run again with a new --name (or the default name)."),
    "rsync": ("Copying the results back failed.", "Network problem or a full disk here.",
              f"Check `df -h ~` and the network; the data is still on the Pi in ~/{PI_LAB}."),
    "align": ("The power trace could not be lined up with the Pi's events.",
              "Usually the Monsoon lost samples inside a sync pulse (rare, README 9).",
              "Just run it again."),
    "verify": ("The aligned run failed the data checks.", "See the errors below.",
               "If they mention the clock or fan, check `pi-power-pin status` (README 4.5)."),
}


def format_failure(job: dict, res: dict) -> str:
    name, stage = job["name"], res.get("stage")
    what, cause, todo = STAGE_HELP.get(stage, ("Unknown failure.", "?", "See the details."))
    if stage == "run" and res.get("run_rc") == 124:
        what = "run.py took longer than --timeout and was abandoned."
        cause = "A slow model, too many --frames, or a hung run."
        todo = (f"It may still be running on the Pi: `ssh {ex.PI} pgrep -af src/run.py`; "
                f"stop it with `{STOP_RUN}`.")
    bar = "=" * 76
    L = [bar, f" RUN FAILED at '{stage}'  {name}", bar,
         f" What happened  {what}", f" Likely cause   {cause}", f" What to do     {todo}"]
    if stage in ("pi_crash", "pi_down"):
        L += [f" Evidence       {LOCAL_LAB / name}.dmesg.log  (kernel log; often empty for "
              "lockups)",
              f"                {LOCAL_LAB / name}.log        (console output, may be partial)",
              "                after it is back: `cat /proc/device-tree/chosen/bootloader/rsts` on "
              "the Pi (0x20 / 0x1020 = watchdog reset)"]
    else:
        L.append(f" Logs           {LOCAL_LAB / name}.log, {LOCAL_LAB / name}.dmesg.log")
    L.append(" Details")
    L += [f"   {s}" for s in _lines(str(res.get("error") or ""))[-15:]] or ["   (none)"]
    L.append(bar)
    return "\n".join(L)


# ---- commands ------------------------------------------------------------------------------------
def pi_run_py(sh) -> tuple[str, list[str]]:
    """('busy' | 'idle' | 'error', pgrep lines) for run.py on the Pi."""
    rc, out = sh.ssh(PGREP_RUN, 15)
    return ("busy" if rc == 0 else "idle" if rc == 1 else "error"), _lines(out)


def restore_pins(sh, keep_pinned: bool, state: str) -> None:
    if keep_pinned:
        safe_print(f"Clock and fan left pinned (--keep-pinned). Later: ssh {ex.PI} {PIN} restore")
    elif state == "busy":
        safe_print("Clock and fan left pinned because run.py is still running on the Pi.")
    elif state == "error":
        safe_print("Could not check the Pi for run.py; clock and fan left as they are.")
    else:
        rc, _ = sh.ssh(f"{PIN} restore", 60)
        safe_print("Clock and fan restored to normal." if rc == 0
                   else f"pi-power-pin restore failed (rc={rc}); clock and fan may stay pinned.")


def cmd_run(a, sh, status_fn, current_fn, align_fn, verify_fn, now, sleep) -> int:
    jobs = plan_jobs(a, now())
    if a.dry_run:
        for j in jobs:
            cell = {"cell_id": j["name"], "args": j["args"]}
            out = f"{LAB_SUBDIR}/{j['name']}"
            safe_print(f"{j['name']}\n  run.py:   {ex.PI_UV} run --frozen python src/run.py "
                       f"{shlex.join([*j['args'], '--out', out])}\n"
                       f"  Pi:       {ex.pi_command(cell, LAB_SUBDIR)}\n"
                       f"  here:     {LOCAL_LAB / j['name']}/\n  on Pi:    ~/{PI_LAB}/{j['name']}/")
        safe_print("(dry run: no machine was contacted)")
        return 0
    try:
        if align_fn is None:
            from align import align as align_fn
        if verify_fn is None:
            from verify_cell import verify as verify_fn
    except ImportError as e:
        safe_print(f"Cannot load the analysis code ({e}). Start it with ./lab or "
                   "`~/.local/bin/uv run python lab.py` from host/.")
        return 1
    for j in jobs:
        for p in (LOCAL_LAB / j["name"], LOCAL_LAB / f"{j['name']}.log"):
            if p.exists():
                safe_print(f"{p} already exists: choose another --name")
                return 1
    sh = sh or LabShell()
    checks, facts = preflight(sh, status_fn, current_fn, allow_git_mismatch=a.allow_git_mismatch,
                              models=[j["model"] for j in jobs],
                              check_images=[j["check_image"] for j in jobs])
    safe_print(format_checks(checks))
    if any(c.level == "fail" for c in checks):
        return 1
    for j in jobs:
        rc, _ = sh.ssh(f"test -e {PI_LAB}/{j['name']} || test -e {PI_LAB}/{j['name']}.log", 15)
        if rc != 1:
            safe_print(f"~/{PI_LAB}/{j['name']} already exists on the Pi: choose another --name"
                       if rc == 0 else f"could not check the Pi for {j['name']} (rc={rc})")
            return 1

    LOCAL_LAB.mkdir(parents=True, exist_ok=True)
    kept: dict = {}

    def verify_keep(d, mhz):                    # keep the full result for the summary
        kept["verify"] = verify_fn(d, mhz)
        return kept["verify"]

    orch = ex.Orchestrator(LOCAL_LAB, LOCAL_LAB / "results.jsonl", sh=sh, status_fn=status_fn,
                           align_fn=align_fn, verify_fn=verify_keep, current_fn=current_fn,
                           sleep=sleep, auto_power_cycle=0, log=safe_print,
                           pi_subdir=LAB_SUBDIR, run_timeout_s=a.timeout * 60)
    orch.provenance = {"git_host": facts["git_host"], "git_pi": facts["git_pi"]}
    if a.no_thermal_gate:
        orch.thermal_gate = lambda: None
    done: list[tuple[str, str]] = []
    try:
        for i, j in enumerate(jobs):
            kept.clear()
            safe_print(f"\n[{i + 1}/{len(jobs)}] {j['name']}: checking the recorder and the Pi"
                       + ("" if a.no_thermal_gate else ", waiting until the Pi is below "
                          f"{ex.THERMAL_MAX_C:.0f} C (up to 10 min)") + ", then run.py")
            safe_print(f"watch live: ssh {ex.PI} tail -f {PI_LAB}/{j['name']}.log")
            cell = {"cell_id": j["name"], "model_key": j["preset"] or "custom",
                    "cpu_mhz": j["cpu_mhz"], "repeat": 0, "round": 0, "args": j["args"]}
            res = orch.run_cell(cell, 1)
            res["lab"] = {"preset": j["preset"], "name": j["name"], "args": j["args"]}
            ex.append_jsonl(LOCAL_LAB / "results.jsonl", res)
            done.append((j["name"], "ok" if res["status"] == "ok" else f"FAILED at {res['stage']}"))
            safe_print(format_summary(j, res, kept.get("verify") or {}, facts)
                       if res["status"] == "ok" else format_failure(j, res))
            if res["status"] != "ok" and res["stage"] in STOP_STAGES and i + 1 < len(jobs):
                safe_print(f"Not starting the remaining presets after a '{res['stage']}' failure.")
                break
    except KeyboardInterrupt:
        name = jobs[len(done)]["name"] if len(done) < len(jobs) else jobs[-1]["name"]
        state, lines = pi_run_py(sh)
        safe_print("\nInterrupted.")
        if state == "busy":
            pids = ", ".join(s.split()[0] for s in lines)
            safe_print(f"run.py is still running on the Pi (pid {pids}) and will finish on its "
                       f"own; its data will be in ~/{PI_LAB}/{name}.\nTo stop it: {STOP_RUN}")
        elif state == "idle":
            safe_print("run.py is not running on the Pi.")
        else:
            safe_print(f"Could not check the Pi. To stop a run there: {STOP_RUN}")
        restore_pins(sh, a.keep_pinned, state)
        return 130
    restore_pins(sh, a.keep_pinned, pi_run_py(sh)[0])
    if len(jobs) > 1:
        safe_print("\n".join(["Summary"] + [f"  {n}  {s}" for n, s in done]
                             + [f"  {j['name']}  not run" for j in jobs[len(done):]]))
    return 0 if len(done) == len(jobs) and all(s == "ok" for _, s in done) else 1


def cmd_test(a, extra, sh, status_fn, current_fn, now) -> int:
    tokens = [*a.targets, *extra]
    for t in tokens:
        if t.startswith("/") or ".." in t:
            raise UsageError(f"{t!r}: paths must stay inside the project (e.g. tests/test_x.py)")
    args, whole = [], False
    for t in tokens or (["all"] if not a.host_only else []):
        if t == "all":
            args.append("tests/")
            whole = True
        else:
            args += PR1 if t == "pr1" else [t]
    if a.k:
        args += ["-k", a.k]
    out_dir = LOCAL_LAB / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%d-%H%M%S")
    sh = sh or LabShell()
    ok = True
    if not a.host_only:
        if sh.ssh("true", 15)[0] != 0:
            try:
                run_dir = status_fn().get("run_dir")
            except (OSError, ValueError):
                run_dir = None
            safe_print(f"Pi unreachable: {pi_down_detail(sh, current_fn, run_dir)}\n{PI_DOWN_HINT}")
            return 1
        idle = idle_check(sh, True)
        if idle.level != "ok":
            safe_print(f"Not starting the Pi tests: {idle.detail}. They would disturb a "
                       "measurement; wait for it to finish.")
            return 1
        note = None
        if whole and sh.ssh(f"test -f {ex.PI_DIR}/{KNOWN_FAIL_DATA}", 15)[0] != 0:
            args += ["--deselect", KNOWN_FAIL]
            note = (f"Note: deselected {KNOWN_FAIL}: it needs the old run folder "
                    f"{KNOWN_FAIL_DATA.rsplit('/', 1)[0]}, which is not on the Pi. That is "
                    "expected, not a problem.")
        cmd = (f"cd {ex.PI_DIR} && {ex.PI_UV} run --frozen --with pytest pytest -q "
               f"{shlex.join(args)}")
        safe_print(f"$ ssh {ex.PI} '{cmd}'\nrunning on the Pi (usually under a minute)...")
        rc, out = sh.ssh(cmd, a.timeout * 60)
        text = "\n".join(_lines(out))
        log = out_dir / f"{stamp}_pi.log"
        log.write_text(text + "\n")
        safe_print(text)
        safe_print(f"Pi tests: {parse_pytest_summary(text) or f'no pytest summary (rc={rc})'}"
                   f"  [{'ok' if rc == 0 else 'FAILED'}]")
        if note:
            safe_print(note)
        safe_print(f"Full output: {log}\nThe tests heated the Pi; the next run's thermal gate "
                   "may wait a few minutes.")
        ok = rc == 0
    if a.host or a.host_only:
        rc, out = sh.run([sys.executable, "-m", "pytest", "-q"], 900, cwd=ex.HERE)
        log = out_dir / f"{stamp}_host.log"
        log.write_text(out)
        safe_print(out.rstrip())
        if "No module named pytest" in out:
            safe_print("pytest is missing here: run `~/.local/bin/uv sync` in host/ "
                       "(or use ./lab).")
        safe_print(f"Host tests: {parse_pytest_summary(out) or f'no pytest summary (rc={rc})'}"
                   f"  [{'ok' if rc == 0 else 'FAILED'}]\nFull output: {log}")
        ok = ok and rc == 0
    return 0 if ok else 1


def preset_details(key: str) -> str:
    d = args_to_dict(matrix.MODELS[key])
    parts = [f"imgsz {d['--imgsz']}" if "--imgsz" in d else "",
             f"seq {d['--seq']}" if "--seq" in d else "",
             f"batch {d['--batch']}" if "--batch" in d else "",
             d.get("--optimizer", ""),
             f"serve {d['--serve']}" if d.get("--serve", "0") != "0" else "",
             "no bench" if "--no-bench" in d else f"bench {d.get('--bench', '3')} s",
             f"check {d['--check-image']}" if "--check-image" in d else ""]
    return ", ".join(p for p in parts if p)


def cmd_presets() -> int:
    durs: dict = {}
    try:
        for r in ex.load_jsonl(MATRIX_RESULTS):
            if (r.get("status") == "ok" and r.get("cpu_mhz") == matrix.MAIN_MHZ
                    and r.get("duration_s")):
                durs.setdefault(r.get("model_key"), []).append(r["duration_s"])
    except (OSError, ValueError):
        pass
    for title, train in (("Inference", False), ("Training", True)):
        unit = "steps" if train else "frames"
        safe_print(f"{title}\n  {'preset':<20} {'model':<31} {unit:>6}"
                   f"  {'~time':>9}  details")
        for k, toks in matrix.MODELS.items():
            d = args_to_dict(toks)
            if (d["--task"] == "train") != train:
                continue
            t = durs.get(k)
            took = f"{sum(t) / len(t) / 60:.1f} min" if t else "-"
            safe_print(f"  {k:<20} {d['--model']:<31} {d['--frames']:>6}  {took:>9}  "
                       f"{preset_details(k)}")
        safe_print("")
    safe_print(f"~time: mean of the matrix's ok cells at {matrix.MAIN_MHZ} MHz "
               f"({MATRIX_RESULTS}), when present.\nAll run with --sampler monsoon --fan-full "
               "--threads 4 --cpu-mhz 2400 unless overridden. Example: ./lab run yolo11n")
    return 0


# ---- command line --------------------------------------------------------------------------------
def _int(lo: int):
    def f(s):
        v = int(s)
        if v < lo:
            raise argparse.ArgumentTypeError(f"must be at least {lo}")
        return v
    return f


def _num(s: str) -> str:
    """A non-negative number, kept as typed (so `--bench 2` stays `2`, as in the matrix)."""
    try:
        if float(s) >= 0:
            return s
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"{s!r} is not a non-negative number")


RUN_EPILOG = """\
Output (always under runs/lab, never mixed with the matrix):
  here (EliteBook)  ~/power-measurement-pi/runs/lab/<name>/   + <name>.log, <name>.dmesg.log
  on the Pi         ~/Desktop/power-measurement-pi/runs/lab/<name>/
  every run         one line in ~/power-measurement-pi/runs/lab/results.jsonl
The default <name> is <preset or model>_<MHz>_<YYYYmmdd-HHMMSS>. An existing name is refused.

What happens: pre-flight checks (as `./lab check`; any [FAIL] stops), then per preset: recorder
recording? Pi reachable? wait until the Pi is below 50 C, run.py on the Pi (its output goes to
<name>.log there), copy back, align, verify, print a summary. After the last preset the Pi's clock
and fan go back to normal (unless --keep-pinned, or run.py is still running there).

Ctrl-C stops waiting here and tells you whether run.py is still running on the Pi and how to stop
it; it never kills it for you. This script never switches power.

Exit codes: 0 every run ok, 1 a check or run failed (the stage is printed), 2 bad command line,
130 interrupted.

Examples:
  ./lab run yolo11n
  ./lab run yolo11n train_lm64_adam train_resnet18_sgd   # 3-cell smoke, one after another
  ./lab run mbv3 --cpu-mhz 2000 --frames 100
  ./lab run train_lm64_adam --frames 200 --name adam_short
  ./lab run yolo11n --frames 5 --serve 2 --bench 0       # quick look, about a minute
  ./lab run --task train --model torchvision:resnet18 --batch 4 --imgsz 128 --optimizer adam
  ./lab run resnet18 --dry-run                           # show the commands only
Many repeats or clock sweeps: experiment.py plan --only <preset> (README 5.4).
"""


def build_parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(prog="lab.py", description=__doc__, formatter_class=raw)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{check,run,test,presets}")

    c = sub.add_parser("check", formatter_class=raw, help="pre-flight checks",
                       description="Check that everything is ready for a measurement. Read-only: "
                       "nothing is changed on any machine.\n\nChecks: recorder recording at "
                       ">= 4900 Hz and not near its ~9.9-day limit; Pi reachable\n(if not: ping, "
                       "the Monsoon current and what to do); temperature; throttling; the "
                       "pinning helper;\nthe same git commit on both machines; nothing else "
                       "running (run.py or pytest on the Pi,\nthe matrix here).",
                       epilog="Exit codes: 0 ready (warnings allowed), 1 at least one [FAIL].\n\n"
                       "Examples:\n  ./lab check\n  ./lab check --allow-git-mismatch")
    c.add_argument("--allow-git-mismatch", action="store_true",
                   help="report different git commits on the Pi and here as a warning, not a "
                   "failure")

    r = sub.add_parser("run", formatter_class=raw, help="measure one or more workloads",
                       description="Measure workloads on the Pi with the Monsoon: either "
                       "ready-made "
                       "presets (the matrix models, `./lab presets`)\nor your own with --task and "
                       "--model. Any option below overrides that value for every preset given;\n"
                       "with no options a preset runs exactly like a matrix cell at 2400 MHz.",
                       epilog=RUN_EPILOG)
    r.add_argument("presets", nargs="*", metavar="PRESET",
                   help="one or more preset names, run one after another (e.g. yolo11n lm64 "
                   "train_resnet18_adam). Leave out to use --task/--model.")
    w = r.add_argument_group("workload (only typed options override the preset)")
    w.add_argument("--task", choices=["image", "lm", "train"],
                   help="custom workload, together with --model: image = image inference "
                   "(detection or classification), lm = language-model inference (prefill + "
                   "decode), train = training steps")
    w.add_argument("--model", metavar="NAME",
                   help="custom workload, together with --task. image: yolo11n.pt, yolov8n.pt "
                   "or torchvision:<name> (e.g. torchvision:resnet50); lm: minigpt, "
                   "minigpt-explicit or hf:<name>; train: torchvision:<name> or "
                   "minigpt-explicit. *.pt weights not on the Pi are downloaded from the "
                   "internet on first use; hf: models download inside load_weights (needs "
                   "internet on the Pi, and it is measured)")
    w.add_argument("--frames", type=_int(1), metavar="N",
                   help="measured iterations per mode (for train: optimizer steps). More gives "
                   "steadier averages but a longer run (preset values aim at 10-30 s; run.py "
                   "default 30)")
    w.add_argument("--warmup", type=_int(0), metavar="N",
                   help="unmeasured warm-up passes before measuring (default: preset, else 10)")
    w.add_argument("--threads", type=_int(1), metavar="N",
                   help="PyTorch CPU threads (default 4, the Pi 5 has 4 cores)")
    w.add_argument("--imgsz", type=_int(32), metavar="PX",
                   help="input image size in pixels, image tasks and image training (default: "
                   "preset, else 640)")
    w.add_argument("--seq", type=_int(1), metavar="N",
                   help="prompt length in tokens, language models (default: preset, else 64)")
    w.add_argument("--batch", type=_int(1), metavar="N",
                   help="training batch size, --task train only, at least 2 (default: preset, "
                   "else 8)")
    w.add_argument("--optimizer", choices=["sgd", "adam"],
                   help="training optimizer: sgd (momentum 0.9) or adam (default: preset, "
                   "else sgd)")
    w.add_argument("--serve", type=_int(0), metavar="N",
                   help="inference only: N single requests with idle gaps after the modes "
                   "(phase 'serve'); 0 = none (presets: 10, training: 0)")
    w.add_argument("--serve-gap-ms", type=_num, metavar="MS",
                   help="idle time between serve requests (default 200)")
    w.add_argument("--bench", type=_num, metavar="SECONDS",
                   help="seconds per isolated layer benchmark (bench:* phases, per-layer-type "
                   "power); 0 = no benchmarks (presets: 1-2 s, training: none)")
    w.add_argument("--check-image", metavar="PATH",
                   help="detection sanity check: this image (relative to "
                   "~/Desktop/power-measurement-pi on the Pi) must give a detection with "
                   "confidence >= --conf, or the run fails (YOLO presets: bus.jpg)")
    w.add_argument("--no-check-image", action="store_true",
                   help="drop the preset's detection sanity check")
    w.add_argument("--conf", type=_num, metavar="X",
                   help="confidence threshold for --check-image (default 0.25)")
    w.add_argument("--live-camera", action="store_true",
                   help="use real Pi-camera frames instead of a fixed test image")
    m = r.add_argument_group("conditions and output")
    m.add_argument("--cpu-mhz", type=int, choices=MHZ_CHOICES, default=matrix.MAIN_MHZ,
                   metavar="MHZ",
                   help="lock all 4 cores at this clock, 1500-2400 in steps of 100 (default "
                   "2400; the matrix also uses 2000 and 1500). The fan always runs at full speed")
    m.add_argument("--name", metavar="NAME",
                   help="run folder name (one preset only): letters, digits, '.', '_', '-'. "
                   "Default <preset or model>_<MHz>_<YYYYmmdd-HHMMSS>; an existing name is refused")
    f = r.add_argument_group("safety and flow")
    f.add_argument("--timeout", type=_int(1), default=40, metavar="MIN",
                   help="give up waiting for run.py after this many minutes (default 40)")
    f.add_argument("--no-thermal-gate", action="store_true",
                   help="start at once even if the Pi is warm. By default each run waits until "
                   "the Pi is below 50 C (up to 10 min), as in the matrix")
    f.add_argument("--keep-pinned", action="store_true",
                   help="leave the clock and fan pinned afterwards (default: restore them when "
                   "no run.py is running on the Pi)")
    f.add_argument("--allow-git-mismatch", action="store_true",
                   help="run even if the Pi and the EliteBook are on different commits (both "
                   "are recorded in results.jsonl)")
    f.add_argument("--dry-run", action="store_true",
                   help="only print the run.py and Pi commands and the folders; contacts no "
                   "machine")

    t = sub.add_parser("test", formatter_class=raw, help="run the unit tests on the Pi / here",
                       description="Run the Pi's unit tests (tests/) over ssh, and/or the host "
                       "tests (host/tests) here.\nRefused while run.py, pytest or the matrix is "
                       "running: the tests load the CPU and heat the Pi\n(the next run's thermal "
                       "gate may then wait a few minutes). Output is saved to\n"
                       "runs/lab/tests/<time>_pi.log and _host.log.",
                       epilog="If the whole suite runs and runs/fix_abab_smoke is missing on the "
                       "Pi, tests/test_phases.py::\ntest_no_idle_contamination is deselected "
                       "(it needs that old data; expected).\n\nExit codes: 0 all passed, 1 a "
                       "test failed or the tests could not run, 2 bad command line.\n\n"
                       "Examples:\n  ./lab test                        # whole Pi suite\n"
                       "  ./lab test pr1                    # the 3 tests from the PR #1 review\n"
                       "  ./lab test tests/test_pinning.py -x\n  ./lab test -k topk\n"
                       "  ./lab test --host                 # Pi suite, then host suite\n"
                       "  ./lab test --host-only")
    t.add_argument("targets", nargs="*", metavar="TARGET",
                   help="all (default: the whole tests/ suite), pr1 (the 3 PR #1 review tests), "
                   "or pytest paths/node ids/options such as tests/test_pinning.py or -x. "
                   "Paths must stay inside the project")
    t.add_argument("-k", metavar="EXPR", help="only tests whose name matches EXPR (pytest -k)")
    t.add_argument("--host", action="store_true", help="also run the host suite here")
    t.add_argument("--host-only", action="store_true",
                   help="only the host suite; don't contact the Pi")
    t.add_argument("--timeout", type=_int(1), default=20, metavar="MIN",
                   help="give up on the Pi tests after this many minutes (default 20)")

    sub.add_parser("presets", formatter_class=raw, help="list the ready-made workloads",
                   description="List the 14 matrix workloads that `./lab run` accepts by name, "
                   "with their typical duration\nfrom the matrix results when available.",
                   epilog="Examples:\n  ./lab presets\n  ./lab run lm64")
    return ap


def main(argv=None, *, sh=None, status_fn=None, current_fn=None, align_fn=None, verify_fn=None,
         now=None, sleep=None) -> int:
    ap = build_parser()
    a, extra = ap.parse_known_args(argv)
    if extra and a.cmd != "test":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    status_fn = status_fn or ex.recorder_status
    current_fn = current_fn or ex.halted_current
    now = now or datetime.datetime.now
    try:
        if a.cmd == "presets":
            return cmd_presets()
        if a.cmd == "check":
            sh = sh or LabShell()
            checks, _ = preflight(sh, status_fn, current_fn,
                                  allow_git_mismatch=a.allow_git_mismatch)
            safe_print(format_checks(checks))
            return 1 if any(c.level == "fail" for c in checks) else 0
        if a.cmd == "run":
            return cmd_run(a, sh, status_fn, current_fn, align_fn, verify_fn, now,
                           sleep or time.sleep)
        return cmd_test(a, extra, sh, status_fn, current_fn, now)
    except UsageError as e:
        print(f"lab.py {a.cmd}: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        safe_print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
