#!/usr/bin/env python3
"""SP3 experiment orchestrator (EliteBook). Resumable: one manifest, one append-only results log.

    uv run python experiment.py plan [--smoke|--probe] [--only yolo11n,lm64] --out ../runs/matrix/manifest.jsonl
    uv run python experiment.py plan --v2 {A,A-input,B,C,D,soak,ref,all,pilot} [--include-fp16]
        [--include-optional] [--no-concurrent] [--include-int8d] [--include-prune] --out ...
    uv run python experiment.py run --manifest ../runs/matrix/manifest.jsonl [--auto-power-cycle 3]
    uv run python experiment.py status --manifest ../runs/matrix/manifest.jsonl

Per cell: recorder recording -> Pi reachable -> Pi below 50 C -> run.py on the Pi (output to a
log file there) with `dmesg -W` streamed here -> rsync the run back -> align.py -> verify_cell.py
-> append one line to results.jsonl. The recorder is never stopped, except by the opt-in
power-cycle recovery for a halted Pi. A failed cell is retried once at the end of its round.
Cells already `ok` in results.jsonl are skipped, so `run` can be restarted at any time.

Before each cell: the recorder's n_total must be rising (read twice, 2 s apart) with more than an
hour of samples left before the 32-bit ceiling (the recorder once stalled silently at 2^32-1 while
/status kept saying recording), and the EliteBook runs dir and the Pi project dir need free disk
(more for `family: "soak"` cells). A cell's `timeout_s` overrides RUN_TIMEOUT_S. run.py exit codes
3/4 (thermal abort / hard stop) end the cell at stage "thermal_abort" with the Pi's
thermal_abort.json as the error; it is not retried and the matrix goes on with the next cell.

Stale run.py: before a cell the Pi must not be running run.py (stage "pi_busy": ends the matrix,
a leftover run.py would corrupt the next cell). When the cell's ssh times out (rc 124) the Pi's
run.py gets SIGINT and up to 60 s to exit (its finally restores the fan and stops perf); the cell
ends at stage "timeout" and says whether it exited. If `dmesg -W` of the cell shows "Undervoltage
detected" the cell fails at stage "undervoltage" (retried like any failure, matrix goes on).

Halted-Pi recovery (--auto-power-cycle N, at most N times per invocation): only when the Pi has
not answered for PI_DOWN_S and the Monsoon shows halted-level current (mean < 0.4 A and max
< 0.6 A over the last 5 s). Then: stop recorder, vout_off.py --confirm-pi-halted, start the
recorder again (the Pi boots when Vout comes back), wait for the Pi.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI = "maccpi@192.168.0.243"
PI_HOST = "192.168.0.243"
PI_DIR = "Desktop/power-measurement-pi"
PI_UV = "~/.local/bin/uv"
RECORDER_URL = "http://localhost:8051"
RECORDER_CMD = ["systemd-run", "--user", "--unit=monsoon-recorder", "--same-dir", "--collect",
                os.path.expanduser("~/.local/bin/uv"), "run", "python", "recorder.py",
                "--vout", "5.35", "--current-limit", "6"]
THERMAL_MAX_C = 50.0
THERMAL_WAIT_S = 600
RUN_TIMEOUT_S = 40 * 60
PI_DOWN_S = 180
MIN_FREE_GB = 10            # EliteBook runs dir
PI_MIN_FREE_GB = 5          # Pi project dir
SOAK_EXTRA_GB = 2           # added to both minima for family "soak" cells
THERMAL_EXIT_CODES = (3, 4)  # run.py: thermal abort (85 C), hard stop (90 C)
# `pytho[n]`: the pattern must not match the ssh command line that carries it
PI_RUN_PGREP = "pgrep -af 'pytho[n] src/run.py'"
PI_RUN_PKILL_INT = "pkill -INT -f 'pytho[n] src/run.py'"
RUN_EXIT_WAIT_S, RUN_EXIT_POLL_S = 60, 5
DISK_RETRY_WAIT_S = 10
ABORT_STAGES = ("pi_down", "recorder", "disk", "pi_busy")      # these stop the matrix
UNDERVOLTAGE_MARK = "undervoltage detected"                    # kernel: "hwmon hwmon3: Undervoltage detected!"
UNDERVOLTAGE_MAX_LINES = 5
SAMPLE_CEILING = 2 ** 32 - 1            # the Monsoon library stops here
STALL_MARGIN_SAMPLES = 5000 * 3600      # refuse to start with less than an hour left
STALL_READ_GAP_S = 2
HALT_MEAN_A, HALT_MAX_A = 0.4, 0.6
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=8"]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_jsonl(path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cell_state(results: list[dict]) -> dict:
    """cell_id -> {"ok": bool, "attempts": int, "last": row}."""
    st: dict = {}
    for r in results:
        s = st.setdefault(r["cell_id"], {"ok": False, "attempts": 0, "last": None})
        s["attempts"] += 1
        s["ok"] = s["ok"] or r["status"] == "ok"
        s["last"] = r
    return st


def pending(manifest: list[dict], results: list[dict], max_attempts: int) -> list[dict]:
    st = cell_state(results)
    return [c for c in manifest
            if not st.get(c["cell_id"], {}).get("ok")
            and st.get(c["cell_id"], {}).get("attempts", 0) < max_attempts]


def pi_command(cell: dict, subdir: str = "runs/matrix") -> str:
    """run.py for one cell, output to <subdir>/<cell_id> under PI_DIR. `subdir` and the cell id
    are not quoted: callers pass validated names (matrix ids, lab.py's name pattern)."""
    out = f"{subdir}/{cell['cell_id']}"
    args = shlex.join([*cell["args"], "--out", out])
    return (f"cd {PI_DIR} && mkdir -p {subdir} && rm -rf {out} && "
            f"{{ [ ! -f {out}.log ] || mv {out}.log {out}.log.$(date +%s); }} && "
            f"{PI_UV} run --frozen python src/run.py {args} > {out}.log 2>&1")


def parse_temp(out: str) -> float | None:
    """`temp=45.5'C` -> 45.5 (other lines, e.g. ssh locale warnings, are ignored)."""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("temp="):
            try:
                return float(line.split("=")[1].split("'")[0])
            except (IndexError, ValueError):
                return None
    return None


class Shell:
    """subprocess wrapper; tests replace it."""

    def run(self, cmd: list[str], timeout: float, cwd=None) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            return 124, f"timeout after {timeout}s: {e}"
        return p.returncode, (p.stdout + p.stderr)

    def popen(self, cmd: list[str], out_path):
        f = open(out_path, "w")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f

    def ssh(self, cmd: str, timeout: float) -> tuple[int, str]:
        return self.run([*SSH, PI, cmd], timeout)

    def ping(self) -> bool:
        return self.run(["ping", "-c", "1", "-W", "2", PI_HOST], 10)[0] == 0


def recorder_status() -> dict:
    with urllib.request.urlopen(RECORDER_URL + "/status", timeout=5) as r:
        return json.loads(r.read())


def recorder_stalled(st_a: dict, st_b: dict) -> str | None:
    """Why the recorder cannot be trusted for another cell, or None. `st_a`, `st_b` are two /status
    reads a couple of seconds apart: n_total must have increased (the recorder has stalled
    silently at the 32-bit ceiling while /status kept saying recording) and there must be more
    than an hour of samples left before that ceiling."""
    restart = "restart the recorder safely, README section 8"
    a, b = st_a.get("n_total"), st_b.get("n_total")
    if not isinstance(a, int) or not isinstance(b, int):
        return f"recorder /status has no usable n_total ({a!r}, {b!r}); cannot tell it is sampling"
    if b <= a:
        return f"recorder is not sampling: n_total stayed at {b} between two /status reads; {restart}"
    if b >= SAMPLE_CEILING - STALL_MARGIN_SAMPLES:
        return (f"recorder n_total {b} is within an hour of the 32-bit ceiling {SAMPLE_CEILING} "
                f"where it stops silently; {restart}")
    return None


def parse_df_avail_gb(out: str) -> float | None:
    """Available GiB from `df -Pk <dir>` output (last line, 4th column; ssh warnings are ignored)."""
    for line in reversed(out.splitlines()):
        f = line.split()
        if len(f) >= 6 and f[1].isdigit() and f[3].isdigit():
            return int(f[3]) / 1024 ** 2
    return None


def undervoltage_lines(dmesg_text: str, limit: int = UNDERVOLTAGE_MAX_LINES) -> list[str]:
    """The kernel's "Undervoltage detected" lines of a `dmesg -W` capture (at most `limit`)."""
    hits = [ln.strip() for ln in dmesg_text.splitlines() if UNDERVOLTAGE_MARK in ln.lower()]
    return hits[:limit]


def local_free_gb(path) -> float:
    """Free GiB on the filesystem holding `path` (or its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1024 ** 3


def halted_current(run_dir) -> tuple[float, float]:
    """(mean, max) input current over the last 5 s of the recording, finite samples only.
    (nan, nan) when fewer than half the samples are finite; recover_pi then refuses to cycle."""
    import numpy as np
    import store
    s = store.open_samples(run_dir)
    n = len(s)
    I = np.array(s[max(0, n - 25000):n], dtype=np.float64)[:, 0]
    fin = I[np.isfinite(I)]
    if I.size == 0 or fin.size < 0.5 * I.size:
        return float("nan"), float("nan")
    return float(fin.mean()), float(fin.max())


def halt_confirmed(mean_a: float, max_a: float) -> bool:
    """True only for valid (finite, non-negative) readings below both halt limits. Fails closed."""
    return (math.isfinite(mean_a) and math.isfinite(max_a) and mean_a >= 0 and max_a >= 0
            and mean_a < HALT_MEAN_A and max_a < HALT_MAX_A)


def exit_code(summary: dict) -> int:
    """0 only when the matrix was not aborted and every cell is ok."""
    return 0 if summary["aborted"] is None and summary["ok"] == summary["cells"] else 3


def parse_git_head(out: str) -> str | None:
    """The commit hash line from `git rev-parse --short HEAD` output. ssh stdout and stderr are
    merged, so warnings (e.g. the Pi's setlocale message) can come before or after it."""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if 7 <= len(line) <= 40 and all(c in "0123456789abcdef" for c in line):
            return line
    return None


class RecorderUnavailable(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, runs_dir, results_path, sh=None, status_fn=recorder_status,
                 align_fn=None, verify_fn=None, current_fn=halted_current, sleep=time.sleep,
                 auto_power_cycle: int = 0, log=print, pi_subdir: str = "runs/matrix",
                 run_timeout_s: float | None = None, local_free_fn=None):
        self.runs = Path(runs_dir)
        self.results_path = Path(results_path)
        self.sh = sh or Shell()
        self.status_fn = status_fn
        self.align_fn = align_fn
        self.verify_fn = verify_fn
        self.current_fn = current_fn
        self.sleep = sleep
        self.cycles_left = auto_power_cycle
        self.log = log
        self.pi_subdir = pi_subdir          # run folder on the Pi, relative to PI_DIR
        self.run_timeout_s = run_timeout_s  # None: RUN_TIMEOUT_S, read at call time
        self.local_free_fn = local_free_fn  # free GiB on the runs dir; None: local_free_gb,
                                            # resolved at call time
        self.provenance: dict = {}

    # ---- health -------------------------------------------------------------------------------
    def pi_up(self) -> bool:
        return self.sh.ssh("true", 30)[0] == 0

    def wait_pi(self, timeout_s: float) -> bool:
        t_end = time.monotonic() + timeout_s
        while True:
            if self.pi_up():
                return True
            if time.monotonic() >= t_end:
                return False
            self.sleep(10)

    def recover_pi(self) -> str | None:
        """Make the Pi reachable; returns an error string or None. Power-cycles only if allowed
        and both halt checks agree."""
        if self.wait_pi(PI_DOWN_S):
            return None
        if self.cycles_left <= 0:
            return f"Pi unreachable for {PI_DOWN_S}s (auto power cycle disabled or used up)"
        if self.sh.ping():
            return "Pi answers ping but not ssh; not power-cycling"
        try:
            st = self.status_fn()
            run_dir = st["run_dir"]
        except (OSError, ValueError, KeyError) as e:
            raise RecorderUnavailable(f"recorder unreachable during Pi recovery: {e}") from e
        try:
            mean_a, max_a = self.current_fn(run_dir)
        except Exception as e:  # noqa: BLE001 - unreadable current means no power cycle
            return f"halt current unreadable ({type(e).__name__}: {e}); not power-cycling"
        if not halt_confirmed(mean_a, max_a):
            return (f"Pi unreachable but current reads {mean_a:.2f} A (max {max_a:.2f} A); "
                    "halt not confirmed, not power-cycling")
        self.cycles_left -= 1
        self.log(f"[power-cycle] Pi halted (I mean {mean_a:.3f} A, max {max_a:.3f} A); "
                 f"cycling Vout ({self.cycles_left} left)")
        steps = [(["systemctl", "--user", "stop", "monsoon-recorder"], 60),
                 ([sys.executable, "vout_off.py", "--confirm-pi-halted"], 120),
                 (RECORDER_CMD, 60)]
        for i, (cmd, to) in enumerate(steps):
            if i:
                self.sleep(5)
            rc, out = self.sh.run(cmd, to, cwd=HERE)
            if rc != 0:
                return f"power cycle step {cmd[0:3]} failed rc={rc}: {out[-300:]}"
        append_jsonl(self.runs / "power_cycles.jsonl",
                     {"t": _now(), "mean_A": mean_a, "max_A": max_a})
        self.sleep(10)
        if not self.wait_pi(300):
            return "Pi did not come back within 300 s after the power cycle"
        self.sleep(20)
        return None

    def run_py_running(self) -> tuple[bool, str]:
        """(running, detail) for run.py on the Pi. Fails closed: an unanswered check counts as
        running (pgrep exits 1 for "no match"; anything else, incl. ssh's 255, is unknown)."""
        rc, out = self.sh.ssh(PI_RUN_PGREP, 30)
        if rc == 1:
            return False, ""
        if rc == 0:
            return True, out.strip()[-300:]
        return True, f"could not check for a running run.py (rc={rc}): {out.strip()[-200:]}"

    def stop_run_py(self) -> bool:
        """SIGINT the Pi's run.py (its finally restores the fan and stops perf) and wait up to
        RUN_EXIT_WAIT_S for it to exit. True if it is gone."""
        self.sh.ssh(PI_RUN_PKILL_INT, 30)
        waited = 0
        while True:
            running, _ = self.run_py_running()
            if not running:
                return True
            if waited >= RUN_EXIT_WAIT_S:
                return False
            self.sleep(RUN_EXIT_POLL_S)
            waited += RUN_EXIT_POLL_S

    def thermal_gate(self) -> str | None:
        t_end = time.monotonic() + THERMAL_WAIT_S
        temp = None
        while True:
            rc, out = self.sh.ssh("vcgencmd measure_temp", 30)
            temp = parse_temp(out) if rc == 0 else None
            if temp is not None and temp < THERMAL_MAX_C:
                return None
            if time.monotonic() >= t_end:
                return f"Pi still at {temp} C after {THERMAL_WAIT_S}s (gate {THERMAL_MAX_C} C)"
            self.sleep(10)

    def disk_check(self, cell: dict) -> str | None:
        """Free space on the EliteBook runs dir and the Pi project dir; soak cells need more."""
        extra = SOAK_EXTRA_GB if cell.get("family") == "soak" else 0
        need_here, need_pi = MIN_FREE_GB + extra, PI_MIN_FREE_GB + extra
        here = (self.local_free_fn or local_free_gb)(self.runs)
        if here < need_here:
            return f"EliteBook {self.runs}: {here:.1f} GB free, need {need_here} GB"
        for attempt in (1, 2):         # one retry: a transient ssh failure must not stop a matrix
            rc, out = self.sh.ssh(f"df -Pk {PI_DIR}", 30)
            pi = parse_df_avail_gb(out) if rc == 0 else None
            if pi is not None or attempt == 2:
                break
            self.sleep(DISK_RETRY_WAIT_S)
        if pi is None:
            return f"could not read free space on the Pi {PI_DIR} (rc={rc}): {out[-200:]}"
        if pi < need_pi:
            return f"Pi {PI_DIR}: {pi:.1f} GB free, need {need_pi} GB"
        return None

    # ---- one cell ----------------------------------------------------------------------------
    def run_cell(self, cell: dict, attempt: int) -> dict:
        cid = cell["cell_id"]
        res = {"cell_id": cid, "model_key": cell["model_key"], "cpu_mhz": cell["cpu_mhz"],
               "repeat": cell["repeat"], "round": cell["round"], "attempt": attempt,
               "started": _now(), "status": "fail", "stage": None, "error": None,
               **self.provenance}
        t0 = time.monotonic()
        local = self.runs / cid

        def fail(stage, err):
            res.update(stage=stage, error=str(err)[-2000:], ended=_now(),
                       duration_s=round(time.monotonic() - t0, 1))
            return res

        try:
            st = self.status_fn()
        except (OSError, ValueError) as e:
            return fail("recorder", f"recorder unreachable: {e}")
        if not st.get("recording"):
            return fail("recorder", f"recorder not recording: {st}")
        self.sleep(STALL_READ_GAP_S)
        try:
            st_b = self.status_fn()
        except (OSError, ValueError) as e:
            return fail("recorder", f"recorder unreachable: {e}")
        err = recorder_stalled(st, st_b)
        if err:
            return fail("recorder", err)
        try:
            err = self.recover_pi()
        except RecorderUnavailable as e:
            return fail("recorder", str(e))
        if err:
            return fail("pi_down", err)
        try:
            st = self.status_fn()      # a power cycle starts a new recording
            res["recorder_dir"] = st["run_dir"]
        except (OSError, ValueError, KeyError) as e:
            return fail("recorder", f"recorder unreachable: {e}")
        err = self.disk_check(cell)
        if err:
            return fail("disk", err)
        busy, detail = self.run_py_running()
        if busy:
            return fail("pi_busy", f"run.py is already running on the Pi: {detail}")
        err = self.thermal_gate()
        if err:
            return fail("thermal", err)

        local.mkdir(parents=True, exist_ok=True)
        dm, dm_f = self.sh.popen([*SSH, PI, "dmesg -W"], local.parent / f"{cid}.dmesg.log")
        try:
            t_run = time.monotonic()
            timeout = cell.get("timeout_s") or (
                RUN_TIMEOUT_S if self.run_timeout_s is None else self.run_timeout_s)
            rc, out = self.sh.ssh(pi_command(cell, self.pi_subdir), timeout)
            res["run_s"] = round(time.monotonic() - t_run, 1)
            if rc == 124:              # local timeout: the ssh died, run.py may still be going
                res["run_exited"] = self.stop_run_py()
        finally:
            dm.terminate()
            dm_f.close()
        res["run_rc"] = rc
        # always fetch what exists (log + run dir) for diagnosis
        self.sh.run(["rsync", "-a", f"{PI}:{PI_DIR}/{self.pi_subdir}/{cid}.log",
                     str(local.parent / f"{cid}.log")], 300)
        rrc, rout = self.sh.run(["rsync", "-a", f"{PI}:{PI_DIR}/{self.pi_subdir}/{cid}/",
                                 str(local) + "/"], 600)
        if rc in THERMAL_EXIT_CODES:        # run.py ended itself; the Pi is up
            f = local / "thermal_abort.json"
            try:
                what = json.dumps(json.loads(f.read_text()))
            except (OSError, ValueError):
                what = f"run.py rc={rc} (thermal) but thermal_abort.json is missing or unreadable"
            return fail("thermal_abort", what)
        if rc == 124:
            gone = ("and exited" if res["run_exited"] else
                    f"and is STILL RUNNING {RUN_EXIT_WAIT_S} s later (the next cell will stop "
                    "the matrix at pi_busy)")
            return fail("timeout", f"run.py exceeded {timeout:.0f} s; SIGINT sent {gone}: "
                                   f"{out[-300:]}")
        if rc != 0:
            if not self.pi_up():
                return fail("pi_crash", f"Pi went down during the run (rc={rc}): {out[-500:]}")
            tail = (local.parent / f"{cid}.log")
            msg = tail.read_text()[-1500:] if tail.exists() else out[-1500:]
            return fail("run", f"run.py rc={rc}: {msg}")
        if rrc != 0:
            return fail("rsync", rout)
        try:
            uv = undervoltage_lines((local.parent / f"{cid}.dmesg.log").read_text(errors="replace"))
        except OSError:
            uv = []
        if uv:
            return fail("undervoltage", "kernel reported under-voltage during the run:\n"
                                        + "\n".join(uv))

        try:
            ali = self.align_fn(st["run_dir"], local, local)
        except Exception as e:  # noqa: BLE001 - any align failure fails the cell
            return fail("align", f"{type(e).__name__}: {e}")
        res["align"] = {k: ali.get(k) for k in ("max_residual_ms", "rms_residual_ms",
                                                "drift_ppm", "slip_warning",
                                                "slip_samples_total", "n_edges")}
        try:
            ver = self.verify_fn(local, cell["cpu_mhz"])
        except Exception as e:  # noqa: BLE001
            return fail("verify", f"{type(e).__name__}: {e}")
        res["verify"] = {k: ver.get(k) for k in ("errors", "warnings", "mean_P_W", "step_P_W",
                                                 "baseline_P_W", "step_coverage",
                                                 "energy_total_J")}
        if not ver.get("ok"):
            return fail("verify", "; ".join(ver.get("errors", [])))
        res.update(status="ok", ended=_now(), duration_s=round(time.monotonic() - t0, 1))
        return res

    # ---- matrix -------------------------------------------------------------------------------
    def run(self, manifest: list[dict], max_attempts: int = 2) -> dict:
        self.runs.mkdir(parents=True, exist_ok=True)
        todo = pending(manifest, load_jsonl(self.results_path), max_attempts)
        rounds = sorted({c["round"] for c in todo})
        self.log(f"[matrix] {len(todo)} cells pending in {len(rounds)} rounds")
        aborted = None
        for rnd in rounds:
            queue = [c for c in todo if c["round"] == rnd]
            while queue and not aborted:
                retry = []
                for cell in queue:
                    st = cell_state(load_jsonl(self.results_path)).get(cell["cell_id"], {})
                    attempt = st.get("attempts", 0) + 1
                    self.log(f"[cell] {cell['cell_id']} attempt {attempt} ({_now()})")
                    res = self.run_cell(cell, attempt)
                    append_jsonl(self.results_path, res)
                    self.log(f"[cell] {cell['cell_id']} -> {res['status']}"
                             + (f" at {res['stage']}: {str(res['error'])[:300]}"
                                if res["status"] != "ok" else f" in {res['duration_s']} s"))
                    if res["status"] != "ok":
                        if res["stage"] in ABORT_STAGES:
                            aborted = f"{res['stage']}: {res['error']}"
                            break
                        if attempt < max_attempts and res["stage"] != "thermal_abort":
                            retry.append(cell)
                queue = retry
            if aborted:
                break
        final = cell_state(load_jsonl(self.results_path))
        ids = [c["cell_id"] for c in manifest]
        summary = {"cells": len(ids), "ok": sum(1 for i in ids if final.get(i, {}).get("ok")),
                   "aborted": aborted}
        self.log(f"[matrix] done: {summary}")
        return summary


def status_report(manifest: list[dict], results: list[dict]) -> str:
    st = cell_state(results)
    ids = [c["cell_id"] for c in manifest]
    ok = [i for i in ids if st.get(i, {}).get("ok")]
    failed = [i for i in ids if i in st and not st[i]["ok"]]
    durs = [r["duration_s"] for r in results if r.get("status") == "ok" and r.get("duration_s")]
    mean = sum(durs) / len(durs) if durs else None
    left = len(ids) - len(ok)
    lines = [f"cells {len(ids)}  ok {len(ok)}  failed {len(failed)}  not run "
             f"{len(ids) - len(ok) - len(failed)}"]
    if mean:
        lines.append(f"mean ok cell {mean / 60:.1f} min; remaining ~{left * mean / 3600:.1f} h")
    by_stage: dict = {}
    for i in failed:
        by_stage.setdefault(st[i]["last"]["stage"], []).append(i)
    for stage, cids in sorted(by_stage.items(), key=lambda kv: str(kv[0])):
        lines.append(f"  failed at {stage}: {', '.join(cids)}")
    slips = sum(1 for r in results if r.get("status") == "ok"
                and (r.get("align") or {}).get("slip_warning"))
    lines.append(f"ok cells with slip_warning: {slips}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI
    import matrix_v2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--probe", action="store_true", help="every model once at 2400 MHz")
    p.add_argument("--only", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--v2", choices=matrix_v2.CHOICES, default=None,
                   help="dataset v2 family (host/matrix_v2.py) instead of the v1 matrix")
    for flag, what in (("fp16", "B: fp16 yolo11n cells"), ("optional", "C: threads x yolo11m"),
                       ("int8d", "B: int8-dynamic cells"),
                       ("prune", "B: pruned yolo11n cells")):
        p.add_argument(f"--include-{flag}", action="store_true", help=f"with --v2, add {what}")
    p.add_argument("--no-concurrent", action="store_true",
                   help="with --v2, leave out the D concurrency cells (run.py --concurrent)")
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--results", default=None, help="default: results.jsonl next to the manifest")
    r.add_argument("--max-attempts", type=int, default=2)
    r.add_argument("--auto-power-cycle", type=int, default=0,
                   help="allow up to N guarded Vout power cycles of a halted Pi")
    r.add_argument("--allow-git-mismatch", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", default=None)
    a = ap.parse_args()

    if a.cmd == "plan":
        only = [x for x in a.only.split(",") if x]
        if a.v2:
            if a.smoke or a.probe or only:
                raise SystemExit("--v2 cannot be combined with --smoke, --probe or --only")
            rows = matrix_v2.cells(a.v2, seed=a.seed, include_fp16=a.include_fp16,
                                   include_optional=a.include_optional,
                                   include_concurrent=not a.no_concurrent,
                                   include_int8d=a.include_int8d, include_prune=a.include_prune)
        else:
            if any((a.include_fp16, a.include_optional, a.no_concurrent, a.include_int8d,
                    a.include_prune)):
                raise SystemExit("--include-* options need --v2")
            import matrix
            rows = matrix.cells(seed=a.seed, smoke=a.smoke, only=only or None, probe=a.probe)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite a manifest")
        out.write_text("".join(json.dumps(c) + "\n" for c in rows))
        print(f"wrote {len(rows)} cells to {out}")
        return

    manifest = load_jsonl(a.manifest)
    results = Path(a.results or Path(a.manifest).with_name("results.jsonl"))
    if a.cmd == "status":
        print(status_report(manifest, load_jsonl(results)))
        return

    from align import align
    from verify_cell import verify
    import logging
    runs = Path(a.manifest).resolve().parent
    logging.basicConfig(filename=runs / "experiment.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")

    def log(msg):
        print(msg, flush=True)
        logging.info(msg)

    sh = Shell()
    host_head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    rc, out = sh.ssh(f"git -C {PI_DIR} rev-parse --short HEAD", 30)
    pi_head = parse_git_head(out) if rc == 0 else None
    if pi_head != host_head and not a.allow_git_mismatch:
        raise SystemExit(f"git mismatch: host {host_head} vs Pi {pi_head} "
                         "(pull both, or --allow-git-mismatch)")
    orch = Orchestrator(runs, results, sh=sh, align_fn=align,
                        verify_fn=lambda d, mhz: verify(d, mhz),
                        auto_power_cycle=a.auto_power_cycle, log=log)
    orch.provenance = {"git_host": host_head, "git_pi": pi_head}
    try:
        summary = orch.run(manifest, a.max_attempts)
    finally:
        sh.ssh("sudo -n /usr/local/sbin/pi-power-pin restore", 60)
    if summary["ok"] != summary["cells"]:
        log(f"[matrix] incomplete: {summary['ok']}/{summary['cells']} cells ok")
    raise SystemExit(exit_code(summary))


if __name__ == "__main__":
    main()
