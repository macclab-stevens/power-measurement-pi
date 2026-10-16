#!/usr/bin/env python3
"""SP3 experiment orchestrator (EliteBook). Resumable: one manifest, one append-only results log.

    uv run python experiment.py plan [--smoke|--probe] [--only yolo11n,lm64] --out ../runs/matrix/manifest.jsonl
    uv run python experiment.py run --manifest ../runs/matrix/manifest.jsonl [--auto-power-cycle 3]
    uv run python experiment.py status --manifest ../runs/matrix/manifest.jsonl

Per cell: recorder recording -> Pi reachable -> Pi below 50 C -> run.py on the Pi (output to a
log file there) with `dmesg -W` streamed here -> rsync the run back -> align.py -> verify_cell.py
-> append one line to results.jsonl. The recorder is never stopped, except by the opt-in
power-cycle recovery for a halted Pi. A failed cell is retried once at the end of its round.
Cells already `ok` in results.jsonl are skipped, so `run` can be restarted at any time.

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
                "--vout", "5.25", "--current-limit", "6"]
THERMAL_MAX_C = 50.0
THERMAL_WAIT_S = 600
RUN_TIMEOUT_S = 40 * 60
PI_DOWN_S = 180
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


def pi_command(cell: dict) -> str:
    cid = cell["cell_id"]
    args = shlex.join([*cell["args"], "--out", f"runs/matrix/{cid}"])
    return (f"cd {PI_DIR} && mkdir -p runs/matrix && rm -rf runs/matrix/{cid} && "
            f"{{ [ ! -f runs/matrix/{cid}.log ] || mv runs/matrix/{cid}.log "
            f"runs/matrix/{cid}.log.$(date +%s); }} && "
            f"{PI_UV} run --frozen python src/run.py {args} > runs/matrix/{cid}.log 2>&1")


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


class RecorderUnavailable(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, runs_dir, results_path, sh=None, status_fn=recorder_status,
                 align_fn=None, verify_fn=None, current_fn=halted_current, sleep=time.sleep,
                 auto_power_cycle: int = 0, log=print):
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
        err = self.thermal_gate()
        if err:
            return fail("thermal", err)

        local.mkdir(parents=True, exist_ok=True)
        dm, dm_f = self.sh.popen([*SSH, PI, "dmesg -W"], local.parent / f"{cid}.dmesg.log")
        try:
            t_run = time.monotonic()
            rc, out = self.sh.ssh(pi_command(cell), RUN_TIMEOUT_S)
            res["run_s"] = round(time.monotonic() - t_run, 1)
        finally:
            dm.terminate()
            dm_f.close()
        res["run_rc"] = rc
        # always fetch what exists (log + run dir) for diagnosis
        self.sh.run(["rsync", "-a", f"{PI}:{PI_DIR}/runs/matrix/{cid}.log",
                     str(local.parent / f"{cid}.log")], 300)
        rrc, rout = self.sh.run(["rsync", "-a", f"{PI}:{PI_DIR}/runs/matrix/{cid}/",
                                 str(local) + "/"], 600)
        if rc != 0:
            if not self.pi_up():
                return fail("pi_crash", f"Pi went down during the run (rc={rc}): {out[-500:]}")
            tail = (local.parent / f"{cid}.log")
            msg = tail.read_text()[-1500:] if tail.exists() else out[-1500:]
            return fail("run", f"run.py rc={rc}: {msg}")
        if rrc != 0:
            return fail("rsync", rout)

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
                        if res["stage"] in ("pi_down", "recorder"):
                            aborted = f"{res['stage']}: {res['error']}"
                            break
                        if attempt < max_attempts:
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--probe", action="store_true", help="every model once at 2400 MHz")
    p.add_argument("--only", default="")
    p.add_argument("--seed", type=int, default=7)
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
        import matrix
        only = [x for x in a.only.split(",") if x]
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
    rc, pi_head = sh.ssh(f"git -C {PI_DIR} rev-parse --short HEAD", 30)
    pi_head = pi_head.strip().splitlines()[-1] if rc == 0 and pi_head.strip() else None
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
