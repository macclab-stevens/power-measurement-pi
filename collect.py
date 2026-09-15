#!/usr/bin/env python3
"""Long-duration data collection for the YOLO power harness.

Repeatedly runs the capture pipeline (src/run.py) for many cycles so you can
collect hours of per-layer / per-op power data unattended.

Usage (from the project dir /home/macc2026/Desktop/pi_power_measurement):
    uv run python collect.py --hours 6
    uv run python collect.py --runs 50 --frames 30 --bench 3
    uv run python collect.py --hours 4 --cooldown 10 --conf 0.3
    uv run python collect.py --hours 8 --frames 40 --bench 5

Stops when the time budget elapses or --runs cycles complete. Every cycle is an
isolated src/run.py subprocess (fresh camera / sampler / model) so long runs
never accumulate memory or drift, and one bad cycle cannot take down the rest.

Writes (under the output dir, default runs/):
    collect_report.csv   - one row per completed cycle (overview)
    collect.log          - full run log + per-cycle status
    <ts>/                - one full dataset per cycle (as a normal run)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_RE = re.compile(
    r"\[summary\] fps=(?P<fps>[0-9.]+) baseline_W=(?P<bw>[0-9.]+) "
    r"hz=(?P<hz>[0-9.]+) layer_rows=(?P<lr>[0-9]+) op_rows=(?P<or_>[0-9]+) "
    r"perop_rows=(?P<pr>[0-9]+) skipped=(?P<sk>[0-9]+) best_conf=(?P<conf>[0-9.]+)"
)
OK_RE = re.compile(r"\[ok\] dataset written to (?P<out>\S+)")


def _fmt(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="stop after this many hours of collection (0 = no time budget)")
    ap.add_argument("--runs", type=int, default=0,
                    help="stop after this many cycles (0 = no count budget)")

    # per-cycle pipeline options (forwarded to src/run.py)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--bench", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--weights", default="yolo11n.pt")

    ap.add_argument("--cooldown", type=float, default=8.0,
                    help="seconds to wait between cycles (let the Pi cool/settle)")
    ap.add_argument("--max-fail", type=int, default=1,
                    help="consecutive failed cycles before collection aborts")
    ap.add_argument("--out", default="runs", help="output base directory")
    args = ap.parse_args()

    if args.hours <= 0 and args.runs <= 0:
        ap.error("give at least one of --hours or --runs")

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(args.out, "collect.log"))],
    )
    log = logging.getLogger("collect")

    deadline = time.time() + args.hours * 3600.0 if args.hours > 0 else None
    report_path = os.path.join(args.out, "collect_report.csv")
    header = ["cycle", "started", "duration_s", "returncode", "outdir",
              "fps", "baseline_W", "hz", "layer_rows", "op_rows",
              "perop_rows", "skipped", "best_conf"]
    report = open(report_path, "w", newline="")
    writer = csv.DictWriter(report, fieldnames=header)
    writer.writeheader()
    report.flush()

    log.info(f"collector start | budget: hours={args.hours} runs={args.runs} | "
             f"cycle: frames={args.frames} bench={args.bench}s warmup={args.warmup} "
             f"conf={args.conf} imgsz={args.imgsz} threads={args.threads} "
             f"weights={args.weights} cooldown={args.cooldown}s out={args.out}")

    cycle = 0
    consecutive_fail = 0
    started_at = time.time()
    try:
        while True:
            if args.runs > 0 and cycle >= args.runs:
                log.info("run-count budget reached; stopping")
                break
            if deadline is not None and time.time() >= deadline:
                log.info("time budget reached; stopping")
                break

            cycle += 1
            t0 = time.time()
            log.info(f"--- cycle {cycle} start {_fmt(t0)} ---")
            cmd = [sys.executable, "src/run.py",
                   "--frames", str(args.frames), "--bench", str(args.bench),
                   "--warmup", str(args.warmup), "--conf", str(args.conf),
                   "--imgsz", str(args.imgsz), "--threads", str(args.threads),
                   "--weights", args.weights]
            try:
                p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=7200)
            except subprocess.TimeoutExpired:
                log.error(f"cycle {cycle}: timed out (2h cap) - treating as failure")
                consecutive_fail += 1
                writer.writerow({"cycle": cycle, "started": _fmt(t0),
                                 "duration_s": time.time() - t0, "returncode": "TIMEOUT",
                                 "outdir": ""})
                report.flush()
            else:
                dur = time.time() - t0
                out = (p.stdout or "") + "\n" + (p.stderr or "")
                m_sum = SUMMARY_RE.search(out)
                m_ok = OK_RE.search(out)
                row = {
                    "cycle": cycle, "started": _fmt(t0), "duration_s": round(dur, 1),
                    "returncode": p.returncode,
                    "outdir": m_ok.group("out") if m_ok else "",
                    "fps": m_sum.group("fps") if m_sum else "",
                    "baseline_W": m_sum.group("bw") if m_sum else "",
                    "hz": m_sum.group("hz") if m_sum else "",
                    "layer_rows": m_sum.group("lr") if m_sum else "",
                    "op_rows": m_sum.group("or_") if m_sum else "",
                    "perop_rows": m_sum.group("pr") if m_sum else "",
                    "skipped": m_sum.group("sk") if m_sum else "",
                    "best_conf": m_sum.group("conf") if m_sum else "",
                }
                if p.returncode == 0:
                    consecutive_fail = 0
                    log.info(f"cycle {cycle} OK ({dur:.0f}s): fps={row['fps']} "
                             f"baseline={row['baseline_W']}W hz={row['hz']} "
                             f"perop={row['perop_rows']} skipped={row['skipped']} "
                             f"conf={row['best_conf']} -> {row['outdir']}")
                else:
                    consecutive_fail += 1
                    log.error(f"cycle {cycle} FAILED rc={p.returncode} ({dur:.0f}s)")
                    tail = "\n".join(out.strip().splitlines()[-5:])
                    log.error(f"  last output:\n{tail}")
                writer.writerow(row)
                report.flush()

            if consecutive_fail >= args.max_fail and args.runs and \
                    cycle < args.runs + args.max_fail:
                log.error(f"{consecutive_fail} consecutive failures; aborting")
                break

            # stop before cooldown if budgets are already met
            if args.runs > 0 and cycle >= args.runs:
                continue
            if deadline is not None and time.time() >= deadline:
                continue
            if args.cooldown > 0:
                log.info(f"cooldown {args.cooldown}s ...")
                time.sleep(args.cooldown)

    finally:
        report.close()

    elapsed_h = (time.time() - started_at) / 3600.0
    log.info(f"collector done: {cycle} cycles over {elapsed_h:.2f}h "
             f"-> datasets in {args.out}/, report {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())