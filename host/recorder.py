#!/usr/bin/env python3
"""Always-on Monsoon HVPM recorder for the Pi 5 power lab.

The HVPM main channel *powers the Pi*, so this process never switches Vout off
(stopping it leaves the output at the last setting). Each USB block is stored with its
host arrival time (PTP-disciplined time.time_ns) and the raw device dropped field (diagnostic only).
HTTP on --port:
  GET /status                     -> recording, serial, vout_V, n_total, dropped, sample_hz, run_dir
  GET /window?t0_ns=&t1_ns=       -> count, mean_P (W), max_P (W), complete

    uv run python recorder.py --vout 5.25 --current-limit 6
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import shutil
import signal
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from clockfit import fit_sample_clock
from store import SampleStore

VOUT_MIN, VOUT_MAX = 4.80, 5.40
CURRENT_LIMIT_MAX = 8.0
MAX_WINDOW_S = 600   # /window span cap: bounds memory per request
log = logging.getLogger("recorder")


def validate_args(vout: float, current_limit: float) -> None:
    if not (VOUT_MIN <= vout <= VOUT_MAX):
        raise SystemExit(f"--vout {vout} outside safe Pi 5 range [{VOUT_MIN}, {VOUT_MAX}]")
    if not (0 < current_limit <= CURRENT_LIMIT_MAX):
        raise SystemExit(f"--current-limit {current_limit} outside (0, {CURRENT_LIMIT_MAX}] A")


# ---- live window stats -------------------------------------------------------
def _clock(store):
    """Clock fit over recent blocks, cached on the store for 1 s (refit as data arrives)."""
    now = time.monotonic()
    cached = getattr(store, "_clock_cache", None)
    if cached is None or now - cached[0] > 1.0 or cached[2] != store.seq:
        last_idx, host_ns = store.recent_blocks()
        if last_idx.size == 0:
            return None
        store._clock_cache = (now, fit_sample_clock(last_idx, host_ns), store.seq)
    return store._clock_cache[1]


def window_stats(store, t0_ns: int, t1_ns: int) -> dict:
    """Stats over samples whose time is in [t0_ns, t1_ns). NaN (dropped) samples are excluded."""
    c = _clock(store)
    if c is None:
        return {"count": 0, "mean_P": None, "max_P": None, "complete": False}
    eps = 0.01  # float64 at ~1.8e18 ns has ~256 ns granularity (~0.001 sample)
    i0 = int(np.ceil(float(c.to_idx(t0_ns)) - eps))
    i1 = int(np.ceil(float(c.to_idx(t1_ns)) - eps))
    complete = bool(c.to_ns(store.n_total - 1) >= t1_ns)
    I, V = store.read(i0, i1)
    P = I.astype(np.float64) * V
    P = P[np.isfinite(P)]
    if P.size == 0:
        return {"count": 0, "mean_P": None, "max_P": None, "complete": complete}
    return {"count": int(P.size), "mean_P": float(P.mean()), "max_P": float(P.max()),
            "complete": complete}


def make_server(store, status_fn, host: str, port: int) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/status":
                    self._send(200, {**status_fn(), "n_total": store.n_total,
                                     "dropped": store.dropped,
                                     "raw_dropped_nonzero_blocks": store.raw_dropped_nonzero_blocks, "run_dir": str(store.dir),
                                     "disk_free_gb": round(
                                         shutil.disk_usage(store.dir).free / 1e9, 1)})
                elif u.path == "/window":
                    q = parse_qs(u.query)
                    t0, t1 = int(q["t0_ns"][0]), int(q["t1_ns"][0])
                    if t1 <= t0 or t1 - t0 > MAX_WINDOW_S * 1e9:
                        self._send(400, {"error": f"window must satisfy 0 < t1-t0 <= "
                                                  f"{MAX_WINDOW_S} s"})
                        return
                    self._send(200, window_stats(store, t0, t1))
                else:
                    self._send(404, {"error": "unknown path"})
            except Exception as e:  # noqa: BLE001 - report, never kill the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return ThreadingHTTPServer((host, port), H)


# ---- device process -----------------------------------------------------------
def device_proc(cfg, q, stop_evt):  # pragma: no cover - needs the HVPM
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from Monsoon import HVPM, pmapi
    from Monsoon import Operations as op
    from Monsoon import sampleEngine as se

    class StreamingEngine(se.SampleEngine):
        def _SampleEngine__vectorProcess(self, measurements):
            se.SampleEngine._SampleEngine__vectorProcess(self, measurements)
            host_ns = time.time_ns()
            cols = []
            for attr in ("mainCurrent", "mainVoltage"):
                name = "_SampleEngine__" + attr
                chunks = getattr(self, name)
                setattr(self, name, [])
                cols.append(np.concatenate([np.asarray(c, dtype=np.float64) for c in chunks])
                            if chunks else np.empty(0))
            self._SampleEngine__timeStamps = []
            n = min(len(cols[0]), len(cols[1]))
            if n:
                q.put(("data", cols[0][:n] / 1000.0, cols[1][:n], host_ns, int(self.dropped)))
            if stop_evt.is_set():
                self._SampleEngine__stopTriggerSet = True

    mon = None
    try:
        protocol = pmapi.USB_protocol()
        serials = protocol.enumerateDevices()
        if not serials:
            raise RuntimeError("No Monsoon on USB (2ab9:0001)")
        serial = cfg["serial"] or serials[0]
        protocol.Connect(op.HardwareModel.HVPM, serial)
        mon = HVPM.Monsoon()
        mon.setup_usb(serial, protocol)
        try:
            mon.stopSampling()
        except Exception:  # noqa: BLE001
            pass
        mon.setRunTimeCurrentLimit(cfg["current_limit"])
        mon.setVout(cfg["vout"])
        time.sleep(0.2)
        mon.fillStatusPacket()
        q.put(("info", {"serial": int(mon.statusPacket.serialNumber), "vout_V": cfg["vout"],
                        "current_limit_A": cfg["current_limit"]}))
        eng = StreamingEngine(mon)
        eng.ConsoleOutput(False)
        eng.enableChannel(se.channels.MainCurrent)
        eng.enableChannel(se.channels.MainVoltage)
        eng.disableChannel(se.channels.USBCurrent)
        eng.disableChannel(se.channels.USBVoltage)
        eng.startSampling(se.triggers.SAMPLECOUNT_INFINITE, 1)
    except Exception as e:  # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        # keep-on: Vout is deliberately NOT set to 0 - the Pi is powered by this channel
        if mon is not None:
            try:
                mon.closeDevice()
            except Exception:  # noqa: BLE001
                pass
        q.put(("stopped", None))


def main():  # pragma: no cover - needs the HVPM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vout", type=float, required=True)
    ap.add_argument("--current-limit", type=float, default=6.0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8051)
    a = ap.parse_args()
    validate_args(a.vout, a.current_limit)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(a.outdir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    store = SampleStore(run_dir)
    info: dict = {"recording": False}
    rate = {"t": time.monotonic(), "n": 0, "hz": 0.0}

    def status():
        return {**info, "sample_hz": rate["hz"]}

    ctx = mp.get_context("spawn")
    q, stop_evt = ctx.Queue(maxsize=10_000), ctx.Event()
    proc = ctx.Process(target=device_proc, daemon=True,
                       args=({"vout": a.vout, "current_limit": a.current_limit,
                              "serial": a.serial}, q, stop_evt))
    proc.start()
    srv = make_server(store, status, a.host, a.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("recording to %s, HTTP :%d", run_dir, a.port)

    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    (run_dir / "recorder_meta.json").write_text(json.dumps(vars(a), indent=2))
    while True:
        if stopping.is_set():
            stop_evt.set()
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            if not proc.is_alive():
                break
            continue
        kind = msg[0]
        if kind == "data":
            _, I_A, V_V, host_ns, dropped = msg
            store.append(I_A, V_V, host_ns, dropped)
            info["recording"] = True
            now = time.monotonic()
            rate["n"] += len(I_A)
            if now - rate["t"] >= 5.0:
                rate["hz"] = rate["n"] / (now - rate["t"])
                rate.update(t=now, n=0)
                log.info("n=%d dropped=%d %.1f Hz", store.n_total, store.dropped, rate["hz"])
        elif kind == "info":
            info.update(msg[1])
            (run_dir / "device.json").write_text(json.dumps(msg[1], indent=2))
            log.info("device %s", msg[1])
        elif kind == "error":
            info["recording"] = False
            log.error("device: %s", msg[1])
        elif kind == "stopped":
            break
    info["recording"] = False
    srv.shutdown()
    store.close()
    log.info("stopped; Vout left ON at %.2f V", a.vout)


if __name__ == "__main__":
    main()
