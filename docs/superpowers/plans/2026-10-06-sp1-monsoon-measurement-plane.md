# SP1: Monsoon Measurement Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Pi's PMIC fast-loop power sampling with an always-on Monsoon HVPM recorder on the EliteBook, plus a nanosecond event log on the Pi. Then align the two to well under 1 ms, so that every 5 kHz power sample is labeled with the phase, mode, iteration, and layer running at that moment.

**Architecture:**
- **Pi side:**
  - Every phase marker and every layer begin/end is written to an in-memory `EventLog`, timestamped with `time.time_ns()`. The lab's PTP/GPS keeps that clock within about 0.03 ms of GPS.
  - The Pi emits 4-core busy/idle sync-pulse trains at the start and end of each run.
  - `MonsoonSampler` keeps the existing `PowerSampler` contract, so `run.py` and `microbench.py` barely change. Its `window()` asks the recorder over HTTP.
- **EliteBook side:**
  - `host/recorder.py` owns the HVPM. It always runs with keep-on, because it powers the Pi.
  - It appends float32 I/V to `samples.f32` and logs a block table: the host arrival `time_ns` and the cumulative dropped-sample count for every USB block.
  - It serves `/status` and `/window` on port 8051.
- **Offline:** `host/align.py` does three things:
  - Fits a coarse sample clock from the block table (lower envelope).
  - Refines it with the sync-pulse edges, using a linear offset + drift fit.
  - Labels every sample and writes `alignment.json` and `power_trace.parquet`.

**Tech Stack:**
- **Pi:** Python 3.13, torch 2.14 (existing locked env).
- **Host:** Python 3.12, `monsoon==0.1.88`, pyusb, numpy, pandas, pyarrow.
- **Both:** stdlib `http.server`/`urllib`; pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-monsoon-experiment-design.md`. Revision 1 at the end of the spec overrides the earlier sections.

## Global Constraints

- The Monsoon output is never switched off by our code. The recorder always behaves as keep-on. Changing Vout means: shut down the Pi cleanly, restart the recorder, and the Pi reboots.
- Vout is accepted only within **4.80–5.40 V**. Anything else is refused at startup.
- Pi timestamps use `time.time_ns()` as primary and `time.monotonic_ns()` as secondary. Columns are `t_real_ns`, `t_mono_ns`.
- Recorder sample rate is a nominal 5000 Hz. Stored per sample: `I_A`, `V_V` as float32. `P_W = I_A * V_V`.
- Units contract is unchanged: `_W/_ms/_mJ/_ns` suffixes; `E_mJ = P_W × t_ms`.
- Alignment PASS requires `max_residual_ms < 1.0`. Drift is reported in ppm.
- `run.py --sampler pmic` (the default) must behave exactly as before, with `schema_version` 3. `--sampler monsoon` writes `schema_version: 4`.
- The pulse pattern is ON 150 / OFF 100 / ON 300 / OFF 100 / ON 50 ms, with ≥ 500 ms idle guard before and after.
- Passwords never appear in code, docs, or commits.
- `~/monsoon/monsoon_monitor.py` on the EliteBook is left unmodified. Our recorder is a separate file in this repo.

## Review Focus

1. **Recorder down or unreachable when `run.py --sampler monsoon` starts.** It must fail immediately with a clear message, not after 5 minutes of work. Test: `test_monsoon_sampler_start_fails_fast` (Task 3).
2. **Dropped USB samples in the middle of a run.** NaN gaps must be inserted so index-based time does not slip, and window means must ignore NaN. Tests: `test_dropped_samples_insert_nan_gap` and `test_window_ignores_nan` (Task 5).
3. **Window requested for "now", before the USB data has arrived.** The sampler must wait for `complete: true` and must not return a truncated mean. Test: `test_window_waits_until_complete` (Task 3).
4. **A sync train is missing or washed out** (e.g. the Pi was already busy). `align.py` must raise `AlignError("sync pulses not found …")` and must not write labels from a bad fit. Test: `test_align_raises_when_no_pulses` (Task 6).
5. **Nested layers** (a container module around a Conv). Each sample must take the innermost layer, and samples between layers must keep the phase label with an empty `layer_path`. Test: `test_align_labels_innermost_layer` (Task 6).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/eventlog.py` (new) | `EventLog`: thread-safe append of timestamped events; CSV writer. Torch-free. |
| `src/syncpulse.py` (new) | `emit(events, tag)`: 4-core busy/idle pulse train with logged edges. Torch-free. |
| `src/powersampler.py` (modify) | Add `MonsoonSampler` (HTTP `window`, event log, context-only thread) and `make_sampler()`. |
| `src/layers.py` (modify) | Optional `events` sink: begin/end per hooked module call; mode/iter labels. |
| `src/run.py` (modify) | `--sampler/--recorder-url`; sync pulses; `events.csv`; schema v4 meta. |
| `src/sync_check.py` (new) | N pulse trains only, for alignment validation. |
| `host/pyproject.toml` (new) | Separate uv project for the EliteBook (no torch). |
| `host/store.py` (new) | `SampleStore`: append samples + block table, NaN gap insertion, `read(i0, i1)`. |
| `host/clockfit.py` (new) | `SampleClock` + `fit_sample_clock()`: lower-envelope idx → host-ns fit. |
| `host/recorder.py` (new) | HVPM device process (adapted `StreamingEngine` hook) + HTTP `/status` `/window`. |
| `host/align.py` (new) | Pulse edge detection, offset + drift fit, sample labeling, outputs. |
| `tests/test_eventlog.py`, `tests/test_syncpulse.py`, `tests/test_monsoon_sampler.py` (new) | Torch-free Pi-side tests (run on the NUC with system `python3 -m pytest`). |
| `tests/test_layer_events.py` (new) | Torch test (runs on the Pi). |
| `host/tests/test_store.py`, `test_clockfit.py`, `test_recorder_http.py`, `test_align.py` (new) | Host tests (`cd host && uv run pytest`). |

**Where tests run:**
- Torch-free `tests/*`: on the NUC, with `python3 -m pytest tests/test_eventlog.py tests/test_syncpulse.py tests/test_monsoon_sampler.py`. The NUC has pytest 7.4 and numpy, and no torch.
- `host/tests`: on the EliteBook, with `cd host && ~/.local/bin/uv run pytest -q`.
- Full Pi suite: on the Pi, with `~/.local/bin/uv run pytest tests/ -q`.

---

### Task 1: EventLog

**Files:**
- Create: `src/eventlog.py`
- Test: `tests/test_eventlog.py`

**Interfaces:**
- Produces:
  - `EventLog()` with:
    - `.marker(name: str) -> None`
    - `.begin(name: str, layer_path: str = "") -> None`
    - `.end(name: str, layer_path: str = "") -> None`
    - `.edge(kind: str, name: str) -> None` (kind ∈ `pulse_on|pulse_off`)
    - `.labels: dict` (keys `mode`, `iter`)
    - `.window_id: int`, `.phase: str`
    - `.rows() -> list[dict]`
    - `.write_csv(path) -> None`
  - Module constant `FIELDS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eventlog.py
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eventlog import EventLog, FIELDS


def test_rows_have_wall_and_mono_ns_and_are_ordered():
    ev = EventLog()
    before = time.time_ns()
    ev.marker("warmup")
    ev.marker("forward")
    after = time.time_ns()
    rows = ev.rows()
    assert [r["kind"] for r in rows] == ["marker", "marker"]
    assert before <= rows[0]["t_real_ns"] <= rows[1]["t_real_ns"] <= after
    assert rows[0]["t_mono_ns"] <= rows[1]["t_mono_ns"]


def test_marker_bumps_window_id_only_on_new_phase():
    ev = EventLog()
    ev.marker("forward")
    ev.marker("forward")
    ev.marker("gap")
    assert [r["window_id"] for r in ev.rows()] == [1, 1, 2]
    assert ev.phase == "gap"


def test_begin_end_track_nesting_depth():
    ev = EventLog()
    ev.begin("layer", layer_path="model.0")
    ev.begin("layer", layer_path="model.0.conv")
    ev.end("layer", layer_path="model.0.conv")
    ev.end("layer", layer_path="model.0")
    assert [(r["kind"], r["depth"]) for r in ev.rows()] == [
        ("begin", 1), ("begin", 2), ("end", 2), ("end", 1)]


def test_labels_and_bench_config_id_are_recorded():
    ev = EventLog()
    ev.labels["mode"] = "decode"
    ev.labels["iter"] = 3
    ev.marker("bench:abc123")
    r = ev.rows()[0]
    assert (r["mode"], r["iter"], r["config_id"]) == ("decode", 3, "abc123")


def test_write_csv_roundtrip(tmp_path):
    ev = EventLog()
    ev.marker("warmup")
    ev.edge("pulse_on", "sync_start")
    p = tmp_path / "events.csv"
    ev.write_csv(p)
    rows = list(csv.DictReader(open(p)))
    assert list(rows[0].keys()) == FIELDS
    assert rows[1]["kind"] == "pulse_on" and rows[1]["name"] == "sync_start"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_eventlog.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'eventlog'`

- [ ] **Step 3: Write the implementation**

```python
# src/eventlog.py
"""Pi-side event log: what the workload is doing, with nanosecond timestamps.

Primary clock is time.time_ns() (lab PTP/GPS-disciplined wall clock, ~0.03 ms
from the TM2500C), so events line up with the EliteBook's Monsoon recording.
monotonic_ns is kept as a secondary column. Appends are lock-guarded list
appends, cheap enough for per-layer begin/end events.
"""
from __future__ import annotations

import csv
import threading
import time

FIELDS = ["t_real_ns", "t_mono_ns", "kind", "name", "depth", "mode", "iter",
          "layer_path", "config_id", "window_id"]


class EventLog:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._depth = 0
        self.window_id = 0
        self.phase = "idle"
        self.labels: dict = {"mode": "", "iter": ""}

    def _add(self, kind: str, name: str, depth: int, layer_path: str = "",
             config_id: str = "") -> None:
        row = {"t_real_ns": time.time_ns(), "t_mono_ns": time.monotonic_ns(),
               "kind": kind, "name": name, "depth": depth,
               "mode": self.labels["mode"], "iter": self.labels["iter"],
               "layer_path": layer_path, "config_id": config_id,
               "window_id": self.window_id}
        with self._lock:
            self._rows.append(row)

    def marker(self, name: str) -> None:
        """Start a new top-level phase. A changed name bumps window_id (PowerSampler semantics)."""
        if name and name != self.phase:
            self.window_id += 1
        self.phase = name or "idle"
        cid = self.phase.split(":", 1)[1] if self.phase.startswith("bench:") else ""
        self._add("marker", self.phase, 0, config_id=cid)

    def begin(self, name: str, layer_path: str = "") -> None:
        self._depth += 1
        self._add("begin", name, self._depth, layer_path=layer_path)

    def end(self, name: str, layer_path: str = "") -> None:
        self._add("end", name, self._depth, layer_path=layer_path)
        self._depth = max(0, self._depth - 1)

    def edge(self, kind: str, name: str) -> None:
        self._add(kind, name, 0)

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)

    def write_csv(self, path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_eventlog.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eventlog.py tests/test_eventlog.py
git commit -m "feat: EventLog with ns wall/mono timestamps and nesting depth"
```

---

### Task 2: Sync-pulse emitter

**Files:**
- Create: `src/syncpulse.py`
- Test: `tests/test_syncpulse.py`

**Interfaces:**
- Consumes: `EventLog.edge(kind, name)` and `EventLog.marker(name)` from Task 1.
- Produces:
  - `PATTERN_MS: tuple[tuple[int, int], ...]`
  - `GUARD_MS: int`
  - `emit(events: EventLog, tag: str, pattern=PATTERN_MS, guard_ms=GUARD_MS, ncores=None) -> None`

  `emit` sets marker `sync:<tag>`. It writes `pulse_on`/`pulse_off` edges named `<tag>`, alternating and starting with `pulse_on`, and ends with a final `pulse_off`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_syncpulse.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eventlog import EventLog
from syncpulse import PATTERN_MS, emit


def test_default_pattern_is_the_spec_pattern():
    assert PATTERN_MS == ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))


def test_emit_logs_alternating_edges_with_pattern_spacing():
    ev = EventLog()
    pattern = ((1, 40), (0, 30), (1, 60))
    emit(ev, "t0", pattern=pattern, guard_ms=20, ncores=2)
    rows = ev.rows()
    assert rows[0]["kind"] == "marker" and rows[0]["name"] == "sync:t0"
    edges = [r for r in rows if r["kind"].startswith("pulse_")]
    assert [e["kind"] for e in edges] == ["pulse_on", "pulse_off", "pulse_on", "pulse_off"]
    assert all(e["name"] == "t0" for e in edges)
    gaps_ms = [(b["t_real_ns"] - a["t_real_ns"]) / 1e6 for a, b in zip(edges, edges[1:])]
    for got, want in zip(gaps_ms, (40, 30, 60)):
        assert abs(got - want) < 15, (gaps_ms, want)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_syncpulse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncpulse'`

- [ ] **Step 3: Write the implementation**

```python
# src/syncpulse.py
"""Power-signature sync pulses for aligning Pi events with the Monsoon trace.

All cores busy-spin during ON slots and sleep during OFF slots. That gives a
~2-3 W step on the Pi's 5 V input that the Monsoon sees within a sample or two.
Each edge's Pi wall-clock time is logged right before the workers are
released/parked. Distinct slot widths make the train unambiguous in
cross-correlation.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

PATTERN_MS = ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))
GUARD_MS = 500


def _spin(go, quit_):  # pragma: no cover - runs in child processes
    x = 0
    while not quit_.is_set():
        if go.is_set():
            x += 1
        else:
            go.wait(0.05)


def emit(events, tag: str, pattern=PATTERN_MS, guard_ms: int = GUARD_MS,
         ncores: int | None = None) -> None:
    ctx = mp.get_context("fork")
    go, quit_ = ctx.Event(), ctx.Event()
    workers = [ctx.Process(target=_spin, args=(go, quit_), daemon=True)
               for _ in range(ncores or os.cpu_count() or 4)]
    for w in workers:
        w.start()
    try:
        events.marker(f"sync:{tag}")
        time.sleep(guard_ms / 1000.0)
        for on, ms in pattern:
            if on:
                events.edge("pulse_on", tag)
                go.set()
            else:
                events.edge("pulse_off", tag)
                go.clear()
            time.sleep(ms / 1000.0)
        events.edge("pulse_off", tag)
        go.clear()
        time.sleep(guard_ms / 1000.0)
    finally:
        quit_.set()
        go.set()  # wake parked workers so they see quit_
        for w in workers:
            w.join(timeout=2)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_syncpulse.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/syncpulse.py tests/test_syncpulse.py
git commit -m "feat: 4-core sync-pulse emitter with logged edges"
```

---

### Task 3: MonsoonSampler + make_sampler

**Files:**
- Modify: `src/powersampler.py` (append new class and factory; `PowerSampler` is unchanged)
- Test: `tests/test_monsoon_sampler.py`

**Interfaces:**
- Consumes: `EventLog` (Task 1). Recorder HTTP API (built in Task 5; faked here):
  - `GET /status` → `{"recording": bool, "sample_hz": float, "serial": int, "vout_V": float, "dropped": int, "n_total": int}`
  - `GET /window?t0_ns=&t1_ns=` → `{"count": int, "mean_P": float|null, "max_P": float|null, "complete": bool}`
- Produces:
  - `MonsoonSampler(recorder_url: str, events: EventLog | None = None, timeout_s: float = 5.0)` with the `PowerSampler` contract:
    - `start()`, `stop()`, `marker()`, `now_ms()`, `window()`, `snapshot_env()`, `rows()` (returns `[]`), `context()`, `achieved_hz()`
    - plus `.events` and `recorder_status() -> dict`
  - `make_sampler(kind: str, recorder_url: str | None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monsoon_sampler.py
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from powersampler import MonsoonSampler, PowerSampler, make_sampler


class Fake:
    recording = True
    incomplete_calls = 0
    last_query = None


def _server(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/status":
                body = {"recording": fake.recording, "sample_hz": 4999.0, "serial": 36517,
                        "vout_V": 5.2, "dropped": 0, "n_total": 10}
            else:
                fake.last_query = {k: int(v[0]) for k, v in parse_qs(u.query).items()}
                done = fake.incomplete_calls <= 0
                fake.incomplete_calls -= 1
                body = {"count": 500, "mean_P": 3.25, "max_P": 4.0, "complete": done}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def fake_recorder():
    fake = Fake()
    srv, url = _server(fake)
    yield fake, url
    srv.shutdown()


def _no_ctx(s):  # don't call vcgencmd on the NUC
    s._ctx_thread = threading.Thread(target=lambda: None)


def test_make_sampler_kinds():
    assert type(make_sampler("pmic", None)) is PowerSampler
    assert isinstance(make_sampler("monsoon", "http://x:1"), MonsoonSampler)
    with pytest.raises(ValueError):
        make_sampler("monsoon", None)


def test_monsoon_sampler_start_fails_fast():
    s = MonsoonSampler("http://127.0.0.1:9", timeout_s=0.5)
    _no_ctx(s)
    with pytest.raises(RuntimeError, match="recorder"):
        s.start()


def test_window_converts_ms_to_wall_ns_and_returns_stats(fake_recorder):
    fake, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    w = s.window(100.0, 200.0)
    assert w == {"count": 500, "mean_P": 3.25, "max_P": 4.0}
    q = fake.last_query
    assert q["t1_ns"] - q["t0_ns"] == 100_000_000
    assert q["t0_ns"] == s._t0_ns + 100_000_000
    assert s.achieved_hz() == 4999.0
    assert s.rows() == []
    s.stop()


def test_window_waits_until_complete(fake_recorder):
    fake, url = fake_recorder
    fake.incomplete_calls = 3
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    assert s.window(0.0, 10.0)["count"] == 500
    assert fake.incomplete_calls < 0
    s.stop()


def test_marker_goes_to_event_log_with_window_ids(fake_recorder):
    _, url = fake_recorder
    s = MonsoonSampler(url)
    _no_ctx(s)
    s.start()
    s.marker("forward")
    s.marker("gap")
    rows = s.events.rows()
    assert [r["name"] for r in rows] == ["forward", "gap"]
    assert s._window_id == 2 and s._phase == "gap"
    s.stop()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_monsoon_sampler.py -q`
Expected: FAIL with `ImportError: cannot import name 'MonsoonSampler'`

- [ ] **Step 3: Write the implementation** (append to `src/powersampler.py`; add `import json`, `import urllib.request` and `from eventlog import EventLog` to the imports at the top)

```python
class MonsoonSampler(PowerSampler):
    """Monsoon HVPM backend (EliteBook recorder). Same public contract as PowerSampler.

    No PMIC fast loop (it burned CPU and polluted the trace); the 5 s context
    thread stays. Markers go to an EventLog on the lab-synced wall clock.
    window() asks the recorder for stats over the same wall-clock range.
    """

    def __init__(self, recorder_url: str, events: EventLog | None = None,
                 timeout_s: float = 5.0) -> None:
        super().__init__()
        self.url = recorder_url.rstrip("/")
        self.events = events or EventLog()
        self._timeout = timeout_s
        self._t0_ns = time.time_ns()
        self._status: dict = {}

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=self._timeout) as r:
            return json.loads(r.read())

    def recorder_status(self) -> dict:
        self._status = self._get("/status")
        return self._status

    def _now(self) -> float:
        return (time.time_ns() - self._t0_ns) / 1e6

    def start(self) -> None:
        try:
            st = self.recorder_status()
        except OSError as e:
            raise RuntimeError(f"Monsoon recorder unreachable at {self.url}: {e}") from e
        if not st.get("recording"):
            raise RuntimeError(f"Monsoon recorder at {self.url} is not recording: {st}")
        self._t0 = time.monotonic()
        self._t0_ns = time.time_ns()
        self._stop.clear()
        self._ctx_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ctx_thread.is_alive():
            self._ctx_thread.join(timeout=10)

    def marker(self, name: str | None) -> None:
        self.events.marker(name or "idle")
        with self._lock:
            self._phase = self.events.phase
            self._window_id = self.events.window_id

    def window(self, t0_ms: float, t1_ms: float) -> dict:
        t0 = self._t0_ns + int(t0_ms * 1e6)
        t1 = self._t0_ns + int(t1_ms * 1e6)
        deadline = time.monotonic() + max(2.0, self._timeout)
        while True:
            w = self._get(f"/window?t0_ns={t0}&t1_ns={t1}")
            if w.get("complete") or time.monotonic() > deadline:
                return {"count": w["count"], "mean_P": w["mean_P"], "max_P": w["max_P"]}
            time.sleep(0.05)

    def rows(self) -> list[dict]:
        return []

    def achieved_hz(self) -> float:
        try:
            return float(self.recorder_status().get("sample_hz", 0.0))
        except OSError:
            return 0.0


def make_sampler(kind: str, recorder_url: str | None):
    if kind == "pmic":
        return PowerSampler()
    if kind == "monsoon":
        if not recorder_url:
            raise ValueError("--sampler monsoon needs --recorder-url")
        return MonsoonSampler(recorder_url)
    raise ValueError(f"unknown sampler '{kind}' (pmic | monsoon)")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_monsoon_sampler.py tests/test_eventlog.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/powersampler.py tests/test_monsoon_sampler.py
git commit -m "feat: MonsoonSampler (HTTP window, event log) + make_sampler"
```

---

### Task 4: Layer events + run.py wiring + sync_check

**Files:**
- Modify: `src/layers.py` (`LayerProfiler.__init__`, `set_frame`, `set_mode`, `_make_pre`, `_make_post`)
- Modify: `src/run.py` (args, sampler construction, sync pulses, outputs, meta)
- Create: `src/sync_check.py`
- Test: `tests/test_layer_events.py` (torch; run on the Pi)

**Interfaces:**
- Consumes: `EventLog` (Task 1), `syncpulse.emit` (Task 2), `make_sampler`/`MonsoonSampler` (Task 3).
- Produces:
  - `LayerProfiler.events: EventLog | None`. When set:
    - each hooked call writes `begin("layer", layer_path=path)` at the same instant as its `t0`, and `end("layer", layer_path=path)` right after `dt`;
    - `set_mode`/`set_frame` update `events.labels["mode"/"iter"]`.
  - Monsoon run directory: `events.csv` (FIELDS from Task 1) and `run_meta.json` with:
    - `"schema_version": 4`, `"sampler": "monsoon"`
    - `"recorder_url"`, `"recorder_status"`
    - `"t0_real_ns"`, `"sync_tags": ["start", "end"]`
  - `src/sync_check.py --trains N --spacing S --out DIR` writes `DIR/events.csv` containing N trains tagged `k0..k{N-1}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layer_events.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn

from eventlog import EventLog
from layers import LayerProfiler


def test_layer_begin_end_nest_and_carry_labels():
    model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()))
    prof = LayerProfiler(model)
    prof.events = EventLog()
    prof.set_mode("forward")
    prof.set_frame(2)
    with torch.no_grad():
        model(torch.zeros(1, 3, 8, 8))
    rows = [r for r in prof.events.rows() if r["kind"] in ("begin", "end")]
    seq = [(r["kind"], r["layer_path"], r["depth"]) for r in rows]
    assert seq == [("begin", "0", 1), ("begin", "0.0", 2), ("end", "0.0", 2),
                   ("begin", "0.1", 2), ("end", "0.1", 2), ("end", "0", 1)]
    assert {(r["mode"], r["iter"]) for r in rows} == {("forward", 2)}


def test_no_events_when_sink_absent():
    model = nn.Sequential(nn.ReLU())
    prof = LayerProfiler(model)
    with torch.no_grad():
        model(torch.zeros(2))
    assert prof.events is None
```

- [ ] **Step 2: Run the test to verify it fails** (on the Pi, after the Task 7 deploy step; or on any machine with torch)

Run: `~/.local/bin/uv run pytest tests/test_layer_events.py -q`
Expected: FAIL with `AttributeError: 'LayerProfiler' object has no attribute 'events'`

- [ ] **Step 3: Implement the layers.py changes**

In `LayerProfiler.__init__`, before `self._register()`, add:

```python
        self.events = None                          # optional EventLog sink (Monsoon backend)
```

Replace `set_frame`/`set_mode`:

```python
    def set_frame(self, frame: int) -> None:
        self.frame = frame
        if self.events is not None:
            self.events.labels["iter"] = frame

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if self.events is not None:
            self.events.labels["mode"] = mode
```

In `_make_pre`, replace the last line `self._stack[path].append((time.perf_counter(), idx))` with:

```python
            if self.events is not None:
                self.events.begin("layer", layer_path=path)
            self._stack[path].append((time.perf_counter(), idx))
```

In `_make_post`, right after `dt = (time.perf_counter() - t0) * 1000.0`, insert:

```python
            if self.events is not None:
                self.events.end("layer", layer_path=path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `~/.local/bin/uv run pytest tests/test_layer_events.py tests/test_identity.py tests/test_features.py -q`
Expected: all pass. `test_identity` and `test_features` must still pass, which confirms there is no behavior change without a sink.

- [ ] **Step 5: Implement the run.py changes**

Imports: replace `from powersampler import PowerSampler` with:

```python
from powersampler import make_sampler
from syncpulse import emit as sync_pulse
```

Arguments: add after `--pin-freq`:

```python
    ap.add_argument("--sampler", default="pmic", choices=["pmic", "monsoon"],
                    help="power source: on-board PMIC (legacy) or EliteBook Monsoon recorder")
    ap.add_argument("--recorder-url", default="http://192.168.0.11:8051",
                    help="Monsoon recorder HTTP base URL (used with --sampler monsoon)")
```

Sampler construction: replace `sampler = PowerSampler()` / `sampler.start()` with:

```python
    sampler = make_sampler(args.sampler, args.recorder_url)
    monsoon = args.sampler == "monsoon"
    if monsoon:
        prof.events = sampler.events
    sampler.start()
```

After the warm-up block, before `sampler.marker("baseline_A")`, add:

```python
        if monsoon:
            sync_pulse(sampler.events, "start")
```

After `perop = bench_configs(...)` and its `_write_csv(... perop_power.csv ...)`, add:

```python
        if monsoon:
            sync_pulse(sampler.events, "end")
            sampler.marker("done")
            sampler.events.write_csv(os.path.join(outdir, "events.csv"))
```

Replace the `samples.csv` write with:

```python
        srows = sampler.rows()
        if not monsoon:
            _write_csv(os.path.join(outdir, "samples.csv"),
                       ["t_ms", "phase", "window_id", "I_A", "V_V", "P_W"], srows)
```

In `meta`, replace `"schema_version": 3,` with:

```python
            "schema_version": 4 if monsoon else 3,
            "sampler": args.sampler,
            "recorder_url": args.recorder_url if monsoon else None,
            "recorder_status": sampler.recorder_status() if monsoon else None,
            "t0_real_ns": getattr(sampler, "_t0_ns", None),
            "sync_tags": ["start", "end"] if monsoon else [],
```

In the `methodology` string, append ` | monsoon: 5 V input power @5 kHz, events on lab PTP wall clock, aligned offline by sync pulses (host/align.py)` when `monsoon` is set. Use a conditional expression.

In acceptance, change the baseline range so it is backend-aware. The board idle on the Monsoon is about 3.2 W, which is above the PMIC core-rail bounds. Replace `BASELINE_W_MIN, BASELINE_W_MAX = 0.5, 8.0` usage with:

```python
    bmin, bmax = (1.5, 8.0) if args.sampler == "monsoon" else (BASELINE_W_MIN, BASELINE_W_MAX)
```

and use `bmin`/`bmax` in both baseline checks.

- [ ] **Step 6: Create `src/sync_check.py`**

```python
"""Emit N sync-pulse trains and write events.csv, for validating alignment.

    uv run python src/sync_check.py --trains 20 --spacing 2 --out runs/sync_check
"""
from __future__ import annotations

import argparse
import os
import time

from eventlog import EventLog
from syncpulse import emit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains", type=int, default=20)
    ap.add_argument("--spacing", type=float, default=2.0, help="idle seconds between trains")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = EventLog()
    ev.marker("idle")
    for k in range(a.trains):
        emit(ev, f"k{k}")
        ev.marker("idle")
        time.sleep(a.spacing)
    ev.marker("done")
    ev.write_csv(os.path.join(a.out, "events.csv"))
    print(f"[ok] {a.trains} trains -> {a.out}/events.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Check that run.py still parses with the old sampler path**

Run (on the Pi): `~/.local/bin/uv run python src/run.py --help | grep -E "sampler|recorder-url"`
Expected: both flags listed.

- [ ] **Step 8: Commit**

```bash
git add src/layers.py src/run.py src/sync_check.py tests/test_layer_events.py
git commit -m "feat: layer begin/end events, --sampler monsoon, sync pulses, events.csv (schema v4)"
```

---

### Task 5: Host recorder (store, clock fit, HVPM process, HTTP)

**Files:**
- Create: `host/pyproject.toml`, `host/store.py`, `host/clockfit.py`, `host/recorder.py`
- Test: `host/tests/test_store.py`, `host/tests/test_clockfit.py`, `host/tests/test_recorder_http.py`

**Interfaces:**
- Produces:
  - `SampleStore(run_dir: Path)` with:
    - `.append(I_A: ndarray, V_V: ndarray, host_ns: int, dropped_cum: int) -> None`
    - `.read(i0: int, i1: int) -> tuple[ndarray, ndarray]`
    - `.n_total: int`, `.dropped: int`
    - `.recent_blocks() -> tuple[ndarray last_idx, ndarray host_ns]`
  - Files: `samples.f32` (interleaved float32 I_A, V_V) and `blocks.csv` with header `block_seq,first_idx,n,host_ns,dropped_cum,gap_inserted`.
  - `load_blocks(run_dir) -> dict[str, ndarray]`
  - `open_samples(run_dir) -> ndarray (N, 2) memmap`
  - `SampleClock(a_ns: float, b_ns: float)` with `.to_ns(idx)` and `.to_idx(t_ns)`.
  - `fit_sample_clock(last_idx, host_ns, rate_hz=5000, segments=20) -> SampleClock`
  - `make_server(store, status_fn, host, port) -> ThreadingHTTPServer`
  - `window_stats(store, t0_ns, t1_ns) -> dict` (keys `count, mean_P, max_P, complete`)
  - Recorder CLI: `uv run python recorder.py --vout V [--current-limit A] [--outdir DIR] [--port 8051]`

- [ ] **Step 1: Create the host project**

```toml
# host/pyproject.toml
[project]
name = "pi-power-host"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["monsoon==0.1.88", "pyusb", "numpy", "pandas", "pyarrow"]

[dependency-groups]
dev = ["pytest"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing store and clock-fit tests**

```python
# host/tests/test_store.py
import numpy as np

from store import SampleStore, load_blocks, open_samples


def test_append_and_read_roundtrip(tmp_path):
    s = SampleStore(tmp_path)
    s.append(np.array([0.5, 0.6]), np.array([5.0, 5.1]), host_ns=1000, dropped_cum=0)
    s.append(np.array([0.7]), np.array([5.2]), host_ns=2000, dropped_cum=0)
    i, v = s.read(0, 3)
    assert np.allclose(i, [0.5, 0.6, 0.7]) and np.allclose(v, [5.0, 5.1, 5.2])
    assert s.n_total == 3
    b = load_blocks(tmp_path)
    assert list(b["first_idx"]) == [0, 2] and list(b["host_ns"]) == [1000, 2000]
    assert open_samples(tmp_path).shape == (3, 2)


def test_dropped_samples_insert_nan_gap(tmp_path):
    s = SampleStore(tmp_path)
    s.append(np.ones(2), np.ones(2), host_ns=1, dropped_cum=0)
    s.append(np.ones(2), np.ones(2), host_ns=2, dropped_cum=3)
    i, _ = s.read(0, s.n_total)
    assert s.n_total == 7 and s.dropped == 3
    assert np.isnan(i[2:5]).all() and np.isfinite(i[[0, 1, 5, 6]]).all()
    b = load_blocks(tmp_path)
    assert list(b["first_idx"]) == [0, 5] and list(b["gap_inserted"]) == [0, 3]


def test_reopen_appends_not_truncates(tmp_path):
    SampleStore(tmp_path).append(np.ones(2), np.ones(2), host_ns=1, dropped_cum=0)
    s2 = SampleStore(tmp_path)
    assert s2.n_total == 2
    s2.append(np.ones(1), np.ones(1), host_ns=2, dropped_cum=0)
    assert open_samples(tmp_path).shape == (3, 2)
```

```python
# host/tests/test_clockfit.py
import numpy as np

from clockfit import SampleClock, fit_sample_clock


def _synthetic(seed=0, ppm=40.0, secs=300, block=50, lat_min_ms=2.0):
    rng = np.random.default_rng(seed)
    b = 200_000.0 * (1 + ppm * 1e-6)
    a = 1.7e18
    last_idx = np.arange(block - 1, int(secs * 5000), block)
    true_ns = a + b * last_idx
    host_ns = true_ns + lat_min_ms * 1e6 + rng.exponential(3e6, size=last_idx.size)
    return last_idx, host_ns, a + lat_min_ms * 1e6, b


def test_fit_recovers_slope_and_lower_envelope():
    last_idx, host_ns, a_env, b = _synthetic()
    c = fit_sample_clock(last_idx, host_ns)
    assert abs(c.b_ns / b - 1) < 3e-6
    probe = np.array([0, 500_000, 1_400_000])
    err_ms = np.abs(c.to_ns(probe) - (a_env + b * probe)) / 1e6
    assert err_ms.max() < 0.5, err_ms


def test_to_idx_inverts_to_ns():
    c = SampleClock(a_ns=1e18, b_ns=200_000.0)
    assert np.allclose(c.to_idx(c.to_ns(np.array([0, 12345]))), [0, 12345])


def test_short_series_falls_back_to_nominal_rate():
    c = fit_sample_clock(np.array([49]), np.array([1e18]))
    assert c.b_ns == 200_000.0 and c.to_ns(49) == 1e18
```

- [ ] **Step 3: Run the tests to verify they fail**

Run (EliteBook): `cd ~/power-measurement-pi/host && ~/.local/bin/uv run pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'store'`

- [ ] **Step 4: Write `host/store.py` and `host/clockfit.py`**

```python
# host/store.py
"""Append-only on-disk store for the always-on Monsoon recording.

samples.f32 : interleaved float32 (I_A, V_V), one row per 5 kHz sample, global index = row.
blocks.csv  : one row per USB block: host arrival time_ns + cumulative device-dropped count.
Dropped samples are filled with NaN rows *before* the block that reports them, so
index-based time never slips.
"""
from __future__ import annotations

import collections
import csv
import os
import threading
from pathlib import Path

import numpy as np

HEADER = ["block_seq", "first_idx", "n", "host_ns", "dropped_cum", "gap_inserted"]
RECENT_BLOCKS = 20_000   # ~10 min of blocks kept in RAM for live clock fits


class SampleStore:
    def __init__(self, run_dir) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._samples_path = self.dir / "samples.f32"
        self._blocks_path = self.dir / "blocks.csv"
        new = not self._blocks_path.exists()
        self._f = open(self._samples_path, "ab")
        self._b = open(self._blocks_path, "a", buffering=1, newline="")
        self._w = csv.writer(self._b)
        if new:
            self._w.writerow(HEADER)
        self.lock = threading.Lock()
        self.n_total = os.path.getsize(self._samples_path) // 8
        self.dropped = 0
        self.seq = 0
        self._recent = collections.deque(maxlen=RECENT_BLOCKS)
        if not new:
            b = load_blocks(self.dir)
            if len(b["block_seq"]):
                self.seq = int(b["block_seq"][-1]) + 1
                self.dropped = int(b["dropped_cum"][-1])

    def append(self, I_A, V_V, host_ns: int, dropped_cum: int) -> None:
        I_A = np.asarray(I_A, dtype=np.float32)
        V_V = np.asarray(V_V, dtype=np.float32)
        with self.lock:
            gap = max(0, int(dropped_cum) - self.dropped)
            if gap:
                np.full((gap, 2), np.nan, dtype=np.float32).tofile(self._f)
                self.n_total += gap
                self.dropped = int(dropped_cum)
            first = self.n_total
            np.column_stack([I_A, V_V]).astype(np.float32).tofile(self._f)
            self._f.flush()
            self.n_total += len(I_A)
            self._w.writerow([self.seq, first, len(I_A), int(host_ns), self.dropped, gap])
            self._recent.append((first + len(I_A) - 1, int(host_ns)))
            self.seq += 1

    def read(self, i0: int, i1: int):
        with self.lock:
            n = self.n_total
        i0, i1 = max(0, i0), min(i1, n)
        if i1 <= i0:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        m = np.memmap(self._samples_path, dtype=np.float32, mode="r", shape=(n, 2))
        out = np.array(m[i0:i1])
        return out[:, 0], out[:, 1]

    def recent_blocks(self):
        with self.lock:
            arr = np.array(self._recent, dtype=np.float64).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    def close(self) -> None:
        self._f.close()
        self._b.close()


def load_blocks(run_dir) -> dict:
    p = Path(run_dir) / "blocks.csv"
    data = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    if data.size == 0:
        data = np.empty((0, len(HEADER)), dtype=np.int64)
    return {k: data[:, j] for j, k in enumerate(HEADER)}


def open_samples(run_dir):
    p = Path(run_dir) / "samples.f32"
    n = os.path.getsize(p) // 8
    return np.memmap(p, dtype=np.float32, mode="r", shape=(n, 2))
```

```python
# host/clockfit.py
"""Map Monsoon sample index -> EliteBook wall-clock ns (PTP-disciplined).

A USB block reaches the host at least some minimum latency after its last sample.
So the true clock line is the *lower envelope* of (last_idx, host_ns): take the
minimum-latency block in each of `segments` equal index segments and fit a
line through those. The remaining constant (minimum USB latency) is removed
later by the sync pulses in align.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOMINAL_NS = 200_000.0  # 1 / 5 kHz


@dataclass
class SampleClock:
    a_ns: float
    b_ns: float

    def to_ns(self, idx):
        return self.a_ns + self.b_ns * np.asarray(idx, dtype=np.float64)

    def to_idx(self, t_ns):
        return (np.asarray(t_ns, dtype=np.float64) - self.a_ns) / self.b_ns


def fit_sample_clock(last_idx, host_ns, rate_hz: int = 5000, segments: int = 20) -> SampleClock:
    x = np.asarray(last_idx, dtype=np.float64)
    y = np.asarray(host_ns, dtype=np.float64)
    b0 = 1e9 / rate_hz
    if x.size < 2 * segments:
        k = int(np.argmin(y - b0 * x))
        return SampleClock(a_ns=y[k] - b0 * x[k], b_ns=b0)
    r = y - b0 * x
    edges = np.linspace(x.min(), x.max() + 1, segments + 1)
    px, py = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.any():
            k = np.flatnonzero(m)[np.argmin(r[m])]
            px.append(x[k])
            py.append(y[k])
    px, py = np.array(px), np.array(py)
    b, a = np.polyfit(px - px[0], py - py[0], 1)   # centered: keeps ns precision at ~1.8e18
    return SampleClock(a_ns=py[0] + a - b * px[0], b_ns=b)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd host && ~/.local/bin/uv run pytest -q tests/test_store.py tests/test_clockfit.py`
Expected: `6 passed`

- [ ] **Step 6: Write the failing HTTP test**

```python
# host/tests/test_recorder_http.py
import json
import threading
import time
import urllib.request

import numpy as np

from recorder import make_server, window_stats
from store import SampleStore


def _store_with_known_power(tmp_path, t0_ns):
    s = SampleStore(tmp_path)
    for k in range(100):  # 100 blocks x 50 samples = 1 s at 5 kHz, P = 2 W then 4 W
        I = np.full(50, 0.4 if k < 50 else 0.8)
        last_idx = k * 50 + 49
        s.append(I, np.full(50, 5.0), host_ns=int(t0_ns + last_idx * 200_000), dropped_cum=0)
    return s


def test_window_stats_by_wall_clock(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    w = window_stats(s, t0 + 600_000_000, t0 + 800_000_000)  # second half only
    assert w["complete"] is True
    assert abs(w["mean_P"] - 4.0) < 1e-6 and w["count"] == 1000
    assert window_stats(s, t0 + 600_000_000, t0 + 5_000_000_000)["complete"] is False


def test_window_ignores_nan(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = SampleStore(tmp_path)
    s.append(np.full(100, 0.4), np.full(100, 5.0), host_ns=t0 + 99 * 200_000, dropped_cum=0)
    s.append(np.full(100, 0.4), np.full(100, 5.0), host_ns=t0 + 299 * 200_000, dropped_cum=100)
    w = window_stats(s, t0, t0 + 300 * 200_000)
    assert w["count"] == 200 and abs(w["mean_P"] - 2.0) < 1e-6


def test_http_status_and_window(tmp_path):
    t0 = 1_800_000_000_000_000_000
    s = _store_with_known_power(tmp_path, t0)
    srv = make_server(s, lambda: {"recording": True, "sample_hz": 5000.0}, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        st = json.loads(urllib.request.urlopen(base + "/status").read())
        assert st["recording"] is True and st["n_total"] == 5000
        w = json.loads(urllib.request.urlopen(
            f"{base}/window?t0_ns={t0}&t1_ns={t0 + 200_000_000}").read())
        assert abs(w["mean_P"] - 2.0) < 1e-6
    finally:
        srv.shutdown()
```

- [ ] **Step 7: Run it to verify it fails**

Run: `cd host && ~/.local/bin/uv run pytest -q tests/test_recorder_http.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'recorder'`

- [ ] **Step 8: Write `host/recorder.py`**

The device process reuses the `StreamingEngine.__vectorProcess` hook from `~/monsoon/monsoon_monitor.py`. That hook is the only way the `monsoon` 0.1.88 library hands over every sample without losing data. This version adds the host arrival stamp and never switches Vout off.

```python
#!/usr/bin/env python3
"""Always-on Monsoon HVPM recorder for the Pi 5 power lab.

The HVPM main channel *powers the Pi*, so this process never switches Vout off
(stopping it leaves the output at the last setting). Each USB block is stored with its
host arrival time (PTP-disciplined time.time_ns) and the cumulative dropped count.
HTTP on --port:
  GET /status                     -> recording, serial, vout_V, n_total, dropped, sample_hz, run_dir
  GET /window?t0_ns=&t1_ns=       -> count, mean_P (W), max_P (W), complete

    uv run python recorder.py --vout 5.2 --current-limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
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
log = logging.getLogger("recorder")


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
                                     "dropped": store.dropped, "run_dir": str(store.dir)})
                elif u.path == "/window":
                    q = parse_qs(u.query)
                    self._send(200, window_stats(store, int(q["t0_ns"][0]), int(q["t1_ns"][0])))
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
    ap.add_argument("--current-limit", type=float, default=5.0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8051)
    a = ap.parse_args()
    if not (VOUT_MIN <= a.vout <= VOUT_MAX):
        raise SystemExit(f"--vout {a.vout} outside safe Pi 5 range [{VOUT_MIN}, {VOUT_MAX}]")
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
```

- [ ] **Step 9: Run all host tests**

Run: `cd host && ~/.local/bin/uv run pytest -q`
Expected: `9 passed`

- [ ] **Step 10: Commit**

```bash
git add host/pyproject.toml host/store.py host/clockfit.py host/recorder.py host/tests/
git commit -m "feat: always-on HVPM recorder (block-stamped store, clock fit, HTTP window)"
```

---

### Task 6: Offline alignment and labeling

**Files:**
- Create: `host/align.py`
- Test: `host/tests/test_align.py`

**Interfaces:**
- Consumes:
  - `load_blocks`, `open_samples` (Task 5)
  - `fit_sample_clock`, `SampleClock` (Task 5)
  - Pi `events.csv` (FIELDS from Task 1)
- Produces:
  - `class AlignError(RuntimeError)`
  - `align(recorder_dir, pi_run_dir, out_dir) -> dict`. It returns `alignment.json` contents: `{c0_ns, c1, max_residual_ms, rms_residual_ms, drift_ppm, n_edges, trains, pass, coarse_clock}`.
  - Writes `out_dir/alignment.json` and `out_dir/power_trace.parquet`. Parquet columns:
    - `t_s` (seconds from first event), `t_pi_ns`
    - `I_A`, `V_V`, `P_W`
    - `phase`, `mode`, `iter`, `layer_path`, `depth`
  - CLI: `uv run python align.py --recorder-dir DIR --pi-run DIR [--out DIR]`
  - Mapping: `monsoon_ns = coarse_ns(idx)`; `t_pi = monsoon_ns - (c0 + c1·(monsoon_ns - ref_ns))`, where `ref_ns` is the first Pi edge.

- [ ] **Step 1: Write the failing test**

```python
# host/tests/test_align.py
import csv
import json

import numpy as np
import pandas as pd
import pytest

from align import AlignError, align
from store import SampleStore

FIELDS = ["t_real_ns", "t_mono_ns", "kind", "name", "depth", "mode", "iter",
          "layer_path", "config_id", "window_id"]
PATTERN = ((1, 150), (0, 100), (1, 300), (0, 100), (1, 50))
T0 = 1_800_000_000_000_000_000           # recording start (true time of sample 0)
B = 200_000.0 * (1 + 30e-6)              # Monsoon oscillator 30 ppm fast
USB_MIN_MS = 2.5                         # constant latency the pulses must remove


def _train_edges(start_ns):
    t, out = start_ns, []
    for on, ms in PATTERN:
        out.append(("pulse_on" if on else "pulse_off", t))
        t += ms * 1_000_000
    out.append(("pulse_off", t))
    return out


def _build(tmp_path, with_pulses=True):
    rec, pi = tmp_path / "rec", tmp_path / "pi"
    pi.mkdir()
    n = int(20 * 5000)
    true_ns = T0 + B * np.arange(n)
    P = np.full(n, 3.2)
    trains = {"start": T0 + 2_000_000_000, "end": T0 + 15_000_000_000}
    events = [("marker", "warmup", 0, "", T0 + 1_000_000_000)]
    for tag, s in trains.items():
        events.append(("marker", f"sync:{tag}", 0, "", s - 500_000_000))
        for kind, t in _train_edges(s):
            events.append((kind, tag, 0, "", t))
    events.append(("marker", "forward", 0, "", T0 + 5_000_000_000))
    L0, L1 = T0 + 6_000_000_000, T0 + 6_400_000_000          # layer "0" span
    C0, C1 = T0 + 6_100_000_000, T0 + 6_200_000_000          # nested "0.0" span
    events += [("begin", "layer", 1, "0", L0), ("begin", "layer", 2, "0.0", C0),
               ("end", "layer", 2, "0.0", C1), ("end", "layer", 1, "0", L1)]
    events.append(("marker", "done", 0, "", T0 + 17_000_000_000))
    if with_pulses:
        for tag, s in trains.items():
            e = _train_edges(s)
            for (k0, t0), (_, t1) in zip(e, e[1:]):
                if k0 == "pulse_on":
                    P[(true_ns >= t0) & (true_ns < t1)] = 6.0
    P += np.random.default_rng(1).normal(0, 0.05, n)
    s = SampleStore(rec)
    blk, rng = 50, np.random.default_rng(2)
    for i0 in range(0, n, blk):
        last = i0 + blk - 1
        host = T0 + B * last + USB_MIN_MS * 1e6 + rng.exponential(2e6)
        s.append(P[i0:i0 + blk] / 5.0, np.full(blk, 5.0), int(host), 0)
    s.close()
    events.sort(key=lambda e: e[4])
    with open(pi / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for kind, name, depth, path, t in events:
            w.writerow({"t_real_ns": int(t), "t_mono_ns": int(t), "kind": kind, "name": name,
                        "depth": depth, "mode": "forward", "iter": 0, "layer_path": path,
                        "config_id": "", "window_id": 0})
    return rec, pi, (L0, L1, C0, C1)


def test_align_recovers_offset_and_passes(tmp_path):
    rec, pi, _ = _build(tmp_path)
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["n_edges"] == 12
    assert res["max_residual_ms"] < 0.5
    assert abs(res["drift_ppm"]) < 5
    saved = json.loads((pi / "alignment.json").read_text())
    assert saved["pass"] is True


def test_align_labels_innermost_layer(tmp_path):
    rec, pi, (L0, L1, C0, C1) = _build(tmp_path)
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]
    assert at((C0 + C1) // 2)["layer_path"] == "0.0"
    assert at((C1 + L1) // 2)["layer_path"] == "0"
    gap = at(L1 + 200_000_000)
    assert gap["layer_path"] == "" and gap["phase"] == "forward"
    on = at(T0 + 2_000_000_000 + 400_000_000)          # inside the 300 ms ON slot
    assert on["phase"] == "sync:start" and on["P_W"] > 5.0


def test_align_raises_when_no_pulses(tmp_path):
    rec, pi, _ = _build(tmp_path, with_pulses=False)
    with pytest.raises(AlignError, match="sync pulses not found"):
        align(rec, pi, pi)
    assert not (pi / "power_trace.parquet").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd host && ~/.local/bin/uv run pytest -q tests/test_align.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'align'`

- [ ] **Step 3: Write `host/align.py`**

```python
#!/usr/bin/env python3
"""Align a Pi run's events.csv with the always-on Monsoon recording and label every sample.

1. Coarse clock: lower-envelope fit of block arrival stamps around the run (clockfit).
2. Sync pulses: detect power edges near each logged Pi edge, then fit
   offset(t) = c0 + c1 * (t - ref) over all edges (USB latency + oscillator drift).
3. Label: each sample gets the latest marker (phase), the latest event's mode/iter,
   and the innermost enclosing layer span.

    uv run python align.py --recorder-dir data/<session> --pi-run ../runs/<cell>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clockfit import fit_sample_clock
from store import load_blocks, open_samples

PASS_MS = 1.0
SEARCH_MS = 60.0          # edge search half-window around the coarse prediction
SMOOTH = 5                # centered moving average (1 ms), symmetric -> no edge shift
PAD_S = 5.0


class AlignError(RuntimeError):
    pass


def _smooth(p):
    k = np.ones(SMOOTH) / SMOOTH
    return np.convolve(np.nan_to_num(p, nan=np.nanmedian(p)), k, mode="same")


def _find_edge(t_ns, p, t_guess, rising, lo, hi):
    m = (t_ns >= t_guess - SEARCH_MS * 1e6) & (t_ns <= t_guess + SEARCH_MS * 1e6)
    idx = np.flatnonzero(m)
    if idx.size < 3:
        return None
    th = (lo + hi) / 2.0
    seg = p[idx]
    above = seg > th
    flips = np.flatnonzero(above[1:] != above[:-1])
    flips = [f for f in flips if above[f + 1] == rising]
    if not flips:
        return None
    f = min(flips, key=lambda f: abs(t_ns[idx[f]] - t_guess))
    i, j = idx[f], idx[f + 1]
    frac = (th - p[i]) / (p[j] - p[i]) if p[j] != p[i] else 0.0
    return t_ns[i] + frac * (t_ns[j] - t_ns[i])


def _edges(ev):
    e = ev[ev["kind"].isin(["pulse_on", "pulse_off"])]
    return list(e.groupby("name", sort=False))


def align(recorder_dir, pi_run_dir, out_dir) -> dict:
    recorder_dir, pi_run_dir, out_dir = Path(recorder_dir), Path(pi_run_dir), Path(out_dir)
    ev = pd.read_csv(pi_run_dir / "events.csv", keep_default_na=False,
                     dtype={"kind": str, "name": str, "mode": str, "iter": str,
                            "layer_path": str, "config_id": str})
    ev = ev.sort_values("t_real_ns", kind="stable").reset_index(drop=True)
    t_first, t_last = int(ev["t_real_ns"].iloc[0]), int(ev["t_real_ns"].iloc[-1])

    b = load_blocks(recorder_dir)
    last_idx = b["first_idx"] + b["n"] - 1
    sel = (b["host_ns"] >= t_first - PAD_S * 1e9) & (b["host_ns"] <= t_last + PAD_S * 1e9)
    if sel.sum() < 2:
        raise AlignError("recording does not cover this run's time range")
    clock = fit_sample_clock(last_idx[sel], b["host_ns"][sel])

    samples = open_samples(recorder_dir)
    i0 = max(0, int(np.floor(clock.to_idx(t_first - 1e9))))
    i1 = min(len(samples), int(np.ceil(clock.to_idx(t_last + 1e9))) + 1)
    IV = np.array(samples[i0:i1], dtype=np.float64)
    t_m = clock.to_ns(np.arange(i0, i1))
    P = IV[:, 0] * IV[:, 1]
    ps = _smooth(P)

    pi_t, mon_t = [], []
    trains = []
    for name, grp in _edges(ev):
        ts = grp["t_real_ns"].to_numpy(dtype=np.float64)
        win = (t_m >= ts[0] - 0.45e9) & (t_m < ts[0] - 0.05e9)
        busy = (t_m >= ts[0]) & (t_m < ts[-1])
        if not win.any() or not busy.any():
            raise AlignError(f"sync pulses not found for train '{name}' (no samples)")
        lo, hi = np.median(ps[win]), np.percentile(ps[busy], 90)
        if hi - lo < 0.5:
            raise AlignError(f"sync pulses not found for train '{name}' "
                             f"(step {hi - lo:.2f} W < 0.5 W)")
        # guess = Pi time + offset estimated from the first rising edge
        first = _find_edge(t_m, ps, ts[0], True, lo, hi)
        if first is None:
            raise AlignError(f"sync pulses not found for train '{name}' (no first edge)")
        off = first - ts[0]
        found = 0
        for kind, t in zip(grp["kind"], ts):
            got = _find_edge(t_m, ps, t + off, kind == "pulse_on", lo, hi)
            if got is not None:
                pi_t.append(t)
                mon_t.append(got)
                found += 1
        trains.append({"name": name, "edges_found": found, "edges_logged": len(ts)})
        if found < len(ts) - 1:
            raise AlignError(f"sync pulses not found for train '{name}' "
                             f"({found}/{len(ts)} edges)")

    pi_t, mon_t = np.array(pi_t), np.array(mon_t)
    ref = pi_t[0]
    offs = mon_t - pi_t
    if len(trains) >= 2:
        c1, c0 = np.polyfit(pi_t - ref, offs, 1)
    else:
        c1, c0 = 0.0, float(np.median(offs))
    resid_ms = (offs - (c0 + c1 * (pi_t - ref))) / 1e6
    res = {
        "c0_ns": float(c0), "c1": float(c1), "drift_ppm": float(c1 * 1e6), "ref_ns": float(ref),
        "max_residual_ms": float(np.abs(resid_ms).max()),
        "rms_residual_ms": float(np.sqrt(np.mean(resid_ms ** 2))),
        "n_edges": int(len(pi_t)), "trains": trains,
        "coarse_clock": {"a_ns": clock.a_ns, "b_ns": clock.b_ns},
    }
    res["pass"] = bool(res["max_residual_ms"] < PASS_MS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alignment.json").write_text(json.dumps(res, indent=2))
    if not res["pass"]:
        raise AlignError(f"alignment residual {res['max_residual_ms']:.3f} ms >= {PASS_MS} ms")

    # ---- label samples on the Pi clock ------------------------------------------
    t_pi = (t_m - c0 + c1 * ref) / (1.0 + c1)
    keep = (t_pi >= t_first) & (t_pi <= t_last)
    t_pi, IV, P = t_pi[keep], IV[keep], P[keep]

    et = ev["t_real_ns"].to_numpy(dtype=np.float64)
    k = np.searchsorted(et, t_pi, side="right") - 1
    mode = ev["mode"].astype(str).to_numpy()[k]
    it = ev["iter"].astype(str).to_numpy()[k]
    mk = ev[ev["kind"] == "marker"]
    mt = mk["t_real_ns"].to_numpy(dtype=np.float64)
    km = np.searchsorted(mt, t_pi, side="right") - 1
    phase = np.where(km >= 0, mk["name"].to_numpy()[np.clip(km, 0, None)], "idle")

    layer = np.full(t_pi.size, "", dtype=object)
    depth = np.zeros(t_pi.size, dtype=np.int16)
    spans = _layer_spans(ev)
    for d in sorted(spans):
        s = spans[d]
        st, en, paths = s[:, 0].astype(np.float64), s[:, 1].astype(np.float64), s[:, 2]
        j = np.searchsorted(st, t_pi, side="right") - 1
        ok = (j >= 0) & (t_pi < en[np.clip(j, 0, None)])
        layer[ok] = paths[j[ok]]
        depth[ok] = d

    df = pd.DataFrame({
        "t_s": (t_pi - t_first) / 1e9, "t_pi_ns": t_pi.astype(np.int64),
        "I_A": IV[:, 0].astype(np.float32), "V_V": IV[:, 1].astype(np.float32),
        "P_W": P.astype(np.float32), "phase": phase, "mode": mode, "iter": it,
        "layer_path": layer.astype(str), "depth": depth,
    })
    df.to_parquet(out_dir / "power_trace.parquet", index=False)
    return res


def _layer_spans(ev) -> dict:
    """depth -> array[(start_ns, end_ns, path)] sorted by start (same-depth spans don't overlap)."""
    lay = ev[ev["kind"].isin(["begin", "end"])]
    open_: dict = {}
    out: dict = {}
    for kind, path, d, t in zip(lay["kind"], lay["layer_path"], lay["depth"], lay["t_real_ns"]):
        key = (path, int(d))
        if kind == "begin":
            open_.setdefault(key, []).append(t)
        elif open_.get(key):
            out.setdefault(int(d), []).append((open_[key].pop(), t, path))
    return {d: np.array(sorted(v, key=lambda r: r[0]), dtype=object) for d, v in out.items()}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recorder-dir", required=True)
    ap.add_argument("--pi-run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        res = align(a.recorder_dir, a.pi_run, a.out or a.pi_run)
    except AlignError as e:
        raise SystemExit(f"[FAIL] {e}")
    print(f"[ok] edges={res['n_edges']} max_residual={res['max_residual_ms']:.3f} ms "
          f"rms={res['rms_residual_ms']:.3f} ms drift={res['drift_ppm']:.2f} ppm "
          f"usb_offset={res['c0_ns'] / 1e6:.3f} ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all host tests**

Run: `cd host && ~/.local/bin/uv run pytest -q`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add host/align.py host/tests/test_align.py
git commit -m "feat: offline sync-pulse alignment + per-sample phase/layer labeling"
```

---

### Task 7: Deploy, power bring-up, and hardware validation

This task runs against real hardware. Each step names the machine it runs on, and every step marked **USER** needs a person.

**Files:**
- Modify: `README.md` (new §14 "Monsoon backend runbook")
- Create: `docs/superpowers/plans/sp1-bringup-log.md` (measured numbers)

- [ ] **Step 1: Deploy the branch** (from the NUC)

```bash
git push maccpi@192.168.0.243:Desktop/power-measurement-pi monsoon-experiment
ssh maccpi@192.168.0.243 'cd ~/Desktop/power-measurement-pi && git checkout monsoon-experiment'
ssh macc2026@192.168.0.11 'git init -q ~/power-measurement-pi'
git push macc2026@192.168.0.11:power-measurement-pi monsoon-experiment
ssh macc2026@192.168.0.11 'cd ~/power-measurement-pi && git checkout -q monsoon-experiment && cd host && ~/.local/bin/uv sync && ~/.local/bin/uv run pytest -q'
ssh maccpi@192.168.0.243 'cd ~/Desktop/power-measurement-pi && ~/.local/bin/uv run pytest tests/ -q'
```

Expected: host `12 passed`. On the Pi, all of `tests/` passes, including `test_layer_events.py`.

- [ ] **Step 2: Give the EliteBook key-based SSH to the Pi** (from the NUC, no password needed)

```bash
ssh macc2026@192.168.0.11 'test -f ~/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub' \
  | ssh maccpi@192.168.0.243 'cat >> ~/.ssh/authorized_keys'
ssh macc2026@192.168.0.11 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new maccpi@192.168.0.243 hostname'
```

Expected: `pimacc`.

- [ ] **Step 3 (USER): Shut down the Pi cleanly**

`! ssh -t maccpi@192.168.0.243 sudo poweroff`. Then confirm on the EliteBook that the current has fallen to the halted level:

```bash
ssh macc2026@192.168.0.11 'tail -1 $(ls -d ~/monsoon/data/*_rpi4b | tail -1)/summary_1s.csv'
```

Expected: the 3rd column (avg mA) is well below the ~638 mA running level. Pi 5 halt draw is roughly 50–150 mA.

- [ ] **Step 4: Stop the old monitor** (EliteBook). This switches Vout to 0, which is safe now that the Pi is halted.

```bash
ssh macc2026@192.168.0.11 'pkill -INT -f "monsoon_monitor.py --vout 5"; sleep 3; pgrep -af monsoon_monitor || echo stopped'
```

- [ ] **Step 5 (USER): Swap in the modified USB-C cable.** Tell Claude when it is done.

- [ ] **Step 6: Start the recorder at Vout 5.10 V** as a user service, so it survives SSH logout:

```bash
ssh macc2026@192.168.0.11 'cd ~/power-measurement-pi/host && systemd-run --user --unit=monsoon-recorder --same-dir \
  ~/.local/bin/uv run python recorder.py --vout 5.10 --current-limit 5'
sleep 60
curl -s http://192.168.0.11:8051/status
```

Expected:
- `"recording": true`, `"sample_hz"` ≈ 5000, `"serial": 36517`.
- The Pi boots and answers `ssh maccpi@192.168.0.243 hostname` within about 60 s.
- If port 8051 is blocked by the EliteBook firewall, open it (sudo; confirm with the user first). Or temporarily run `run.py` with `--recorder-url` through an SSH tunnel: `ssh -L 8051:localhost:8051`.

- [ ] **Step 7: Characterize the lead resistance and choose the final Vout** (on the Pi)

```bash
ssh maccpi@192.168.0.243 'vcgencmd pmic_read_adc EXT5V_V; vcgencmd get_throttled;
  for i in 1 2 3 4; do timeout 30 python3 -c "while 1: pass" & done; sleep 20;
  vcgencmd pmic_read_adc EXT5V_V; vcgencmd get_throttled; wait'
curl -s "http://192.168.0.11:8051/status"
```

Compute `R = (Vout − EXT5V_burn) / I_burn`, where `I_burn` is from the recorder's `/window` over the burn. Then choose `Vout = min(5.40, 4.95 + R × 2.0)`.
- If that Vout differs from 5.10 by more than 0.05 V: Pi `sudo poweroff` (USER) → `systemctl --user stop monsoon-recorder` → restart with the new `--vout`.

Gate:
- `EXT5V_V ≥ 4.85 V` during the burn.
- `get_throttled` = `0x0` after the burn.
- Log Vout, R, idle and burn EXT5V, and idle and burn W in `sp1-bringup-log.md`.

- [ ] **Step 8: Check the Pi clock offset** (lab time)

```bash
scp macc2026@192.168.0.11:Desktop/lab_time_sync/tools/check_sync.py /tmp/ && scp /tmp/check_sync.py maccpi@192.168.0.243:/tmp/
ssh maccpi@192.168.0.243 'python3 /tmp/check_sync.py | grep offset'
```

Gate: `|offset| < 0.2 ms`.

- [ ] **Step 9: Sync validation, 20 trains**

```bash
ssh maccpi@192.168.0.243 'cd ~/Desktop/power-measurement-pi && ~/.local/bin/uv run python src/sync_check.py --trains 20 --spacing 2 --out runs/sync_check'
ssh macc2026@192.168.0.11 'cd ~/power-measurement-pi && mkdir -p runs && rsync -a maccpi@192.168.0.243:Desktop/power-measurement-pi/runs/sync_check runs/ && \
  cd host && ~/.local/bin/uv run python align.py --recorder-dir $(ls -d data/*/ | tail -1) --pi-run ../runs/sync_check'
```

Gate: `[ok] edges≈120`, `max_residual < 1.0 ms`, `|drift| < 50 ppm`. Record the numbers, including `usb_offset`, in the bring-up log.

- [ ] **Step 10: Golden smoke run, monsoon backend**

```bash
ssh maccpi@192.168.0.243 'cd ~/Desktop/power-measurement-pi && ~/.local/bin/uv run python src/run.py --sampler monsoon \
  --task image --model yolo11n.pt --check-image bus.jpg --frames 10 --bench 1 --out runs/m4_smoke_yolo11n'
ssh macc2026@192.168.0.11 'cd ~/power-measurement-pi && rsync -a maccpi@192.168.0.243:Desktop/power-measurement-pi/runs/m4_smoke_yolo11n runs/ && \
  cd host && ~/.local/bin/uv run python align.py --recorder-dir $(ls -d data/*/ | tail -1) --pi-run ../runs/m4_smoke_yolo11n && \
  ~/.local/bin/uv run python -c "
import pandas as pd; d=pd.read_parquet(\"../runs/m4_smoke_yolo11n/power_trace.parquet\")
print(d.groupby(\"phase\")[\"P_W\"].agg([\"mean\",\"count\"]).sort_values(\"count\").tail(8))
f=d[d.phase==\"forward\"]; print(\"forward samples with layer label:\", (f.layer_path!=\"\").mean())
print(\"unlabeled (idle) fraction:\", (d.phase==\"idle\").mean())"'
```

Gate:
- `run.py` prints `[ok]`.
- Alignment passes.
- `forward` mean P is greater than `baseline_A` mean P.
- At least 80% of forward samples carry a layer label.
- The unlabeled (`idle`) fraction is below 1%, meaning every sample between the first and last event belongs to a phase.

- [ ] **Step 11: Write the README §14 runbook and the bring-up log**

The README section covers:
- the topology table (NUC / EliteBook / Pi)
- starting and stopping the recorder, and the warning that stopping it never removes power, but changing Vout requires a Pi shutdown first
- `--sampler monsoon`
- `align.py`
- the gates above

Update the stale access line in §2/§4 to `maccpi@192.168.0.243`, `~/Desktop/power-measurement-pi`.

- [ ] **Step 12: Commit**

```bash
git add README.md docs/superpowers/plans/sp1-bringup-log.md
git commit -m "docs: Monsoon backend runbook + SP1 bring-up measurements"
```
