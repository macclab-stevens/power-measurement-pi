# SP1 Monsoon measurement plane: hardware bring-up log

Date: 2026-10-06. Source: the SDD ledger (`.superpowers/sdd/2026-10-06-sp1-monsoon-measurement-plane/progress.md`).
Pi 5 powered by the HVPM main channel via USB-C; EliteBook hosts the recorder. Entries in order.

| # | Step | Measurement / event | Outcome / decision |
|---|---|---|---|
| 1 | Deploy | `bc87fe2` to Pi and EliteBook; Pi 38 tests, host 25 tests passed; EliteBook-to-Pi key OK | User installed `pi-power-pin` + sudoers, shut Pi down, swapped cable; recorder at 5.10 V / 5 A (`data/20261006T183920`) |
| 2 | Pinning check | performance governor, 2400 MHz (`measure_clock` 2400027136); fan 4/4 at 8021 rpm; cooling_device0 = pwm-fan | Pinning works. `throttled=0x50000` already this boot |
| 3 | Idle, wiring 0 (5.10 V) | idle pinned 4.94 W, EXT5V 4.87 V | R about 0.24 ohm |
| 4 | Finding: `dropped` | HVPM `dropped` is per-packet, flips 0/14 (29 flips / 80 s), not cumulative; effective rate 4993-4995 Hz | R19: record raw only, no gap insertion. Fix `9b1daf0` |
| 5 | Restart at 5.25 V | Recorder 5.25 V / 6.0 A (`data/20261006T185228`), 4992 Hz; boot throttled 0x0 | R20: 25 W ceiling possible, keep cable, current limit 6 A (cap 8 A) |
| 6 | Idle and burn, wiring 0 (5.25 V) | idle pinned 4.92 W, EXT5V 5.02 V. 30 s 4-core burn: 7.73 W mean / 8.80 W peak, EXT5V 4.87 V, 2.4 GHz held, 47.7 C, throttled 0x0 | R about 0.25 ohm; undervoltage ceiling about 2.5 A (~12.5 W) on this cable |
| 7 | Soak (5 min) | peak 54.9 C, 2.4 GHz all samples, throttled 0x0, EXT5V 4.867-4.878 V, fan ~9040 rpm, power 7.79 to 7.89 W (leakage + temp) | PASS |
| 8 | Clock offset | Pi clock offset -0.005 ms vs TM2500C | OK |
| 9 | Sync check (20 trains) | 120/120 edges, max residual 0.257 ms, rms 0.100, drift -0.02 ppm, rise-fall 0.14 ms, envelope dev 0.027 ms, USB offset 0.64 ms, nan 0 | PASS |
| 10 | First smoke (yolo11n, 2400 MHz, fan full) | Undervoltage during warmup (0x0 to 0x50000), transient 0x50005, arm 1000 MHz x3 context rows. I max 3.01 A, p99.9 2.71 A. Cause: 0.25 ohm leads x ~3 A transients, Pi < 4.63 V | Acceptance FAILED correctly (under-voltage rejected) |
| 11 | Alignment on that run | 12/12 edges, max residual 0.158 ms, drift -0.04 ppm, USB offset 0.736 ms. Phase means: idle ~4.99 W, forward 6.58 W, bench 8.56 W. Forward layer-labeled 53% (gate 80%); idle frac 0.05%, ctx_read 1.2% | Alignment PASS. R21: 80% label gate not met by design for ultralytics (pre/post-processing outside hooks); SP2 target |
| 12 | Wiring fixes | (1) user's 2nd wire: R about 0.29 ohm. (2) new USB-C cable: R about 0.33 ohm, undervoltage at boot (2.02 A boot peak). (3) wire replaced (hot-swapped at live 5.25 V; Pi rebooted, no harm): R about 0.11 ohm incremental; idle 0.92 A, EXT5V 5.100 V; burn 1.46 A, 5.042 V; boot throttled 0x0 | Vout stays 5.25 V (extrapolated 3 A: 4.87 V, 5 A: ~4.65 V) |
| 13 | Smoke attempt 2 | Killed by controller's `\| head -30` (SIGPIPE), not a system fault; benches up to then clean (2.4 GHz, ~5000 samples/window) | Rerun with log to file |
| 14 | Passing smoke (yolo11n) | Aligned, schema 4, 2400 MHz, throttled 0 to 0, 1.21M rows, idle 4.87 W, forward 9.6 W/iter (iters 1-9: 9.56-9.79 W), bench mean 9.1 W, I max 2.86 A, forward layer-labeled 65% | `[ok]`, residual 0.166 ms |
| 15 | Sample-loss investigation | Unreported losses of 5-20 ms (26-100 samples) during runs. Pi-clock check: smoke run 4991.851 Hz vs clean 4992.330 Hz, ~113 samples truly lost in 237 s. None over 10+ min idle without queries; reproduced by `/window` polling with Pi idle (15 q/s: 5-10 losses / 2-3 min; 1.5 q/s: 3 losses / 4 min). Own-core device process and niced main did not fix it. EliteBook PTP stack also flapping (`phc2sys` timeouts, restart every ~1-1.5 min), a separate issue | Detector added: `368ffb7`, `5450fc2`, `79eb58a` (R23, R24) |
| 16 | Slip detector validation | First version on the real smoke run: 4 slips / 103 samples (Pi-clock est ~113). Fix round 2 (`79eb58a`): real fixture 4 slips / 106 samples at true times, rate 4992.32 Hz, 100 seeds no false positives | Accepted |
| 17 | User decision | Accept occasional packet loss; plan 10 runs per model and train on the aggregate | Losses flagged (`slip_warning`), not fatal |

Minor open item: `sync_check` drift read 5.80 ppm after the slip-detector change vs -0.02 ppm earlier; reproduced by pre-change code, cause unexplained, deferred.

## Rulings

- Ruling R1: T4b implementer locates --pin-freq code by content, not line numbers — line refs are stale after T4 — cost if wrong: none (content anchors are unique).
- Ruling R2: Pi-side torch tests run by rsync'ing the working tree to maccpi@192.168.0.243:~/sdd-stage/ and running `~/.local/bin/uv run --project ~/Desktop/power-measurement-pi --with pytest pytest tests/...` there — no push to the Pi repo before Task 7 — cost if wrong: an extra pytest download on the Pi.
- Ruling R3: host tests run by rsync'ing host/ to macc2026@192.168.0.11:~/sdd-stage/host and `uv run pytest` there (EliteBook is the target machine) — cost if wrong: uv downloads monsoon/pyarrow into that staging venv.
- Ruling R4: Tasks 1-3 (torch-free Pi-side, complete code in plan, NUC-testable, sequential deps) batched into one implementer dispatch + one review — cost if wrong: a larger review diff (~450 lines).
- Ruling R5: Tasks 4+4b dispatched together (task-4-brief.md already spans both; both edit run.py) — cost if wrong: larger review diff.
- Ruling R6: window() timeout = self._timeout (not max(2.0,…)) and raises; OSError/ValueError retried until deadline then RuntimeError — fail loudly beats under-counted energy — cost if wrong: a transient recorder blip >5 s fails a run (orchestrator retries cells).
- Ruling R7: run.py does not force --cpu-mhz/--fan-full in monsoon mode — unpinned diagnostic runs stay possible; the SP3 orchestrator and Task 7 always pass both — cost if wrong: an unpinned run could slip into the dataset (run_meta.cpu_mhz_pinned=null makes it filterable).
- Ruling R8: dropped accounting via (raw-last)&0xFFFF, delta>25000 = counter reset (flag, no gap), blocks.csv gains dropped_raw,counter_reset — cost if wrong: if the field is per-packet rather than cumulative, gaps are mis-sized; T7 hardware check compares n_total vs elapsed×5000.
- Ruling R9: mode/iter kept only where phase == the event's mode label, else "" (align-side) — cost if wrong: a mode phase whose marker name differs from its mode label loses mode/iter (run.py marks phase with the mode name, so none today).
- Ruling R10: alignment.json reports rise_bias_ms / fall_bias_ms (mean residual per edge direction) + rise_minus_fall_ms; no lag subtraction (not identifiable from pulses alone); T7 sync check characterizes it — cost if wrong: an undetected symmetric lag shifts labels; bounded by syncpulse change in R11.
- Ruling R11: syncpulse workers poll a lock-free RawValue flag (spin while set; sleep 200 µs while clear) instead of mp.Event — removes lock contention on falling edges and bounds rising wake lag ≤0.2 ms — cost if wrong: slightly higher OFF-slot power (constant, absorbed by threshold).
- Ruling R12: run.py sets prof.events=None for the duration of bench_configs (restored in finally) — benches are labeled by bench:<cid> phase, layer spans there are redundant — cost if wrong: no per-layer label inside bench windows (phase still identifies config).
- Ruling R13: cheap minors folded into the Task 6 fix: no astype(str) on layer (object/categorical), NaN-interpolating _smooth, single-train test, mode/iter label test.
- Ruling R14: one fix wave (final-fix-brief.md F1-F5) covering all 4 Important + minors 2,3,4,5,6 + deferred "linearity"/"recorder_status unguarded" — cost if wrong: larger wave, more review surface.
- Ruling R15: envelope linearity gate = 1.0 ms (reviewer suggested ~0.5) to match PASS_MS until T7 measures real USB latency floor stability — cost if wrong: a 0.5–1 ms slip passes (labels off by ≤1 ms).
- Ruling R16: drop spec's ±1 s pulse-uniqueness test — search is ±60 ms around PTP-coarse prediction; >60 ms coarse error fails loudly — cost if wrong: none expected.
- Ruling R17: live-window USB-latency bias (~0.6–0.9% of ΔP in P_delta_ABAB) left uncorrected — authoritative values are offline-aligned — cost if wrong: live perop P_delta slightly low.
- Ruling R18: F3.5 residual is load-bearing (undervoltage during benches = the perop data) — smallest change: move thr1 read after bench_configs try/finally; fold in: pause ctx reads during sync pulses (out-of-scope obs: vcgencmd bursts can land in a pulse train), make ctx label-isolation test meaningful (ctx events carry mode=""/iter="99"), fix "get_throttled failed" wording. One targeted dispatch + scoped re-review — cost if wrong: one extra review round.
- Ruling R19: stop gap insertion from `dropped`; record raw value only (diagnostic); time integrity relies on host-stamp lower-envelope fit + 1 ms linearity gate (F1) — supersedes R8 — cost if wrong: real unreported drops become index slips; gate fails the run (loud), not silent.
- Ruling R20: user: 25 W ceiling possible, keep current cable → recorder --current-limit default 6.0 A (cap 8 A), restart at Vout 5.25 V; heavy (>~2.5 A) runs will fail the undervoltage gate by design — cost if wrong: Vout 5.25 V at Pi under no load ≈5.18 V (within USB spec).
- Ruling R21: 80% forward-layer-label gate is not met by design for ultralytics models (pre/post-proc outside hooks); keep as SP2 target, not an SP1 code defect — cost if wrong: unlabeled 47% of forward time until SP2.
- Ruling R23: envelope-linearity and drift gates become warnings (slip_warning + detected slip list in alignment.json); pulse-residual gate stays fatal — implements user decision; cost if wrong: per-layer labels after a slip shifted by the slip size (5–20 ms) in flagged runs.
- Ruling R24: suffix-minimum detector with median-of-chunks rate, tail guard 10 blocks; drift-only → drift_warning (not slip_warning) — cost if wrong: permanent latency-floor rise counted as slip; sub-1 ms losses invisible.
