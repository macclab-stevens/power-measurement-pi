# Dataset v2

Dataset v1 (224 cells, October 2026) measured six kinds of workload at three clock speeds. Dataset
v2 is the expansion that makes a power predictor harder to fool: more YOLO models, INT8 models,
other image sizes, batch sizes and thread counts, a free-running clock, other fan and serving
conditions, long runs, and hardware performance counters on every cell.

This page is the short version of what v2 is, what the team decided and found while building it,
how to set up the Pi once, and which probes to run before any big matrix. The how-to-run parts are
in [README section 5.4.2](../README.md#542-the-dataset-v2-matrix) and
[INSTRUCTIONS.md](../INSTRUCTIONS.md); the file and field details are in
[README sections 6 and 11](../README.md#6-what-the-output-files-mean).

## Contents

1. [What v2 adds](#1-what-v2-adds)
2. [v1 versus v2: the counter split](#2-v1-versus-v2-the-counter-split)
3. [The holdout](#3-the-holdout)
4. [Decisions and findings so far](#4-decisions-and-findings-so-far)
5. [One-time setup](#5-one-time-setup)
6. [Phase 0 probes](#6-phase-0-probes)
7. [Phase 1 and Phase 2](#7-phase-1-and-phase-2)
8. [Reading a v2 cell](#8-reading-a-v2-cell)

---

## 1. What v2 adds

| | v1 | v2 |
|---|---|---|
| Models | yolov8n, yolo11n, 3 classifiers, MiniGPT, training | the same YOLO nano, plus yolov8s / yolov8m / yolo11s / yolo11m |
| Precision | FP32 | FP32, plus INT8 static yolo11n / yolo11s (FP16 optional) |
| Image size, batch, threads | 640 px, batch 1, 4 threads | 320 / 480 / 640 / 960 px, batch 1 / 2 / 4 / 8, 1 / 2 / 4 threads |
| Input picture | one standard picture | also an empty and a crowded picture, and a stream of 50 different pictures |
| Clock | pinned 1500 / 2000 / 2400 MHz | also free-running (DVFS) |
| Cooling | fan at full speed | also fan `auto` and fan `off` (soaks keep the fan at full speed), and 30-minute soaks |
| Serving | 10 requests, 200 ms gaps | also 1, 5 and 15 requests per second, and two workloads at once |
| Counters | none (a sensor snapshot every 5 s) | `perf` hardware counters 10 times a second and a 1 Hz CPU clock / temperature / fan log on every cell |
| Accuracy | not recorded | the model's mAP on 500 COCO pictures travels with every YOLO cell (`map_reference`) |
| Derived numbers | computed afterwards by hand | `derived.json`: energy and latency per frame |
| Safety | no temperature limit inside the run (cells start below 50 C) | a 1 Hz watchdog aborts at 85 C and stops at 90 C |

The cells are grouped in families `A`, `A-input`, `B`, `C`, `D` and `soak`, plus a `ref` family that
re-runs v1 command lines. The family table, cell counts and flags are in
[README section 5.4.2](../README.md#542-the-dataset-v2-matrix).

## 2. v1 versus v2: the counter split

**The 224 v1 cells have no hardware counters and cannot get them afterwards.** Every
dataset-version-2 cell has `counters.csv`, the `ctr_*` / `pct_*` / `sys_*` columns in
`power_trace.parquet`, and `perf_counters.enabled: true` in `run_meta.json`. Any newer run may have
them too (counters are on by default), but a v1 cell never does.

- A predictor that uses **counter features** trains on **v2 cells only**.
- A predictor that does **not** use them can use v1 and v2 together.

Both are legitimate. Nobody should train on counter features with 224 rows that silently have none.
`run_meta.json` says which a cell is: `dataset_version` 1 or 2.

**The `ref` family and the confounds.** v1 ran with the Monsoon at 5.25 V on Pi kernel 6.18.50. v2
runs at 5.35 V (section 4) on kernel 6.18.60 (installing `perf` upgraded the kernel; the decision
was to stay on it). So the v1-to-v2 comparison has two changes in it. The `ref` family measures them
together: yolov8n and yolo11n with the **exact v1 command lines** at 1500, 2000 and 2400 MHz, 3
repeats, `--perf-counters off`, no `--holdout`. These cells stay `dataset_version` 1 and are not
counter data; they are for comparing against the 224 v1 cells.

`--perf-counters on` is the default for every run, including a v1-shaped command line, so
`./lab run yolo11n` now also records counters (its `dataset_version` stays 1). So: counters are
guaranteed in version-2 cells, possible in any new run, and absent from the 224 v1 cells. The `ref` family
passes `off` so that it is as close to v1 as possible.

## 3. The holdout

A holdout tag in `run_meta.json["holdout"]` says which cells to keep out of predictor training, so
that the predictor can be tested on models it has never seen.

| Model | Tag |
|---|---|
| yolo11s, yolov8s | `primary-s` (the primary holdout: interpolation between the nano and medium models) |
| yolo11m, yolov8m | `secondary-m` (a second, harder holdout) |
| everything else | `null` |

The tag follows the model, not the family, so every cell of an s-model (image sizes, threads,
INT8, ...) carries it. `matrix_v2.py` passes `--holdout` on every v2 cell (`none` for the nano
models). The `ref` cells and any v1-shaped command line carry no tag.

A run is **dataset version 2 if and only if** `--holdout` is given (with any value, even `none`) or
another v2 workload flag is used (`--governor ondemand`, `--fan auto|off`, `--serve-fps`,
`--soak-minutes`, `--frames-source stream`, `--input-image`, `--quant`, `--prune`,
`--concurrent`). Everything else is version 1. `--threads`, `--batch` and `--imgsz` alone do not make
a run v2, so the v1 command lines keep their version.

## 4. Decisions and findings so far

Dated 2026-11-19 and 2026-11-20 unless stated.

**Cut from the default matrix**

- **INT8 dynamic is cut.** Dynamic quantization only replaces Linear-type layers, and YOLO is
  almost all convolutions, so it quantizes 0 modules. `run.py` fails such a run rather than label
  float32 data INT8. The flag `--include-int8d` keeps the 9 cells buildable, and `./lab run` refuses these presets without `--include-cut`.
- **Pruning is cut.** Structured pruning of yolo11n without fine-tuning gives mAP50-95 of about 0
  at 30 % and 50 % (0.021 at 10 % in a one-off check, not recorded in `data/map_reference.json`), even after re-estimating the BatchNorm statistics on 200
  calibration pictures. The tool, the model files and the numbers are kept
  ([../tools/README.md](../tools/README.md)); `--include-prune` keeps the 6 cells buildable (`./lab run` needs `--include-cut`).
- **FP16 is optional (`--include-fp16`, 3 cells).** FP16 really executes in float16 on the Pi CPU
  (verified from the layer profile at run time), but in an informal check whole yolo11n at 640 was
  slower than FP32 (about 553 ms against about 330 ms per frame).

**What the INT8 cells are.** INT8 static (FX graph mode, `qnnpack`, 200 calibration pictures) has
80 quantized convolutions per model. The C2PSA attention blocks, the Detect decode and every SiLU
stay FP32 (listed in the model file's `fp32_modules`). It is not a fully INT8 network.

**mAP is measured on the Pi** (user decision). INT8 (`qnnpack` on ARM) and FP16 run different
kernels on the Pi than on the NUC. The NUC only builds weights offline and prepares data. The
current entries in `data/map_reference.json` are provisional numbers measured on the NUC; the Pi
measurement is plan Task 16 and has not been run. Until each entry carries
`measured_on.machine == "aarch64"`, `verify_cell.py` warns on every YOLO cell.

**Supply voltage 5.25 V to 5.35 V, then 5.40 V.** (2026-11-21: 5.40 V, the recorder's cap, after s/m-model runs still hit under-voltage at 5.35 V; a yolo11m loop then stayed at or above 4.91 V with no alarm.) After a Monsoon restart and a cable reseat, the Pi hit
under-voltage under YOLO load (`throttled` bits `0x50005`) and the firmware throttling cut the
forward-pass power from about 9.6 W to about 7.5 W. The Monsoon output was raised to 5.35 V (the
recorder allows up to 5.40 V). After the change, a YOLO loop showed no throttle bits
([README section 8](../README.md#8-power-safety-turning-the-pi-on-and-off-rewiring)). v1 data is at
5.25 V; see the `ref` family in section 2.

**Pi kernel 6.18.60** (v1 ran on 6.18.50). `apt install linux-perf` upgraded it. The decision was to
stay on 6.18.60 and quantify the difference with the `ref` family.

**Counters (`perf`).**
- With `perf_event_paranoid=2` (the Pi's default), `perf` counts the run's user-space code only;
  events print as `cycles:u`. That is what v2 records, and it needs no root.
- The default set is 10 events: 7 hardware events (`cycles`, `instructions`, `cache-references`,
  `cache-misses`, `branch-misses`, `stalled-cycles-backend`, `LLC-load-misses`) and 3 software
  events (`task-clock`, `context-switches`, `page-faults`). The 7 hardware events fit the Cortex-A76's
  counters at the same time and count 100 % of the time. An eighth made every hardware event
  count only 71-90 % of the time, which would break the 95 % coverage rule.
- `L1-dcache-loads` and `L1-dcache-load-misses` count exactly like `cache-references` and
  `cache-misses` on this CPU, so they are left out. `LLC-loads` and `bus-cycles` are left out too.
- `perf` starts its interval clock 14-37 ms after it is launched (measured). The offset is read
  from when its first output line appears and is corrected, recorded as `offset_ns` in
  `run_meta.json`.
- Overhead: in one A/B (same cell with counters on and off, taken at 5.25 V while the Pi was
  hitting under-voltage) counters on gave 0.15 W lower mean power and 6 % longer latency. That is one
  cell under bad conditions; Phase 0 repeats it (section 6).

**The layer table is unreliable for variants.** In `layer_table.csv`, `bytes_moved`, `macs` and
`arith_intensity` are worked out from the layer's parameters assuming 4-byte floats, and a
quantized module has no parameters to count, so they are wrong for FP16 and INT8 cells. Do not use
those three columns for them.

**Wi-Fi must be off while measuring.** The Pi's Wi-Fi profile (`stevens-iot`) has autoconnect off,
so it stays off across reboots. Bring it up only for downloads, then turn it off again. `./lab check` warns when `wlan0` is up.

**Concurrency** is one process with a second workload as a thread (`src/coload.py`), so there is one
sync envelope and `perf -p` still counts both. `events.csv` gains a `tag` column;
`experiment.py plan --v2 ... --no-concurrent` leaves those cells out. The short wait for the second
workload to finish after the main loop is its own phase, `concurrent_drain`, so it is not charged to
the last frame.

**The orchestrator detects the silent recorder stall** at `n_total` 2^32 - 1 (seen on 2026-10-17),
and `./lab check` shows it. Restart the recorder at least every 9 days
([README section 4.7 and 8](../README.md#47-start-the-recorder)).

## 5. One-time setup

Done once already (2026-11-19/20); this is how to redo it. Nothing here contains a password:
every `sudo` asks for it itself.

**5.1 Newer `pi-power-pin`** (needed for `unpin`, `fan auto` and `fan off`). From the NUC, with the
repo on the Pi up to date (README 12.2):

```bash
ssh -t maccpi@192.168.0.243 'sudo install -o root -g root -m 0755 ~/Desktop/power-measurement-pi/tools/pi-power-pin /usr/local/sbin/pi-power-pin && echo "maccpi ALL=(root) NOPASSWD: /usr/local/sbin/pi-power-pin" | sudo tee /etc/sudoers.d/pi-power-pin >/dev/null && sudo chmod 0440 /etc/sudoers.d/pi-power-pin && sudo visudo -cf /etc/sudoers.d/pi-power-pin'
ssh maccpi@192.168.0.243 'sudo -n /usr/local/sbin/pi-power-pin unpin && sudo -n /usr/local/sbin/pi-power-pin fan auto && sudo -n /usr/local/sbin/pi-power-pin status'
```

The helper commands: `pin <MHz>`, `unpin`, `fan full`, `fan auto`, `fan off` (the fan trip points
move to 85-88 C, so the fan stays off until the temperature where `run.py` aborts), `restore`,
`status`.

**5.2 `perf` on the Pi.** It needs no root to run, only to install:

```bash
ssh -t maccpi@192.168.0.243 'sudo apt install linux-perf'     # also upgraded the kernel to 6.18.60; reboot after
ssh maccpi@192.168.0.243 'perf --version; cat /proc/sys/kernel/perf_event_paranoid'    # paranoid stays 2
```

Leave `perf_event_paranoid` at 2. Per-process counting of the run's own user-space code is what v2
records. (Counting every core system-wide would need the value lowered to 1 with a root `sysctl`;
that was not needed and is not done.) The Pi hard-locks occasionally under load, so after the
reboot wait for `ssh maccpi@192.168.0.243 true` to answer.

**5.3 Wi-Fi off.** The Pi's Wi-Fi profile is `stevens-iot` (the name is case-sensitive), with
autoconnect off, so it stays off across reboots. Turn it off and on with the connection
(`con down` / `con up stevens-iot`), never with `nmcli radio wifi off`:

```bash
ssh -t maccpi@192.168.0.243 'sudo nmcli con down stevens-iot'
ssh maccpi@192.168.0.243 'cat /sys/class/net/wlan0/operstate'    # "down"
# only when something must be downloaded (weights, packages): bring it up, then down again
ssh -t maccpi@192.168.0.243 'sudo nmcli con up stevens-iot'
ssh -t maccpi@192.168.0.243 'sudo nmcli con down stevens-iot'
```

**5.4 Weights, pictures and models on the Pi.** `*.pt`, `inputs/*` and `models/` are not in git.
Build them on the NUC ([../tools/README.md](../tools/README.md)), then copy:

```bash
cd ~/Desktop/power-measurement-pi/tools && ~/.local/bin/uv sync
~/.local/bin/uv run python coco_subset.py                   # pictures: ../inputs/, subsets: tools/data/
~/.local/bin/uv run python quantize_yolo.py --model ../yolo11n.pt    # ../models/yolo11n_int8s.pt
~/.local/bin/uv run python quantize_yolo.py --model ../yolo11s.pt
cd ..
rsync -a inputs models maccpi@192.168.0.243:Desktop/power-measurement-pi/
rsync -a yolo11n.pt yolo11s.pt yolo11m.pt yolov8n.pt yolov8s.pt yolov8m.pt maccpi@192.168.0.243:Desktop/power-measurement-pi/
```

Pruned models (`prune_yolo.py`) are only needed with `--include-prune`.

**5.5 Monsoon output at 5.40 V.** The recorder command in
[README section 4.7](../README.md#47-start-the-recorder) already says `--vout 5.40`. Changing a
running recorder from 5.25 V means the full procedure of
[README section 8](../README.md#8-power-safety-turning-the-pi-on-and-off-rewiring) (halt the Pi,
stop the recorder, switch the output off, start the recorder). Never do it while a measurement
runs.

**5.6 Same code on the Pi and the EliteBook** (README 12.2; the orchestrator checks), then:

```bash
ssh macc2026@192.168.0.11 'cd ~/power-measurement-pi && ./lab check'
```

Everything must be `[ok]`: recorder, recorder stall, pinning helper, helper version, pi wifi, perf.

## 6. Phase 0 probes

Before any matrix time is spent, one lab session of single cells answers a go / no-go question for
each risky family. A no-go cuts or redesigns that family; it never becomes a silent assumption.
Record every outcome (including the cut families) for the Phase 4 report.

Rules for the session: `./lab check` all `[ok]` first, Wi-Fi off, nothing else on the Pi, the Pi
below 50 C at the start of each run (`./lab` waits). Run everything from `~/power-measurement-pi` on
the EliteBook. Each run leaves its folder in `runs/lab/<name>/`; the lab summary and these two
commands give the facts the questions below ask for:

```bash
cd ~/power-measurement-pi/runs/lab/<name>
python3 -c "import json; m=json.load(open('run_meta.json')); print({k: m.get(k) for k in ('variant','executed_dtype','quantized_modules','latency_ms_per_frame','max_temp_C','frames_done','dataset_version','holdout')}); print(m['perf_counters'])"
python3 -c "import json; print(json.dumps(json.load(open('derived.json')), indent=1)); a=json.load(open('alignment.json')); print('align', a['pass'], a['max_residual_ms'], a['n_edges'])"
```

| # | Probe | Command(s) | Go / no-go question | Known so far |
|---|---|---|---|---|
| 1 | YOLO m-model latency | `./lab run yolo11m_640_t4 yolov8m_640_t4` and, for comparison, `./lab run yolo11s_640_t4 yolov8s_640_t4`. For an l-model (needs `yolo11l.pt` in the Pi project folder; fetch it with the Wi-Fi connection up, section 5.3): `./lab run --task image --model yolo11l.pt --check-image bus.jpg --frames 5 --warmup 2 --bench 0 --serve 0` | Do the s and m cells fit the frame budget: the measured `forward` phase at most about 90 s (`latency_ms_per_frame` x frames: 20 for m, 30 for s)? And the whole cell well inside the 40-minute limit? Does an l-model confirm its exclusion? | not run |
| 2 | FP16 viability | `./lab run yolo11n_fp16_640_t4`, and `./lab run yolo11n` for the FP32 baseline | Does FP16 really execute in float16 (`executed_dtype` is `float16`; the run fails otherwise), and what does it cost in latency and energy? | executes in float16, but slower than FP32 (informal check: about 553 ms against about 330 ms). Stays opt-in (`--include-fp16`). The FP32 baseline cell of 2026-11-20 (5.35 V, no throttling): 294 ms and 2952 mJ per frame |
| 3 | INT8 static end to end | `./lab run yolo11n_int8s_640_t4 yolo11s_int8s_640_t4` | Does the calibrated model run end to end with `quantized_modules` 80, a `map_reference` and an acceptable mAP **measured on the Pi** (compare with the FP32 entry)? | runs end to end on the Pi at 5.35 V (2026-11-20 smoke): align 0.147 ms, verify ok, counter coverage 1.0, 207 ms and 1952 mJ per frame against 294 ms and 2952 mJ for FP32, 80 quantized modules; mAP so far only measured on the NUC (provisional, e.g. yolo11n 0.407 against 0.425 FP32 at 640); the Pi mAP is Task 16 |
| 4 | Pruning smoke test | (cut) `./lab run yolo11n_p30_640_t4` would show it | Does a pruned yolo11n load and run? | **No-go.** It loads, but mAP is about 0 and its detection check on `bus.jpg` finds nothing, so the cells are cut (section 4) |
| 5 | perf availability and overhead | `./lab check` (the `perf` line); on the Pi `perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,stalled-cycles-backend,LLC-load-misses,task-clock,context-switches,page-faults true`; then the A/B below | Do the Cortex-A76 events exist, run 100 % of the time (`min_pct_running` 100), and is the overhead small (the spec asks for under 0.1-0.2 W at 10 Hz; otherwise try 5 Hz)? | events exist and fit without multiplexing; counter coverage was 1.0 in all three 2026-11-20 smoke cells. Overhead: one A/B only, under bad conditions (section 4); **repeat it** |
| 6 | Fan-off thermal, supervised | `./lab run yolo11n_fanoff_640_t4 --yes-supervised`, while you stand at the Pi; watch it with `ssh maccpi@192.168.0.243 'sudo -n /usr/local/sbin/pi-power-pin status'` | How fast does the temperature climb (`sysfs.csv`, `temp_C`; `max_temp_C`), and does the 85 C abort trigger cleanly (exit code 3, `thermal_abort.json`, stage `thermal_abort`, fan back to auto afterwards)? | not run on the Pi; the abort logic is covered by unit tests |
| 7 | DVFS alignment and 1 Hz logging | `./lab run yolo11n_640_t4_dvfs` | Do the alignment residuals stay under 1 ms (the clock is pinned only around the sync pulses)? Does `sysfs.csv` have about one row per second, and do the measured phases show the clock moving inside 1500-2400 MHz? | the 2026-11-20 smoke cell passed: align max residual 0.154 ms, verify ok, counter coverage 1.0, 2885 mJ and 293 ms per frame; cpu0 sat at 1500 MHz idle and 2400 MHz under load. Still to confirm in Phase 0: repeat residuals and the 1 Hz log over a longer cell |
| 8 | Concurrency | `./lab run yolo11n_concspin_640_t4 yolo11n_conc2_640_t4` | One sync envelope (12 edges, residual under 1 ms), tagged step events (`tag`, `step_m1` / `step_spin` in the trace), both workloads completed iterations and overlapped, counter coverage at least 95 %, no `concurrent` check failing in `verify` | built and unit-tested; no Monsoon cell has been run yet |

**The perf A/B for probe 5.** `./lab run yolo11n` is the v1 command line with counters on (the
default). For the same cell with counters off, print the exact Pi command with
`./lab run yolo11n --dry-run`, add `--perf-counters off`, and run it by hand with README
section 5.3 (align and verify as shown there). Compare the `forward` mean power (`step_P_W`,
`mean_P_W` in the verify output) and `latency_ms_per_frame` between the two. Repeat each a few times
so the difference is bigger than the spread between repeats. If the overhead is larger than the spec
allows, `run.py --perf-interval-ms 200` samples at 5 Hz; the matrix does not pass that flag, so
switching it needs a one-line change in `host/matrix_v2.py`.

## 7. Phase 1 and Phase 2

**Phase 1 (harness).** The code for everything in this document is on the branch
`dataset-v2-harness`: `run.py` flags, counters and the thermal watchdog, the variants, the NUC
tools, the host merge and checks, the orchestrator pre-checks, the v2 families, concurrency, and the
`./lab` presets and checks. This page and the README / INSTRUCTIONS / tools updates are the last
documentation step.

Phase 1's exit criterion, a 3-cell smoke of the new families that aligns under 1 ms and passes
`verify_cell.py` including counter coverage, **was met on 2026-11-20** with the Monsoon at 5.35 V
(all `throttled` 0x0, no under-voltage events):

| Cell | Max residual | verify | Counter coverage | Per frame |
|---|---|---|---|---|
| yolo11n FP32, pinned 2400 MHz | 0.163 ms | ok | 1.0 | 2952 mJ, 294 ms |
| yolo11n INT8 static | 0.147 ms | ok | 1.0 | 1952 mJ, 207 ms |
| yolo11n DVFS (`ondemand`) | 0.154 ms | ok | 1.0 | 2885 mJ, 293 ms |

In the same session the FP32 cell drew 9.29 W in `forward` and 5.01 W at baseline, against 9.61 W
and 5.15 W for the matching v1 cell. That difference mixes kernel 6.18.60, 5.35 V and `perf` being
on; the `ref` family will quantify it.

Still to do in Phase 1: measure mAP on the Pi (plan Task 16), the final review and merge, and
deploying the branch to the Pi and the EliteBook (same commit on both, README 12.2; the
orchestrator checks), then `./lab test --host`.

**Phase 2 (the pilot).** The pilot runs every configuration group once, to catch naming, timeout,
disk and verification problems at about 30 cells instead of 176:

```bash
cd ~/power-measurement-pi/host
~/.local/bin/uv run python experiment.py plan --v2 pilot --out ../runs/v2_pilot/manifest.jsonl
systemd-run --user --unit=exp-v2-pilot --same-dir --collect \
  ~/.local/bin/uv run python experiment.py run --manifest ../runs/v2_pilot/manifest.jsonl \
  --max-attempts 3 --auto-power-cycle 40
```

It has 31 cells (29 with `--no-concurrent`), ids starting with `pilot_`: the four s / m models,
yolo11s at 2000 and 1500 MHz, the empty and crowded pictures, the INT8 cells, image sizes 320 and 960
(yolo11m at 960 too), batch 2 and 8, 1 and 2 threads, DVFS (two repeats), fan `auto` and `off`,
serving at 1 and 15 requests per second, the stream, the two concurrency cells, a **5-minute soak**
(not 30), and one `ref` cell per v1 model. Watch the first fan-off cell in person. Use
`--no-concurrent` for the first pilot if the concurrency probe (7 above) has not been run.

What to check afterwards, before planning Phase 3:

- **Timings.** Time per cell, especially the m-models at 960 px and at 1500 MHz, which must be well
  under the 40-minute default (the per-layer benchmarks alone are estimated at 6-15 minutes).
  Re-estimate Phase 3 from these.
- **Disk and memory.** EliteBook free space; and the peak memory of `align` + `verify` on the
  5-minute soak. The EliteBook has 7 GB shared with the recorder, so know the 5-minute figure before
  scheduling the 30-minute soaks (which have 6 times more samples).
  Soak cells log no per-layer begin/end events (`layer_events: false`; step events and labels
  stay), and `verify_cell.py` reads only the columns it uses, as categoricals. On a synthetic
  9-million-row, 38-column trace its peak memory is 2.3 GB (it was 6.4 GB, measured on the NUC with
  the same pandas/pyarrow). `align` is not covered by that figure.
- **Verification.** Every pilot cell passes `verify_cell.py`, including counter coverage of at least
  95 %. The first concurrent cell has tagged events, one sync envelope and counter coverage.
- **Naming and manifests.** Cell ids and holdout tags are as expected (`primary-s`, `secondary-m`).
- **The `ref` cells** have `dataset_version` 1 and counters off.

**Phase 3 (the full matrices, split over several days)** follows the spec: 3a the YOLO models (A)
and shapes and threading (C), 3b precision (B), 3c real-world behaviour (D) with the supervised-first
fan-off cells and the two soaks. Between runs, check the recorder's age and `n_total`, and the
EliteBook's free disk. The `ref` family (18 cells) is needed before any v1-versus-v2 comparison.

## 8. Reading a v2 cell

```python
import json, pandas as pd
cell = "runs/v2/yolo11s_640_t4_2400_r00"
meta = json.load(open(f"{cell}/run_meta.json"))
print(meta["dataset_version"], meta["holdout"], meta["variant"], meta["perf_counters"]["enabled"])

df = pd.read_parquet(f"{cell}/power_trace.parquet")
df = df[~df.ctx_read & (df.phase == "forward")]
print(df.groupby("step")[["P_W", "sys_temp_C"]].mean())
# ctr_* is the count in the whole 100 ms interval around a sample: use ratios, or one row per interval
df = df.assign(ipc=df.ctr_instructions / df.ctr_cycles)
print(df.groupby("step").ipc.mean())

print(json.load(open(f"{cell}/derived.json"))["energy_per_frame_mJ"])
```

To build a training set from many cells, read each cell's `run_meta.json` and keep the rows that fit:
`dataset_version == 2` and `perf_counters["enabled"]` for counter features, and
`holdout` not `primary-s` (and not `secondary-m` when that is the second test set) for training.
Drop `ctx_read` rows. For FP16 and INT8 cells, do not use `bytes_moved`, `macs` or
`arith_intensity` from `layer_table.csv`.
