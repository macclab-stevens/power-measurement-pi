# `./lab`: run Pi power measurements and tests from the EliteBook

`./lab` is one command that does everything for a single measurement:

1. Checks that the lab is ready.
2. Runs an ML workload (inference or training) on the Raspberry Pi 5 while the Monsoon measures
   its power.
3. Copies the results back, lines them up with the power recording, and checks them.
4. Prints a plain-English summary.

It also runs the unit tests on the Pi.

| What | Path |
|---|---|
| The command you type | `~/power-measurement-pi/lab` on the EliteBook (`~/Desktop/power-measurement-pi/lab` in the NUC/Pi checkouts) |
| The script behind it | `host/lab.py` |
| This file | `INSTRUCTIONS.md` (project root) |
| Results | `~/power-measurement-pi/runs/lab/` on the EliteBook |

> **Safe by design.** `./lab` never switches the Pi's power, never stops the Monsoon recorder,
> and never kills a run. When something goes wrong it tells you what happened and what to do.

---

## 1. Before you start (once)

`./lab` runs **on the EliteBook** (`macc2026@192.168.0.11`), the laptop connected to the Monsoon.
It expects the lab to be set up as in the README, section 4:

- **Recorder:** the Monsoon recorder is running (`curl -s localhost:8051/status` shows
  `"recording": true`).
- **SSH:** the EliteBook can log into the Pi without a password (`ssh maccpi@192.168.0.243 true`).
- **uv:** `~/.local/bin/uv` exists, and `host/` has its environment (`cd host && ~/.local/bin/uv sync`).
- **Pinning helper:** `sudo -n /usr/local/sbin/pi-power-pin status` works on the Pi.
- **Same code:** the Pi and the EliteBook are on the same git commit (README 12.2).

Log in and go to the project folder:

```bash
ssh macc2026@192.168.0.11
cd ~/power-measurement-pi
```

Every command below is typed there. (From `host/` you can also use
`~/.local/bin/uv run python lab.py ...`; it is the same thing.)

---

## 2. Quick start

```bash
./lab check            # 1. is everything ready?
./lab presets          # 2. what can I run?
./lab run yolo11n      # 3. measure one workload (about 6 minutes)
./lab test pr1         # 4. run the 3 Pi-side tests from the PR #1 review
```

---

## 3. The four commands

### 3.1 `./lab check`: is everything ready?

Read-only: it changes nothing on any machine.

```bash
./lab check
./lab check --allow-git-mismatch
```

What it checks:

| Check | Fails when |
|---|---|
| recorder | not reachable, not recording, or sampling below 4,900 Hz |
| recorder age | (warning) sampling for more than 8.5 days; the Monsoon library stops at about 9.9 days |
| pi reachable | no SSH answer. It then also shows the ping result and the Pi's current draw (about 0.3 A means it is frozen or halted) |
| pi temperature | (warning) 50 °C or warmer; runs then wait for it to cool |
| pi throttling | an under-voltage or throttle **right now**. A flag from earlier since boot (`0x50000`) is only a warning |
| pinning helper | `pi-power-pin` missing or not allowed |
| code version | the Pi and the EliteBook are on different commits |
| idle | a run, a test or the matrix is already running |

| Option | Meaning |
|---|---|
| `--allow-git-mismatch` | Report different git commits as a warning, not a failure. |

Example output:

```
Pre-flight checks
  [ok]   recorder        recording, 4992 Hz, serial 36517, .../host/data/20261007T110402
  [WARN] recorder age    sampling for 9.1 days
                         -> the Monsoon library stops at ~9.9 days; restart it the safe way, README 8; this script won't
  [ok]   pi reachable    maccpi@192.168.0.243
  [ok]   pi temperature  47.2 C (gate 50 C)
  [WARN] pi throttling   throttled=0x50000: happened earlier since boot (old news)
  [ok]   pinning helper  governor=ondemand ... fan_state=1/4
  [ok]   code version    fa97080 on both machines
  [ok]   idle            no matrix, no run.py or pytest on the Pi
Ready (2 warning(s)).
```

### 3.2 `./lab presets`: what can I run?

```bash
./lab presets
```

Lists the 14 ready-made workloads (the same ones as the experiment matrix), with their settings
and typical run time:

| Preset | What it is | Typical time |
|---|---|---|
| `yolov8n`, `yolo11n` | YOLO object detection, 640 px, 60 frames | 5-6 min |
| `effb0`, `mbv3`, `resnet18` | image classification (EfficientNet-B0, MobileNet-V3-small, ResNet-18), 224 px | 2-6.5 min |
| `lm16`, `lm64`, `lm128` | small language model (MiniGPT), prompt length 16/64/128, prefill + decode | under 1 min |
| `train_mbv3_sgd`, `train_mbv3_adam` | MobileNet-V3-small training, batch 8, 128 px | under 1 min |
| `train_resnet18_sgd`, `train_resnet18_adam` | ResNet-18 training, batch 8, 128 px | about 1 min |
| `train_lm64_sgd`, `train_lm64_adam` | MiniGPT training, sequence 64, batch 8 | under 1 min |

The longer inference times come from the per-layer benchmarks (`bench:*` phases). Use
`--bench 0` to skip them for a quick look.

### 3.3 `./lab run`: measure one or more workloads

```bash
./lab run PRESET [PRESET ...] [options]      # ready-made workloads
./lab run --task TASK --model MODEL [options] # your own workload
```

With no options, a preset runs **exactly** like a matrix cell at 2400 MHz, so your result can be
compared with the dataset directly. Any option you type overrides that value for every preset
given.

For each workload, `./lab run`:

1. Runs the pre-flight checks. Any `[FAIL]` stops it.
2. Waits until the Pi is below 50 °C.
3. Pins the CPU clock and runs the fan at full speed.
4. Runs the workload on the Pi.
5. Copies the data back, aligns it with the Monsoon recording, and checks it.
6. Prints a summary.

At the end it puts the Pi's clock and fan back to normal, unless `--keep-pinned` is given or
something is still running on the Pi.

#### Workload options

Only the options you type override the preset.

| Option | Default | Meaning |
|---|---|---|
| `PRESET ...` | none | One or more preset names (see `./lab presets`), run one after another. Leave out to describe your own workload with `--task` and `--model`. |
| `--task {image,lm,train}` | from preset | `image` = image inference (detection or classification), `lm` = language-model inference (prefill + decode), `train` = training steps. Needed together with `--model` when no preset is given. |
| `--model NAME` | from preset | **image:** `yolo11n.pt`, `yolov8n.pt` or `torchvision:<name>` (e.g. `torchvision:resnet50`). **lm:** `minigpt`, `minigpt-explicit` or `hf:<name>`. **train:** `torchvision:<name>` or `minigpt-explicit`. `*.pt` weights missing on the Pi, and `hf:` models, are downloaded from the internet on first use. The `hf:` download happens inside the measured `load_weights` phase. |
| `--frames N` | preset, else 30 | Measured iterations per mode (for training: optimizer steps). More gives steadier averages but a longer run. |
| `--warmup N` | preset, else 10 | Unmeasured warm-up passes before measuring. |
| `--threads N` | 4 | PyTorch CPU threads (the Pi 5 has 4 cores). |
| `--imgsz PX` | preset, else 640 | Input image size in pixels (image tasks and image training). |
| `--seq N` | preset, else 64 | Prompt length in tokens (language models). |
| `--batch N` | preset, else 8 | Training batch size, `--task train` only, at least 2. |
| `--optimizer {sgd,adam}` | preset, else sgd | Training optimizer: SGD (momentum 0.9) or Adam. |
| `--serve N` | presets 10, training 0 | Inference only: after the measured modes, N single requests with idle gaps between them (phase `serve`). 0 = none. |
| `--serve-gap-ms MS` | 200 | Idle time between serve requests. |
| `--bench SECONDS` | presets 1-2, training none | Seconds per isolated layer benchmark (`bench:*` phases, gives per-layer-type power). `0` = no benchmarks. |
| `--check-image PATH` | YOLO presets: `bus.jpg` | Detection sanity check: this image (relative to `~/Desktop/power-measurement-pi` on the Pi) must give a detection with confidence of at least `--conf`, or the run fails. |
| `--no-check-image` | off | Drop the preset's detection sanity check. |
| `--conf X` | 0.25 | Confidence threshold for `--check-image`. |
| `--live-camera` | off | Use real Pi-camera frames instead of a fixed test image. |

#### Conditions and output

| Option | Default | Meaning |
|---|---|---|
| `--cpu-mhz MHZ` | 2400 | Lock all 4 cores at this clock: 1500 to 2400 in steps of 100. The matrix uses 2400, 2000 and 1500. The fan always runs at full speed. |
| `--name NAME` | `<preset or model>_<MHz>_<YYYYmmdd-HHMMSS>` | Run folder name (one preset only). Letters, digits, `.`, `_`, `-`. An existing name is refused, so nothing is ever overwritten. |

#### Safety and flow

| Option | Default | Meaning |
|---|---|---|
| `--timeout MIN` | 40 | Give up waiting for the run after this many minutes. |
| `--no-thermal-gate` | off | Start immediately even if the Pi is warm. By default each run waits (up to 10 min) until the Pi is below 50 °C, as in the matrix. |
| `--keep-pinned` | off | Leave the clock and fan pinned afterwards. |
| `--allow-git-mismatch` | off | Run even if the Pi and the EliteBook are on different commits (both are recorded). |
| `--dry-run` | off | Only print the commands and folders. Contacts no machine. |

#### Examples

```bash
./lab run yolo11n                                     # YOLO11n inference, 2400 MHz
./lab run yolo11n train_lm64_adam train_resnet18_sgd  # 3 workloads, one after another
./lab run train_resnet18_sgd --cpu-mhz 1500           # training at a lower clock
./lab run mbv3 --cpu-mhz 2000 --frames 100
./lab run lm64 --frames 200 --name lm64_quick
./lab run yolo11n --frames 5 --serve 2 --bench 0      # quick look, about a minute
./lab run --task image --model torchvision:resnet50 --imgsz 224 --frames 50
./lab run --task train --model torchvision:resnet18 --batch 4 --imgsz 128 --optimizer adam
./lab run resnet18 --dry-run                          # show what would run
```

For many repeats or clock sweeps, use the matrix (`host/experiment.py plan --only <preset>`,
README 5.4) instead.

#### Watching a run while it goes

`./lab run` prints a line like this at the start of each workload:

```
watch live: ssh maccpi@192.168.0.243 tail -f Desktop/power-measurement-pi/runs/lab/<name>.log
```

Paste it into a second terminal to follow the Pi's output. Closing that window never affects the
run.

#### What the summary means

A successful training run looks like this:

```
 RUN OK  train_lm64_adam_2400_20261016-125443    train, preset train_lm64_adam
 Workload     minigpt-explicit, 200 optimizer steps per mode
 Conditions   CPU pinned 2400 MHz, fan full, Pi 47.2 C at start, git fa97080 (EliteBook) / fa97080 (Pi)
 Duration     28 s total (run.py 15 s)

 Mean power (W)            above idle
   idle (baseline)     5.03            (A 4.94 / B 5.12)
   train               9.30          +4.27
 Per step (W)  backward 10.26   dataload 6.36   forward 9.75   loss 9.75   optimizer_step 8.14   zero_grad 6.31
 Training loss 16.560 -> 0.422 over 206 steps
 Energy        77.6 J in the aligned trace
 Alignment     12 edges, max residual 0.14 ms (limit 1.0 ms), drift 0.6 ppm
               step coverage 90% of the measured modes
 Files         .../runs/lab/train_lm64_adam_2400_20261016-125443/
```

How to read each line:

- **idle (baseline):** the Pi's power doing nothing, measured before (A) and after (B) the work.
- **Mean power / above idle:** average power while the model ran, and how much of that the work
  itself added.
- **Per step:** power inside each part of an iteration. Inference steps are preprocess, forward
  and postprocess. Training steps are dataload, zero_grad, forward, loss, backward and
  optimizer_step. `profiler` is the one-off PyTorch profiling of the first iteration.
- **Training loss:** should go down. That shows the model really trained.
- **Alignment:** how precisely the power recording was lined up with the Pi's timeline. Anything
  under 1 ms passes; 0.1-0.3 ms is typical.
- **Slip warning** (when shown): the Monsoon lost a few milliseconds of samples. This is a known,
  accepted limitation; averages are unaffected.
- **Step coverage:** the share of measured time that carries a step label (below 50% fails).

#### Where the results go

| Where | What |
|---|---|
| `runs/lab/<name>/power_trace.parquet` | every power sample (5,000 per second) with its phase, step and layer label |
| `runs/lab/<name>/phase_energy.csv` | energy and average power per phase, iteration and step |
| `runs/lab/<name>/run_meta.json`, `alignment.json` | run settings, and how well the alignment worked |
| `runs/lab/<name>.log`, `<name>.dmesg.log` | the Pi's console output and kernel messages during the run |
| `runs/lab/results.jsonl` | one line per run (status, timings, summary numbers) |

All paths are under `~/power-measurement-pi/` on the EliteBook. The Pi keeps its own copy in
`~/Desktop/power-measurement-pi/runs/lab/<name>/`.

Load a trace in Python (from `~/power-measurement-pi`):

```bash
~/.local/bin/uv run --project host python -c "import pandas as pd; \
df = pd.read_parquet('runs/lab/<name>/power_trace.parquet'); \
print(df[~df.ctx_read].groupby('step').P_W.mean())"
```

(`ctx_read` marks moments when the Pi was reading its own sensors; drop them for clean numbers.)

### 3.4 `./lab test`: run the unit tests

```bash
./lab test [TARGET ...] [options]
```

| Target / option | Meaning |
|---|---|
| *(nothing)* or `all` | The whole Pi test suite (`tests/`). |
| `pr1` | The 3 Pi-side tests added in the PR #1 review: sensor snapshots are marked in the trace, top-k works with fewer than 5 classes, and live camera frames go through preprocessing. |
| `tests/<file>.py` or `tests/<file>.py::<test>` | One file or one test. Other pytest options such as `-x` are passed through. Paths must stay inside the project. |
| `-k EXPR` | Only tests whose name matches EXPR. |
| `--host` | Also run the EliteBook's own test suite (`host/tests`). |
| `--host-only` | Only the EliteBook suite; don't contact the Pi. |
| `--timeout MIN` | Give up on the Pi tests after this many minutes (default 20). |

Examples:

```bash
./lab test pr1                      # the 3 review tests (a few seconds)
./lab test                          # whole Pi suite
./lab test tests/test_pinning.py -x
./lab test -k topk
./lab test --host                   # Pi suite, then the EliteBook suite
./lab test --host-only
```

Notes:

- **Refused while busy.** Tests are refused while a run, other tests or the matrix are running:
  they load the CPU, and the test that emits sync pulses would disturb a measurement.
- **Heat.** The tests heat the Pi, so the next run may wait a few minutes to cool down.
- **The known failure.** `tests/test_phases.py::test_no_idle_contamination` needs an old data
  folder (`runs/fix_abab_smoke`) that is usually missing. When it is missing, the full suite
  skips that test automatically and says so. This is expected.
- **Saved output.** Output is saved to `runs/lab/tests/<time>_pi.log` (and `_host.log`).

---

## 4. Exit codes (for scripts)

| Code | Meaning |
|---|---|
| 0 | Everything OK (warnings allowed). |
| 1 | A check, run or test failed. The failed stage and what to do are printed. |
| 2 | Bad command line (unknown preset, invalid option or name). |
| 130 | Interrupted with Ctrl-C. |

---

## 5. When something goes wrong

`./lab run` names the stage that failed and explains it. The common ones:

| Failed at | What it means | What to do |
|---|---|---|
| a pre-flight `[FAIL]` | Something isn't ready. | Follow the hint printed under the check. |
| `thermal` | The Pi didn't cool below 50 °C within 10 minutes. | Check the fan, or use `--no-thermal-gate`. |
| `run` | The workload itself failed on the Pi. The last lines of its log are shown. | Usually under-voltage (cable), a bad option, or a model that can't load. |
| `align` | The power recording couldn't be lined up, almost always because the Monsoon lost samples during a sync pulse. | Run it again. |
| `verify` | The data failed a quality check (listed). | Read the listed errors; rerun if unsure. |
| `pi_crash` / `pi_down` | The Pi stopped answering (a known, rare Pi lockup). | Wait 3 minutes (it often reboots itself), then `./lab check`. If it stays down and draws about 0.3 A, follow README section 8. `./lab` never switches power. |

Other situations:

- **Ctrl-C** stops waiting, but the run on the Pi keeps going and its data is still saved there.
  `./lab` tells you whether it is still running and how to stop it
  (`ssh maccpi@192.168.0.243 pkill -INT -f src/run.py`). It never kills it for you.
- **"recorder unreachable":** you are probably not on the EliteBook, or the recorder is not
  running (README 4.7).
- **"recorder age" warning:** the Monsoon library stops after about 9.9 days of sampling. Restart
  the recorder the safe way (README section 8: shut the Pi down, stop the recorder, switch the
  output off, start the recorder). `./lab` won't do this for you, because it means powering the
  Pi off and on.

---

## 6. More

- The whole setup, explained: `README.md`.
- How a run works internally, and the meaning of every output column: README sections 6, 10
  and 11.
- The full 224-run experiment matrix: README section 5.4 (`host/experiment.py`).
