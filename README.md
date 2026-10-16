# Raspberry Pi 5 ML Power Measurement

Measure how much electrical power a Raspberry Pi 5 uses while it runs machine-learning models,
and label every power reading with what the model was doing at that instant.

The result is a dataset like this: *"at 12:07:03.1234 the Pi drew 10.3 W, and it was running the
`backward` step of training iteration 17 of resnet18, inside layer `layer3.1.conv2`"*. That dataset
is meant for training a model that predicts power from what the hardware is doing.

> **New here?** Read sections 1 to 3 to understand the setup, section 4 to set it up, and
> section 5 to run it. Section 9 lists what to do when something breaks.

---

## Contents

1. [What this project does (in plain words)](#1-what-this-project-does-in-plain-words)
2. [Words you will see everywhere](#2-words-you-will-see-everywhere)
3. [The three computers and the power meter](#3-the-three-computers-and-the-power-meter)
4. [One-time setup](#4-one-time-setup)
5. [Everyday use](#5-everyday-use)
   ([checklist](#51-pre-flight-checklist), [the quick way: `./lab`](#52-the-quick-way-lab),
   [by hand](#53-measure-one-model-by-hand), [the matrix](#54-run-the-whole-experiment-matrix))
6. [What the output files mean](#6-what-the-output-files-mean)
7. [Results so far](#7-results-so-far)
8. [Power safety: turning the Pi on and off, rewiring](#8-power-safety-turning-the-pi-on-and-off-rewiring)
9. [Troubleshooting](#9-troubleshooting)
10. [How it works under the hood](#10-how-it-works-under-the-hood)
11. [Reference](#11-reference)
12. [Developing: tests, deploying code, git](#12-developing-tests-deploying-code-git)
13. [Project layout and further docs](#13-project-layout-and-further-docs)

---

## 1. What this project does (in plain words)

A Raspberry Pi 5 runs ML workloads: image models (YOLO object detectors, image classifiers),
small language models, and training loops. We want to know **how much power each part of that
work costs**.

The Pi cannot measure itself accurately, so an external lab instrument, a **Monsoon HVPM power
monitor**, powers the Pi and measures the current and voltage going into it 5,000 times per
second. A laptop (the "EliteBook") talks to the Monsoon over USB and records everything.

The hard part is matching the two timelines: the Monsoon knows *how much power* was used at each
moment, and the Pi knows *what it was doing* at each moment. To line them up:

1. The Pi writes down a timestamp every time it starts or ends something (a phase, a step, a
   layer). That is the **event log**.
2. At the start and end of every run, the Pi deliberately burns all 4 CPU cores in a fixed on/off
   rhythm (the **sync pulse**). This shows up as a clear square wave in the power trace.
3. Afterwards, a script (`host/align.py`) finds that square wave in the Monsoon data and uses it
   to line up the two clocks to well under a millisecond. Then every power sample gets the label
   of whatever the Pi was doing at that instant.

```
        USB-C power (5.25 V)                    USB (data)
 Pi 5  <====================  Monsoon HVPM  -------------------->  EliteBook
 runs the models               measures V and I                   records 5 kHz power,
 writes events.csv             5,000 times/s                      aligns, checks, stores
   ^                                                                   |
   |                        SSH (commands, copy results)               |
   +-------------------------------------------------------------------+
```

A whole **experiment** is many such runs (called **cells**): different models, different CPU
clock speeds, repeated several times. The EliteBook runs them one after another automatically.

---

## 2. Words you will see everywhere

| Word | Meaning |
|---|---|
| **Monsoon / HVPM** | The power monitor (Monsoon High Voltage Power Monitor, serial 36517). It is the Pi's power supply *and* the meter. |
| **Recorder** | `host/recorder.py`, a program on the EliteBook that records the Monsoon non-stop and answers questions about it over HTTP (port 8051). |
| **Run** | One execution of `src/run.py` on the Pi: load a model, measure it, write files. |
| **Cell** | One run as part of the experiment matrix, e.g. `yolo11n_2400_r03` = YOLO11n at 2400 MHz, repeat 3. |
| **Matrix** | The full list of cells (here 224). |
| **Phase** | A big chunk of a run: `load_weights`, `warmup`, `forward`, `train`, `serve`, `bench:<op>`, ... |
| **Step** | A piece of one iteration: `preprocess`, `forward`, `postprocess`, or for training `dataload`, `zero_grad`, `forward`, `loss`, `backward`, `optimizer_step`. |
| **Layer** | One module of the neural network (e.g. `model.3.conv`). Every layer call is logged too. |
| **Sync pulse** | The deliberate on/off CPU burst used to line up the clocks. |
| **Alignment** | Matching the Pi timeline to the Monsoon timeline (`host/align.py`). |
| **Residual** | How far off the alignment is at each sync edge. Must be below 1 ms, typically 0.1-0.3 ms. |
| **Slip** | The Monsoon occasionally loses a few milliseconds of samples without saying so. Alignment detects and flags these. Accepted as a known limitation. |
| **Pinning** | Locking the CPU clock to a fixed speed (1500, 2000 or 2400 MHz) and the fan at full speed, so neither changes during a run and confuses the power numbers. |
| **Baseline** | The power the Pi uses while idle (about 4.9 W at 2400 MHz with the fan at full speed). |

---

## 3. The three computers and the power meter

| Machine | Address / login | What it does |
|---|---|---|
| **Pi 5** (8 GB) | `maccpi@192.168.0.243`, project in `~/Desktop/power-measurement-pi` | Runs the ML workloads (`src/run.py`). |
| **EliteBook** (laptop) | `macc2026@192.168.0.11`, project in `~/power-measurement-pi` | Monsoon host: runs the recorder, the experiment orchestrator, alignment and checks. All results end up here. |
| **NUC** (desktop) | `192.168.0.205`, project in `~/Desktop/power-measurement-pi` | Development machine: where code is edited, tested, committed and sent to the other two. |
| **Monsoon HVPM** | USB to the EliteBook | Powers the Pi through USB-C at 5.25 V (6 A limit) and measures it. |
| **Lab time server** | `192.168.0.20` (GPS/PTP) | Keeps the Pi and EliteBook clocks in sync (both run `timemaster`). |

**Why two separate computers for the Pi and the Monsoon?** If the Pi recorded its own power, the
recording itself would use CPU and pollute the measurement. Keeping the recorder on another machine
means the Pi does nothing but the workload.

---

## 4. One-time setup

You only do this once (or after reinstalling a machine). If the lab is already set up, skip to
[section 5](#5-everyday-use).

### 4.1 Wiring

1. **Remove the Pi's normal power supply.** The Monsoon replaces it.
2. Connect the **Monsoon main-channel output** to the Pi's **USB-C power input**. Use a short, thick
   cable: the resistance of this path matters. At about 3 A peak, a 0.25-0.33 ohm cable drops the
   Pi below its under-voltage threshold (about 4.63 V), and the run is rejected. The current wire
   measures about 0.11 ohm. **Check polarity with a multimeter before switching on.**
3. Connect the **Monsoon USB port** to the EliteBook.
4. Power the Monsoon itself as its manual describes.

The Pi stays powered as long as the Monsoon output is on. Read [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring)
before you ever unplug or rewire anything.

### 4.2 SSH keys (so machines can talk without passwords)

The orchestrator on the EliteBook logs into the Pi automatically, so it needs a key. Run these
yourself (they ask for the password once):

```bash
# on the NUC
ssh-copy-id maccpi@192.168.0.243
ssh-copy-id macc2026@192.168.0.11
# on the EliteBook
ssh-copy-id maccpi@192.168.0.243
```

Test: `ssh maccpi@192.168.0.243 true` must return without asking for a password.
Never put passwords into scripts or this repo.

### 4.3 Get the code onto all three machines

```bash
git clone git@github.com:macclab-stevens/power-measurement-pi.git
cd power-measurement-pi && git checkout monsoon-experiment
```

Paths used throughout: `~/Desktop/power-measurement-pi` on the Pi and the NUC,
`~/power-measurement-pi` on the EliteBook. (If a machine cannot reach GitHub, see
[section 12.2](#122-deploying-code-to-the-pi-and-the-elitebook).)

### 4.4 Install `uv` (the Python environment tool) on the Pi and the EliteBook

[`uv`](https://docs.astral.sh/uv/) creates the Python environment with the exact package versions
from `uv.lock`, so every machine measures with the same software.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs to ~/.local/bin/uv
```

### 4.5 Pi setup

```bash
cd ~/Desktop/power-measurement-pi
~/.local/bin/uv sync --frozen        # installs torch, torchvision, ultralytics, opencv (large: ~5.6 GB)
```

Then install the **pinning helper**. Changing the CPU clock and fan needs root, and the experiment
runs unattended, so a tiny root script is allowed to run without a password. It accepts only four
fixed commands (`pin <MHz>`, `fan full`, `restore`, `status`). One sudo prompt:

```bash
ssh -t maccpi@192.168.0.243 'sudo install -o root -g root -m 0755 ~/Desktop/power-measurement-pi/tools/pi-power-pin /usr/local/sbin/pi-power-pin && echo "maccpi ALL=(root) NOPASSWD: /usr/local/sbin/pi-power-pin" | sudo tee /etc/sudoers.d/pi-power-pin >/dev/null && sudo chmod 0440 /etc/sudoers.d/pi-power-pin && sudo visudo -cf /etc/sudoers.d/pi-power-pin'
```

Check it: `sudo -n /usr/local/sbin/pi-power-pin status` prints the governor, clock, fan state and
temperature.

Check the clock sync: `systemctl is-active timemaster` should print `active`.

### 4.6 EliteBook setup

1. Let normal users access the Monsoon over USB (udev rule, needs sudo once):

   ```bash
   echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2ab9", ATTR{idProduct}=="0001", MODE="0664", GROUP="plugdev"' \
     | sudo tee /etc/udev/rules.d/99-monsoon.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

   Your user must be in the `plugdev` group (`groups` shows it). Check that `lsusb` lists a
   device with ID `2ab9:0001`.

2. Install the host environment (the `host/` folder has its own small environment: Monsoon
   library, numpy, pandas, pyarrow):

   ```bash
   cd ~/power-measurement-pi/host && ~/.local/bin/uv sync
   ```

3. Start the recorder (next section). Check clock sync: `systemctl is-active timemaster`.

### 4.7 Start the recorder

The recorder must run **all the time**. It switches the Monsoon output on (which powers and
boots the Pi) and records non-stop into `host/data/<timestamp>/`.

```bash
cd ~/power-measurement-pi/host
systemd-run --user --unit=monsoon-recorder --same-dir --collect \
  ~/.local/bin/uv run python recorder.py --vout 5.25 --current-limit 6
```

`systemd-run` keeps it running in the background after you log out. Check it:

```bash
curl -s localhost:8051/status
# {"recording": true, "serial": 36517, "vout_V": 5.25, "sample_hz": 4992.4, ...}
```

`sample_hz` is about 4,992 (the Monsoon's real rate, slightly under the nominal 5,000).

The Monsoon library stops after about 9.9 days of continuous sampling. Restart the recorder
before then, following [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring).

---

## 5. Everyday use

### 5.1 Pre-flight checklist

Run these before any measurement. All of them must look right. `./lab check`
([section 5.2](#52-the-quick-way-lab)) does all of this, and more, in one go.

```bash
# EliteBook: recorder running and recording?
ssh macc2026@192.168.0.11 'curl -s localhost:8051/status'          # "recording": true

# Pi: reachable, no power problems, cool?
ssh maccpi@192.168.0.243 'vcgencmd get_throttled; vcgencmd measure_temp'
#   throttled=0x0 is ideal. 0x50000 means an under-voltage happened earlier since boot (old news).
#   A low bit set (0x1, 0x4, ...) means a problem right now.

# Same code version on both?
ssh maccpi@192.168.0.243 'git -C Desktop/power-measurement-pi log --oneline -1'
ssh macc2026@192.168.0.11 'git -C power-measurement-pi log --oneline -1'
```

### 5.2 The quick way: `./lab`

`./lab` (the script `host/lab.py`) does a whole single measurement in one command: pre-flight
checks, run.py on the Pi, copy back, align, verify, and a plain-English summary. Run it on the
**EliteBook**, from `~/power-measurement-pi` (the recorder runs there). Every command has a
detailed `--help`.

```bash
cd ~/power-measurement-pi
./lab check                       # is everything ready? read-only, changes nothing
./lab presets                     # the 14 ready-made workloads (the matrix models)
./lab run yolo11n                 # one preset, exactly like a matrix cell at 2400 MHz
./lab run yolo11n train_lm64_adam train_resnet18_sgd    # the 3-cell smoke test, in sequence
./lab run train_resnet18_sgd --cpu-mhz 1500 --frames 10 # override any value
./lab run --task image --model torchvision:resnet50 --imgsz 224 --frames 50   # your own
./lab run resnet18 --dry-run      # only show the commands; contacts no machine
./lab test pr1                    # the 3 tests from the PR #1 review, on the Pi
./lab test --host                 # the whole Pi suite, then the host suite here
```

- **What `check` looks at:** the recorder (recording at about 4,992 Hz, not near its ~9.9-day
  limit), the Pi (reachable, temperature, throttling, the pinning helper), the same git commit on
  both machines, and that nothing else is running (run.py or pytest on the Pi, the matrix here).
  If the Pi does not answer, it shows the Monsoon current and what to do
  ([section 9](#9-troubleshooting)).
- **`run` always** adds `--sampler monsoon --fan-full --cpu-mhz <MHz>` (default 2400). It refuses
  to start if any check fails. It waits until the Pi is below 50 °C, as the matrix does
  (`--no-thermal-gate` to skip). It prints a line to follow the Pi's output live:
  `ssh maccpi@192.168.0.243 tail -f Desktop/power-measurement-pi/runs/lab/<name>.log`.
- **Where files go:** `runs/lab/<name>/` on both machines, plus `<name>.log` (Pi console) and
  `<name>.dmesg.log` (Pi kernel log) next to it on the EliteBook. Every run adds one line to
  `runs/lab/results.jsonl`. The default `<name>` is `<preset or model>_<MHz>_<date-time>`;
  `--name` picks your own, and an existing name is refused. Test output goes to
  `runs/lab/tests/`.
- **Afterwards** the Pi's clock and fan are put back to normal, unless `--keep-pinned` is given
  or run.py is still running on the Pi.
- **Ctrl-C** stops waiting and tells you whether run.py is still running on the Pi and how to
  stop it (`ssh maccpi@192.168.0.243 pkill -INT -f src/run.py`). It never kills it for you.
- **It never switches power** and never stops the recorder. After a Pi lockup it tells you what
  to do ([section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring) and
  [section 9](#9-troubleshooting)).
- **Exit codes:** 0 all ok, 1 a check, run or test failed (the failed stage is printed with an
  explanation), 2 bad command line, 130 interrupted.

`./lab` is a two-line wrapper for `~/.local/bin/uv run --project host python host/lab.py`, so
the host environment (numpy, pandas) is used. For many repeats or clock sweeps use the matrix
([section 5.4](#54-run-the-whole-experiment-matrix)).

### 5.3 Measure one model by hand

This is what `./lab run` does for you, step by step. On the **Pi**:

```bash
cd ~/Desktop/power-measurement-pi
~/.local/bin/uv run --frozen python src/run.py \
    --sampler monsoon --cpu-mhz 2400 --fan-full \
    --task image --model yolo11n.pt --check-image bus.jpg \
    --frames 30 --serve 10 \
    --out runs/my_test > runs/my_test.log 2>&1
tail runs/my_test.log          # ends with "[ok] dataset written to runs/my_test"
```

What the flags mean:

- `--sampler monsoon`: use the Monsoon, not the Pi's built-in meter.
- `--cpu-mhz 2400 --fan-full`: pin the clock and fan. Always pass both for real data.
- `--task image --model yolo11n.pt`: which workload. See [section 11.1](#111-srcrunpy-flags) for
  every option, including `--task lm` (language model) and `--task train`.
- `--out runs/my_test`: where the results go.

> **Never pipe the output into `head`** (`... | head`). When `head` exits, the run is killed
> mid-measurement. Always redirect into a file as shown.

Then, on the **EliteBook**, copy the run over and align it:

```bash
cd ~/power-measurement-pi
rsync -a maccpi@192.168.0.243:Desktop/power-measurement-pi/runs/my_test runs/
cd host
REC=$(curl -s localhost:8051/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')
~/.local/bin/uv run python align.py --recorder-dir "$REC" --pi-run ../runs/my_test
~/.local/bin/uv run python verify_cell.py ../runs/my_test --cpu-mhz 2400
```

`align.py` prints `[ok] edges=12 max_residual=0.17 ms ...` when it worked. `verify_cell.py`
prints a summary and `"ok": true`.

### 5.4 Run the whole experiment matrix

The matrix (defined in `host/matrix.py`) is 14 workloads × 3 clock speeds:

- **Inference:** yolov8n, yolo11n, efficientnet_b0, mobilenet_v3_small, resnet18, and MiniGPT with
  prompt lengths 16, 64 and 128.
- **Training:** mobilenet_v3_small, resnet18 and MiniGPT (prompt length 64), each with SGD and
  with Adam.
- **Repeats:** each runs 10 times at 2400 MHz and 3 times each at 1500 and 2000 MHz.

That makes 224 cells and about 9 hours. The order is shuffled within each round so that slow
drifts (temperature, time of day) don't line up with one model.

All on the **EliteBook**, from `~/power-measurement-pi/host`:

```bash
# 1. Make the to-do list (a "manifest"). Use a new folder name for each experiment.
~/.local/bin/uv run python experiment.py plan --out ../runs/matrix/manifest.jsonl
#    Try it small first:  --smoke (3 cells)  or  --probe (every model once)

# 2. Start it in the background (survives logging out)
systemd-run --user --unit=exp-matrix --same-dir --collect \
  ~/.local/bin/uv run python experiment.py run --manifest ../runs/matrix/manifest.jsonl \
  --max-attempts 3 --auto-power-cycle 40

# 3. Check on it whenever you like
~/.local/bin/uv run python experiment.py status --manifest ../runs/matrix/manifest.jsonl
tail -f ../runs/matrix/experiment.log
```

The `status` output looks like this:

```
cells 224  ok 150  failed 0  not run 74
mean ok cell 2.2 min; remaining ~2.7 h
ok cells with slip_warning: 83
```

For every cell, the orchestrator does these steps:

1. Checks that the recorder is recording and the Pi is reachable.
2. Waits until the Pi is below 50 °C, so every cell starts from a similar temperature.
3. Runs `src/run.py` on the Pi over SSH, and streams the Pi's kernel log (`dmesg`) to a file.
4. Copies the results back (`rsync`).
5. Aligns them (`align.py`) and checks them (`verify_cell.py`).
6. Writes one line to `results.jsonl`, then goes to the next cell.

**It is safe to stop and restart.** Run the same `run` command again: cells already marked `ok`
are skipped. A failed cell is retried at the end of its round, up to `--max-attempts` times.

To stop it: `systemctl --user stop exp-matrix`.

**`--auto-power-cycle 40`** handles the Pi lockups described in [section 9](#9-troubleshooting).
If the Pi stops answering and the Monsoon confirms it is really halted, the orchestrator
switches the power off and on to reboot it, at most 40 times. Each cycle is logged to
`power_cycles.jsonl`. Leave it at `0` if you don't want the experiment to touch the power.

---

## 6. What the output files mean

Each cell ends up in `~/power-measurement-pi/runs/<experiment>/<cell_id>/` on the EliteBook.

### 6.1 The files you will actually use

| File | What it is | Typical use |
|---|---|---|
| `power_trace.parquet` | **The main dataset.** One row per Monsoon sample (5,000 per second) with the label of what the Pi was doing. | Training the power predictor, plotting. |
| `phase_energy.csv` | Energy and average power per phase, iteration and step. | Quick comparisons ("how much energy does backward cost?"). |
| `run_meta.json` | Everything about the run: model, settings, clock, temperatures, software versions, training loss curve. | Knowing what the cell was. |
| `alignment.json` | How well the clocks were lined up, and whether samples were lost. | Data-quality filtering. |

### 6.2 `power_trace.parquet` columns

| Column | Meaning |
|---|---|
| `t_s` | Seconds since the run started. |
| `t_pi_ns` | The sample's time on the Pi's (lab-synced) clock, in nanoseconds. |
| `I_A`, `V_V`, `P_W` | Current (A), voltage (V) and power (W) going into the Pi. |
| `phase` | The big chunk: `load_weights`, `first:<mode>`, `warmup`, `sync:start`, `baseline_A`, `forward` / `prefill` / `decode` / `train`, `serve`, `baseline_B`, `bench:<config>`, `sync:end`, `gap`, `idle`. |
| `mode`, `iter` | Inside a measured phase: which mode and which iteration number. Empty elsewhere. |
| `step` | The sub-part of the iteration (see the table below), or empty. |
| `layer_path` | The innermost network layer running at that instant, or empty. |
| `depth` | How deeply nested that layer is. |
| `ctx_read` | `True` while the Pi was reading its own sensors (`vcgencmd`), which briefly costs a little power. Drop these rows for clean numbers. |

Steps per task:

| Task | Steps in each iteration |
|---|---|
| Image inference | `preprocess` (resize/letterbox the image, normalize) → `forward` → `postprocess` (NMS for detectors, top-5 for classifiers) |
| Language model | `preprocess` (token ids) → `forward` → `postprocess` (pick the next token) |
| Training | `dataload` → `zero_grad` → `forward` → `loss` → `backward` → `optimizer_step` |
| `serve` phase | the task's steps, then `idle_between_requests` |

Load one in Python:

```python
import pandas as pd
df = pd.read_parquet("runs/matrix/train_resnet18_sgd_2400_r00/power_trace.parquet")
df = df[~df.ctx_read]                                   # drop sensor-read moments
print(df[df.phase == "train"].groupby("step").P_W.mean())
# roughly: backward 10.6, dataload 8.7, forward 11.2, loss 9.2, optimizer_step 8.9, zero_grad 8.2
```

### 6.3 Experiment-level files

These are in the experiment folder (e.g. `runs/matrix/`):

| File | What it is |
|---|---|
| `manifest.jsonl` | The to-do list: one cell per line, with its exact `run.py` arguments. |
| `results.jsonl` | One line per attempt: status, which stage failed, timings, alignment and check summaries, git commit. |
| `experiment.log` | Human-readable progress log. |
| `<cell>.log` | The Pi's console output for that cell. |
| `<cell>.dmesg.log` | The Pi's kernel messages during that cell (evidence if the Pi crashes). |
| `power_cycles.jsonl` | When the orchestrator had to power-cycle the Pi (only if it happened). |

### 6.4 Other files in a cell folder

- `events.csv`: the Pi's raw timeline, one row per event. It is the input to `align.py`.
- `layer_table.csv`: per-layer timing, shapes and compute cost (MACs, bytes moved).
- `perop_power.csv`: power of each layer type measured in isolation (the `bench:*` phases).
  Training cells skip this (`--no-bench`).
- `op_table.csv`: PyTorch operator timings.
- `context.csv`: temperature, clock and throttle flags every 5 s.

Column-by-column details are in [section 11](#11-reference).

---

## 7. Results so far

Full matrix, 2026-10-07: **224 / 224 cells passed** (4 cells needed a retry because of a
Monsoon sample loss during a sync pulse; no Pi lockups during the matrix). Worst alignment error
in any kept cell: 0.36 ms. Data: 4.4 GB in `~/power-measurement-pi/runs/matrix/` on the EliteBook.

Average power (W), ± spread across repeats:

| Workload | 2400 MHz | 2000 MHz | 1500 MHz |
|---|---|---|---|
| Idle baseline (fan at full speed) | 4.9 | 4.5 | 4.0 |
| resnet18 inference | 10.70 ± 0.06 | 8.73 | 6.24 |
| yolov8n / yolo11n inference | 9.94 / 9.58 | 8.26 / 7.96 | 5.95 / 5.88 |
| efficientnet_b0 / mobilenet_v3_small inference | 9.26 / 8.97 | 7.79 / 7.46 | 5.77 / 5.59 |
| MiniGPT prefill / decode | 7.6-8.1 / 8.6 | 6.5-6.9 / 7.1 | 5.0-5.2 / 5.3 |
| resnet18 training (SGD / Adam) | 10.55 / 10.38 | 8.77 / 8.65 | 6.32 / 6.28 |
| mobilenet_v3 and MiniGPT training | 9.4-9.6 | 7.8-8.0 | 5.7-5.8 |

What stands out:

- **Repeats agree** within about ±0.1 W, so the occasional sample loss does not distort averages.
- **Clock speed matters a lot.** Going from 2400 to 1500 MHz cuts the power above idle by roughly
  2.5 to 4 W.
- **Training steps differ.** `forward` and `backward` cost the most (about 9.6-11.2 W at
  2400 MHz). `zero_grad`, `dataload` and the SGD optimizer step cost the least (about 6.5-8.7 W).
- **For detectors, pre- and post-processing** use 1.5-2 W less than the network itself.

---

## 8. Power safety: turning the Pi on and off, rewiring

The Monsoon is the Pi's only power source. Some of its behaviour is surprising:

- **Stopping the recorder does NOT turn the Pi off.** The output stays on ("keep-on"), so
  restarting the recorder never cuts the Pi's power by accident.
- **A Pi that has been shut down (`sudo poweroff`) stays off** until its power is switched off and
  on again. Restarting the recorder alone does not do that.
- **The only program that switches the output off is `host/vout_off.py`**, and it refuses to run
  without `--confirm-pi-halted`.

**To rewire, change the voltage, or restart the recorder cleanly**, do exactly this, in order:

1. Shut the Pi down cleanly: `ssh -t maccpi@192.168.0.243 sudo poweroff`. Wait until it stops
   answering `ping 192.168.0.243`.
2. Stop the recorder (the output stays on): `systemctl --user stop monsoon-recorder`.
3. Switch the output off: `cd ~/power-measurement-pi/host && ~/.local/bin/uv run python vout_off.py --confirm-pi-halted`
4. Rewire if needed, and check polarity with a multimeter.
5. Start the recorder ([section 4.7](#47-start-the-recorder)). The output comes on and the Pi boots.

Every recorder start creates a new folder in `host/data/`; old recordings are kept.

---

## 9. Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| Pi does not answer `ping`/`ssh`, and the Monsoon shows it drawing only about 0.28-0.34 A (how to check: end of this section) | **Pi lockup** (see below). | Wait 3 minutes: sometimes it reboots by itself. If not, follow [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring) steps 2, 3 and 5 (it is already halted). The matrix orchestrator does this itself with `--auto-power-cycle`. |
| Cell failed at `align` with "residual >= 1.0 ms" | The Monsoon lost samples right inside a sync pulse, so the clocks could not be matched. | Nothing. The cell is retried automatically. It happened 4 times in 228 attempts. |
| `slip_warning` in results | The Monsoon lost a few ms of samples during the run (about once a minute). Labels after that point are shifted by a few ms. | Accepted. Repeats average it out. Filter on `alignment.json` → `slip_samples_total` if you need perfect labels. |
| Run fails with "under-voltage occurred during run" | The Pi's input voltage dipped below about 4.63 V. | Check the power cable (short and thick, about 0.1 ohm). Heavy workloads above about 3 A are at risk. |
| Run fails with "baseline drift" or clock errors | The Pi changed speed, or something else ran during the measurement. | Make sure nothing else runs on the Pi. Check `pi-power-pin status`. |
| `Monsoon recorder unreachable` | The recorder is not running. | Start it ([section 4.7](#47-start-the-recorder)). |
| The run dies silently | Its output was piped into `head`. | Redirect output into a file instead. |
| `git mismatch: host ... vs Pi ...` when starting the matrix | The Pi and the EliteBook run different code versions. | Update both ([section 12.2](#122-deploying-code-to-the-pi-and-the-elitebook)). |

**Pi lockups (open issue).** On 2026-10-07 the Pi froze 4 times in about 70 minutes of load. All
four were before the matrix started; there were none during the 9-hour matrix.

- **What it looks like:** the Pi's current suddenly drops from about 0.9 A to 0.34 A within 10 ms.
  The supply voltage stays steady and the kernel logs nothing.
- **What happens next:** twice it rebooted itself after a few seconds. The Pi firmware recorded a
  watchdog reset: `/proc/device-tree/chosen/bootloader/rsts` read `0x20` or `0x1020`. Twice it
  stayed frozen until its power was switched off and on.
- **Possible link:** two of the four happened while running mobilenet_v3_small.
- **Not reproduced:** a tight loop on the layer that was running at one lockup, and thousands of
  sensor reads under load, both ran clean.
- **Suspected cause:** the board or its firmware.

To capture evidence next time, turn on a persistent system log on the Pi:
`sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald`.

To see the Pi's current draw yourself, run this on the EliteBook from `host/`:

```bash
~/.local/bin/uv run python -c "
import json, urllib.request, numpy as np; from store import open_samples
d = json.load(urllib.request.urlopen('http://localhost:8051/status'))['run_dir']
s = open_samples(d); I = np.array(s[len(s)-25000:len(s)])[:, 0]
print(f'last 5 s: mean {I.mean():.3f} A, max {I.max():.3f} A')"
```

About 0.9-1 A is idle with the fan at full speed; about 0.28 A means the Pi is frozen or halted.

---

## 10. How it works under the hood

### 10.1 One run, start to finish (`src/run.py`)

1. **Pin** the CPU clock and fan (`tools/pi-power-pin`), and record their state.
2. **`load_weights`**: build or load the model. This is measured too.
3. **`first:<mode>`**: the very first inference or training step. It is slow (memory allocation,
   kernel setup), so it is measured separately.
4. **`warmup`**: a few more passes, so temperature and caches settle.
5. **Sync pulse (start)**, then **`baseline_A`**: 2 s of idle to measure the idle power.
6. **Measured modes** (`forward`, or `prefill` + `decode`, or `train`): N iterations, each split
   into steps, with every layer call logged.
7. **`serve`** (inference only): single requests with 200 ms idle gaps, to see how power behaves
   when going between busy and idle.
8. **`baseline_B`**: 2 s idle again. The idle power is the mean of A and B; if they differ by more
   than 0.5 W the run fails.
9. **`bench:<config>`** (inference only): each distinct layer type is run alone in a tight loop
   for 1-2 s, giving its steady-state power.
10. **Sync pulse (end)**, write `events.csv`, `run_meta.json` and the other tables.
11. **Acceptance checks**: the clock stayed pinned, no throttling or under-voltage, the fan stayed
    at full, the model produced sane output. Any failure fails the run.

### 10.2 Lining up the clocks (`host/align.py`)

1. **Coarse:** both machines are synced to the lab GPS time server, and the recorder stamps every
   USB data block with the EliteBook time on arrival. That gives a first guess to within a few
   milliseconds.
2. **Fine:** the sync pulse pattern (all cores ON 150 ms, OFF 100, ON 300, OFF 100, ON 50) is
   searched for near the guessed position. Its 12 edges (6 at the start, 6 at the end) are matched
   to the Pi's logged edge times. A straight-line fit (offset plus drift) maps Pi time to Monsoon
   time.
3. **Check:** if any edge is more than 1 ms off the fit, the run fails. Typical is 0.1-0.3 ms.
4. **Label:** every sample gets the latest phase, the innermost step and the innermost layer
   active at that instant.
5. **Sample-loss detection:** a lost chunk of samples shows up as a permanent jump in the
   relationship between "sample number" and "arrival time". The detector reports those as `slips`.

### 10.3 The experiment orchestrator (`host/experiment.py`)

- `plan` expands `host/matrix.py` into a manifest.
- `run` works through it round by round; each attempt is appended to `results.jsonl`.
- **Failure stages:**
  - `recorder` and `pi_down` stop the whole experiment, because nothing can work until a human
    looks.
  - `pi_crash`, `run`, `rsync`, `align` and `verify` are retried.
- **Power-cycle guard:** it only switches the power off and on when all three are true:
  - the Pi has not answered for 3 minutes;
  - it does not answer `ping`;
  - the Monsoon shows halted-level current (mean below 0.4 A and max below 0.6 A over 5 s).
- **Last step:** at the very end it restores the Pi's normal clock and fan settings.

---

## 11. Reference

### 11.1 `src/run.py` flags

| Flag | Default | Meaning |
|---|---|---|
| `--task` | `image` | `image` (detection/classification), `lm` (language model), `train` |
| `--model` | `yolo11n.pt` | `yolo11n.pt`, `yolov8n.pt` (any ultralytics weights); `torchvision:<name>` (e.g. `torchvision:resnet18`); `minigpt`, `minigpt-explicit`; `hf:<name>` (Hugging Face causal LM) |
| `--sampler` | `pmic` | `monsoon` for real data; `pmic` is the old on-board meter |
| `--recorder-url` | `http://192.168.0.11:8051` | where the recorder is |
| `--cpu-mhz` | none | pin the clock: 1500-2400 in steps of 100 |
| `--fan-full` | off | fan at full speed for the whole run |
| `--frames` | 30 | measured iterations per mode |
| `--warmup` | 10 | warm-up passes |
| `--threads` | 4 | PyTorch threads |
| `--imgsz` | 640 | image size (image and image-training tasks) |
| `--seq` | 64 | prompt length (LM tasks) |
| `--batch` | 8 | training batch size (must be at least 2) |
| `--optimizer` | `sgd` | `sgd` (momentum 0.9, lr 0.01) or `adam` (lr 1e-3) |
| `--serve` | 0 | number of single requests in the `serve` phase |
| `--serve-gap-ms` | 200 | idle time between serve requests |
| `--bench` | 3.0 | seconds per isolated layer benchmark |
| `--no-bench` | off | skip the layer benchmarks |
| `--check-image` | none | detection sanity check: this image must give a detection with at least `--conf` confidence |
| `--conf` | 0.25 | detection confidence threshold |
| `--live-camera` | off | use real camera frames (Pi camera) instead of a fixed image |
| `--out` | `runs/<timestamp>` | output folder |

### 11.2 `host/experiment.py`

| Command | Options |
|---|---|
| `plan --out <manifest>` | `--smoke` (3 cells), `--probe` (each model once at 2400 MHz), `--only a,b` (only these model keys), `--seed N` (shuffle order) |
| `run --manifest <manifest>` | `--max-attempts N` (default 2), `--auto-power-cycle N` (default 0), `--results <file>`, `--allow-git-mismatch` |
| `status --manifest <manifest>` | `--results <file>` |

### 11.3 `events.csv` (Pi side)

Columns: `t_real_ns` (lab-synced wall clock), `t_mono_ns`, `kind`, `name`, `depth`, `mode`, `iter`,
`layer_path`, `config_id`, `window_id`.

| `kind` | Written when |
|---|---|
| `marker` | a phase starts (`name` = phase) |
| `step_begin` / `step_end` | a step starts or ends (`name` = step) |
| `begin` / `end` | a layer call starts or ends (`layer_path`, nesting `depth`) |
| `pulse_on` / `pulse_off` | a sync-pulse edge |
| `ctx_begin` / `ctx_end` | the Pi's sensor read starts or ends |

### 11.4 `alignment.json`

- **Pass/fail:** `pass` and `fail_reasons`. The only fatal check is `max_residual_ms` < 1.0.
- **Fit quality:** `n_edges`, `trains`, `max_residual_ms`, `rms_residual_ms`.
- **The fit itself:** `c0_ns` (offset, mostly USB latency), `c1` and `drift_ppm`.
- **Warnings:**
  - `slip_warning` and `slips` (each with `t_ns`, `step_ms`, `samples`) and `slip_samples_total`;
  - `drift_warning` (drift above 50 ppm);
  - `envelope_max_dev_ms` and `nan_fraction`.

### 11.5 `run_meta.json` (most useful keys)

- **What ran:** `task`, `model`, `modes`, `frames_per_mode`, `threads`, `imgsz`, `seq_len`,
  `batch`, `optimizer`.
- **Training:** `train_loss` (loss per training iteration).
- **Lifecycle timing:** `load_weights_s`, `first_call_ms`.
- **Idle power:** `baseline_W`, `baseline_A_W`, `baseline_B_W`, `baseline_drift_W`.
- **Pinning and power health:** `cpu_mhz_pinned`, `fan_full`, `pin_status_start`/`_end`,
  `throttled_start`/`_end`.
- **Recorder:** `recorder_status_start`/`_end`.
- **Software:** `torch_version`, `schema_version` (4 for Monsoon runs).

### 11.6 `phase_energy.csv`

One row per (`phase`, `iter`, `step`): `n_samples`, `mean_P_W`, `dur_ms`, `E_mJ` (energy in
millijoules = W × ms), `ctx_read_frac`, `t_start_s`, `t_end_s`.

### 11.7 Units

Numeric columns carry their unit in the name: `_W` watts, `_A` amps, `_V` volts, `_ms`
milliseconds, `_ns` nanoseconds, `_mJ` millijoules, `_C` °C, `_MHz`. Energy is always
`E_mJ = P_W × t_ms`.

---

## 12. Developing: tests, deploying code, git

### 12.1 Tests

The easy way, from `~/power-measurement-pi` on the EliteBook
([section 5.2](#52-the-quick-way-lab)):

```bash
./lab test                  # the whole Pi suite, over ssh
./lab test pr1              # only the 3 tests from the PR #1 review
./lab test tests/test_pinning.py -x
./lab test --host           # Pi suite, then the host suite here (--host-only: host suite only)
```

It refuses to start while a measurement or the matrix is running (the tests heat the Pi), and
saves the output to `runs/lab/tests/`. By hand:

```bash
# Pi-side code (needs torch; run on the Pi)
cd ~/Desktop/power-measurement-pi && ~/.local/bin/uv run --frozen --with pytest pytest tests/
# Host-side code (run on the EliteBook)
cd ~/power-measurement-pi/host && ~/.local/bin/uv run pytest
```

`tests/test_phases.py::test_no_idle_contamination` needs an old run folder
(`runs/fix_abab_smoke`) and fails where that folder is missing. That is expected; `./lab test`
deselects it in that case and says so.

### 12.2 Deploying code to the Pi and the EliteBook

Edit and commit on the NUC. Then either:

- **via GitHub:** `git push origin monsoon-experiment`, then `git pull` on the Pi and the EliteBook;
- **directly over SSH** (works without GitHub access on those machines), from the NUC:

  ```bash
  for h in macc2026@192.168.0.11:power-measurement-pi maccpi@192.168.0.243:Desktop/power-measurement-pi; do
    u=${h%%:*}; d=${h#*:}
    git push -f "ssh://$u/~/$d" monsoon-experiment:refs/heads/incoming
    ssh $u "cd $d && git merge --ff-only incoming && git log --oneline -1"
  done
  ```

Both machines must be on the same commit before a matrix starts (the orchestrator checks).

### 12.3 What is in git and what is not

- **Tracked:** code, tests, docs, `uv.lock`.
- **Not tracked** (see `.gitignore`):
  - `runs/` (measurement data, back it up separately);
  - `host/data/` (raw Monsoon recordings, about 0.15 GB per hour, 3.5 GB per day);
  - `.venv/`, model weights (`*.pt`), logs.

---

## 13. Project layout and further docs

```
src/                  Pi side
  run.py              one measurement run (entry point)
  tasks.py            workloads: ImageTask, TokenLMTask, TrainTask
  layers.py           per-layer hooks, timing, compute-cost features
  eventlog.py         the Pi's event log (phases, steps, layers)
  syncpulse.py        the sync-pulse emitter
  powersampler.py     Monsoon client (and the old PMIC sampler)
  microbench.py       isolated per-layer power benchmarks
  pinning.py          talks to tools/pi-power-pin
  sync_check.py       emit sync pulses only (alignment self-test)
lab                   one-command launcher (wrapper for host/lab.py)
host/                 EliteBook side
  lab.py              single runs and tests: check / run / test / presets
  recorder.py         always-on Monsoon recorder + HTTP API
  align.py            line up clocks, label every sample
  verify_cell.py      check one cell, write phase_energy.csv
  experiment.py       run the matrix (plan / run / status)
  matrix.py           which models, clocks, repeats
  vout_off.py         manual safety tool: switch the Monsoon output off
  store.py, clockfit.py   recording storage and clock fitting
tools/pi-power-pin    root helper for clock and fan pinning
tests/, host/tests/   unit tests
modeling/, trace_forecast/, analysis/   earlier analysis on the old PMIC data
docs/                 design docs, plans, logs
```

Further reading:

- Design spec: `docs/superpowers/specs/2026-10-06-monsoon-experiment-design.md`
- Measurement-plane bring-up log (what was measured and decided, SP1):
  `docs/superpowers/plans/sp1-bringup-log.md`
- Workloads and matrix plan (SP2 and SP3): `docs/superpowers/plans/2026-10-07-sp2-sp3-workloads-and-matrix.md`
- The old on-board PMIC harness (schema v3), `collect.py`, model-agnostic adapters:
  `docs/legacy-pmic-harness.md`
- **Next step (not built yet), SP4:** one index across all cells, for training the power
  predictor.
