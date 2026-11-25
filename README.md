# Raspberry Pi 5 ML Power Measurement

Measure how much electrical power a Raspberry Pi 5 uses while it runs machine-learning models,
and label every power reading with what the model was doing at that instant.

The result is a dataset like this: *"at 12:07:03.1234 the Pi drew 10.3 W, and it was running the
`backward` step of training iteration 17 of resnet18, inside layer `layer3.1.conv2`"*. That dataset
is meant for training a model that predicts power from what the hardware is doing.

> **New here?** Read sections 1 to 3 to understand the setup, section 4 to set it up, and
> section 5 to run it. Section 9 lists what to do when something breaks.
>
> **Dataset v2** (more YOLO models, precision variants, hardware counters, thermal and serving
> conditions) is described in [docs/dataset-v2.md](docs/dataset-v2.md). Where this README covers
> it, the sections say so.

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
        USB-C power (5.40 V)                    USB (data)
 Pi 5  <====================  Monsoon HVPM  -------------------->  EliteBook
 runs the models               measures V and I                   records 5 kHz power,
 writes events.csv             5,000 times/s                      aligns, checks, stores
   ^                                                                   |
   |                        SSH (commands, copy results)               |
   +-------------------------------------------------------------------+
```

A whole **experiment** is many such runs (called **cells**): different models, different CPU
clock speeds, repeated several times. The EliteBook runs them one after another automatically.

There are two datasets. **Dataset v1** (224 cells, October 2026) is the original matrix.
**Dataset v2** adds more YOLO models, INT8 models, other image sizes, thread counts and batch
sizes, free-running clocks, fan and serving conditions, and hardware performance counters on every
cell. [docs/dataset-v2.md](docs/dataset-v2.md) explains what is in it and what stays different
from v1.

---

## 2. Words you will see everywhere

| Word | Meaning |
|---|---|
| **Monsoon / HVPM** | The power monitor (Monsoon High Voltage Power Monitor, serial 36517). It is the Pi's power supply *and* the meter. |
| **Recorder** | `host/recorder.py`, a program on the EliteBook that records the Monsoon non-stop and answers questions about it over HTTP (port 8051). |
| **Run** | One execution of `src/run.py` on the Pi: load a model, measure it, write files. |
| **Cell** | One run as part of the experiment matrix, e.g. `yolo11n_2400_r03` = YOLO11n at 2400 MHz, repeat 3. |
| **Matrix** | The full list of cells: 224 for dataset v1, and the families of [section 5.4.2](#542-the-dataset-v2-matrix) for dataset v2. |
| **Dataset v1 / v2** | v1 = the 224-cell matrix of October 2026 (`dataset_version` 1 in `run_meta.json`). v2 = the expansion (`dataset_version` 2). Details: [docs/dataset-v2.md](docs/dataset-v2.md). |
| **Family** | A group of v2 cells that varies one thing: `A` YOLO models, `A-input` empty vs crowded picture, `B` INT8, `C` image size / batch / threads, `D` clock, fan, serving and concurrency conditions, `soak` 30-minute runs, `ref` the v1 cross-check. |
| **Holdout** | A tag in `run_meta.json` (`primary-s`, `secondary-m`, or none) that says which cells to keep out of predictor training. s-models are the primary holdout, m-models the secondary one. |
| **Counters** | Hardware performance counters (CPU cycles, instructions, cache misses, ...) recorded 10 times a second with `perf` on every v2 cell (`counters.csv`). |
| **Phase** | A big chunk of a run: `load_weights`, `warmup`, `forward`, `train`, `serve`, `bench:<op>`, ... |
| **Step** | A piece of one iteration: `preprocess`, `forward`, `postprocess`, or for training `dataload`, `zero_grad`, `forward`, `loss`, `backward`, `optimizer_step`. |
| **Layer** | One module of the neural network (e.g. `model.3.conv`). Every layer call is logged too. |
| **Sync pulse** | The deliberate on/off CPU burst used to line up the clocks. |
| **Alignment** | Matching the Pi timeline to the Monsoon timeline (`host/align.py`). |
| **Residual** | How far off the alignment is at each sync edge. Must be below 1 ms, typically 0.1-0.3 ms. |
| **Slip** | The Monsoon occasionally loses a few milliseconds of samples without saying so. Alignment detects and flags these. Accepted as a known limitation. |
| **Pinning** | Locking the CPU clock to a fixed speed (1500, 2000 or 2400 MHz) and the fan at full speed, so neither changes during a run and confuses the power numbers. Some v2 cells deliberately do not pin: **DVFS** cells let the clock float, and fan `auto` / `off` cells change the cooling. |
| **Baseline** | The power the Pi uses while idle (about 4.9 W at 2400 MHz with the fan at full speed). |

---

## 3. The three computers and the power meter

| Machine | Address / login | What it does |
|---|---|---|
| **Pi 5** (8 GB) | `maccpi@192.168.0.243`, project in `~/Desktop/power-measurement-pi` | Runs the ML workloads (`src/run.py`). Its Wi-Fi must be off while measuring ([section 4.8](#48-extra-setup-for-dataset-v2)). |
| **EliteBook** (laptop) | `macc2026@192.168.0.11`, project in `~/power-measurement-pi` | Monsoon host: runs the recorder, the experiment orchestrator, alignment and checks. All results end up here. |
| **NUC** (desktop) | `192.168.0.205`, project in `~/Desktop/power-measurement-pi` | Development machine: where code is edited, tested, committed and sent to the other two. For dataset v2 it also builds the INT8 weights and the COCO picture subsets (`tools/`, [tools/README.md](tools/README.md)). |
| **Monsoon HVPM** | USB to the EliteBook | Powers the Pi through USB-C at 5.40 V (6 A limit) and measures it. |
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
runs unattended, so a tiny root script is allowed to run without a password. It accepts only a few
fixed commands:

| Command | What it does |
|---|---|
| `pin <MHz>` | lock all 4 cores at that clock (1500 to 2400, steps of 100) |
| `unpin` | back to the `ondemand` governor, 1500-2400 MHz (used by the DVFS cells); fan untouched |
| `fan full` | fan at maximum |
| `fan auto` | the stock fan trip points (50 / 60 / 67.5 / 75 °C) |
| `fan off` | fan trip points moved up to 85-88 °C: the fan stays off for the whole run and only comes back as a backstop at the temperature where `run.py` aborts anyway |
| `restore` | `unpin` plus `fan auto` |
| `status` | print governor, clocks, fan state and temperature |

Dataset v2 needs this newer version (an older helper has no `unpin` and no `fan off`). Run the
same command again to upgrade it; `./lab check` shows `helper version` as a warning while the old
one is installed. One sudo prompt:

```bash
ssh -t maccpi@192.168.0.243 'sudo install -o root -g root -m 0755 ~/Desktop/power-measurement-pi/tools/pi-power-pin /usr/local/sbin/pi-power-pin && echo "maccpi ALL=(root) NOPASSWD: /usr/local/sbin/pi-power-pin" | sudo tee /etc/sudoers.d/pi-power-pin >/dev/null && sudo chmod 0440 /etc/sudoers.d/pi-power-pin && sudo visudo -cf /etc/sudoers.d/pi-power-pin'
```

Check it: `sudo -n /usr/local/sbin/pi-power-pin status` prints the governor, clock, fan state and
temperature. After an upgrade, `sudo -n /usr/local/sbin/pi-power-pin unpin` and
`sudo -n /usr/local/sbin/pi-power-pin fan auto` must also work without a password.

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
  ~/.local/bin/uv run python recorder.py --vout 5.40 --current-limit 6
```

`systemd-run` keeps it running in the background after you log out. Check it:

```bash
curl -s localhost:8051/status
# {"recording": true, "serial": 36517, "vout_V": 5.4, "sample_hz": 4992.4, ...}
```

`sample_hz` is about 4,992 (the Monsoon's real rate, slightly under the nominal 5,000).

The Monsoon library stops after about 9.9 days of continuous sampling (at 2^32 - 1 samples).
It stops **silently**: `/status` keeps saying `"recording": true` while `n_total` no longer
rises (seen on 2026-10-17). So the way to know it is alive is that `n_total` increases between two
`/status` reads. `./lab check` and the experiment orchestrator do exactly that, and refuse to
start a cell when `n_total` is not rising or is less than an hour from the ceiling. **Restart the
recorder at least every 9 days**, following [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring).

### 4.8 Extra setup for dataset v2

Dataset v2 needs a few more things. Short list here; the commands, and why each is needed, are in
[docs/dataset-v2.md, section 5](docs/dataset-v2.md#5-one-time-setup).

- **`perf` on the Pi** (the Debian package `linux-perf`), for the hardware counters. It runs without
  root: the default `perf_event_paranoid=2` allows counting the run's own user-space code, which is
  what v2 records. Installing it upgraded the Pi's kernel (6.18.50 to 6.18.60).
- **The newer `pi-power-pin`** (section 4.5).
- **Weights, picture inputs and model files on the Pi:** `*.pt` weights, `inputs/` and `models/`
  are not in git, so they are copied there ([tools/README.md](tools/README.md)).
- **Pi Wi-Fi off while measuring.** The radio adds power noise. `./lab check` warns when `wlan0` is up.
- **Monsoon output at 5.40 V** (the recorder command in section 4.7). At 5.25 V the Pi hit
  under-voltage under YOLO load (see [section 9](#9-troubleshooting)).

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
#   throttled=0x0 is ideal. 0x50000 means an under-voltage happened since boot: reboot the Pi
#   before a long matrix (run.py also watches the live alarm, sysfs uv_alarm, and the cell dmesg log).
#   A low bit set (0x1, 0x4, ...) means a problem right now.

# Same code version on both?
ssh maccpi@192.168.0.243 'git -C Desktop/power-measurement-pi log --oneline -1'
ssh macc2026@192.168.0.11 'git -C power-measurement-pi log --oneline -1'

# Pi: Wi-Fi off? (it must be "down" while measuring)
ssh maccpi@192.168.0.243 'cat /sys/class/net/wlan0/operstate'

# EliteBook: is the recorder really sampling? n_total must grow between two reads
ssh macc2026@192.168.0.11 'curl -s localhost:8051/status | grep -o "\"n_total\": [0-9]*"; sleep 2; curl -s localhost:8051/status | grep -o "\"n_total\": [0-9]*"'
```

### 5.2 The quick way: `./lab`

`./lab` (the script `host/lab.py`) does a whole single measurement in one command: pre-flight
checks, run.py on the Pi, copy back, align, verify, and a plain-English summary. Run it on the
**EliteBook**, from `~/power-measurement-pi` (the recorder runs there). Every command has a
detailed `--help`. Step-by-step instructions for every option: [INSTRUCTIONS.md](INSTRUCTIONS.md).

```bash
cd ~/power-measurement-pi
./lab check                       # is everything ready? read-only, changes nothing
./lab presets                     # the ready-made workloads: 14 from the v1 matrix, 48 dataset-v2 ones
./lab run yolo11n                 # one preset, exactly like a matrix cell at 2400 MHz
./lab run yolo11n_int8s_640_t4    # a dataset-v2 preset (INT8 yolo11n)
./lab run yolo11n_fanoff_640_t4 --yes-supervised   # fan off: watch it in person
./lab run yolo11n train_lm64_adam train_resnet18_sgd    # the 3-cell smoke test, in sequence
./lab run train_resnet18_sgd --cpu-mhz 1500 --frames 10 # override any value
./lab run --task image --model torchvision:resnet50 --imgsz 224 --frames 50   # your own
./lab run resnet18 --dry-run      # only show the commands; contacts no machine
./lab test pr1                    # the 3 tests from the PR #1 review, on the Pi
./lab test --host                 # the whole Pi suite, then the host suite here
```

- **What `check` looks at:** the recorder (recording at about 4,992 Hz, not near its ~9.9-day
  limit, and really sampling: `n_total` grows), the Pi (reachable, temperature, throttling, the
  pinning helper and whether it is the new version, Wi-Fi off, `perf` installed), the same git
  commit on both machines, and that nothing else is running (run.py or pytest on the Pi, the
  matrix here). If the Pi does not answer, it shows the Monsoon current and what to do
  ([section 9](#9-troubleshooting)). The check takes about 2 seconds longer than before because it
  reads `n_total` twice.
- **`run` always** adds `--sampler monsoon --fan-full --cpu-mhz <MHz>` (default 2400) to a v1
  preset or your own workload. A dataset-v2 preset already carries its own flags (including its
  fan mode and `--holdout`), so only `--cpu-mhz` and what you type change. `run` refuses to start
  if any check fails. It waits until the Pi is below 50 °C, as the matrix does
  (`--no-thermal-gate` to skip). It prints a line to follow the Pi's output live:
  `ssh maccpi@192.168.0.243 tail -f Desktop/power-measurement-pi/runs/lab/<name>.log`.
- **Cut presets** (INT8 dynamic `*_int8d_*`, pruned `*_p30_*` / `*_p50_*`) are not listed by
  `./lab presets` and `./lab run` refuses them unless you add `--include-cut`.
- **Fan-off presets need `--yes-supervised`.** The fan stays off until 85 °C, where `run.py`
  stops the run, so the first such run must be watched in person. Without the flag `./lab run`
  refuses (`--dry-run` still works).
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

Two matrices exist. The v1 matrix (`host/matrix.py`, section 5.4.1) is what produced the 224
cells of dataset v1 and is unchanged (its `run.py` command lines are the same; since `--perf-counters`
now defaults to `on`, a new cell planned from it also records counters, and stays `dataset_version` 1;
the 224 existing v1 cells never have counters). The dataset-v2 matrix (`host/matrix_v2.py`, section 5.4.2)
is the expansion.

#### 5.4.1 The v1 matrix

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

#### 5.4.2 The dataset v2 matrix

`host/matrix_v2.py` builds the v2 cells, grouped into families. Every cell is a YOLO detection
run (`--task image`, `bus.jpg` as the sanity image, `--sampler monsoon`, 4 threads, 640 px unless
the family says otherwise, `--perf-counters on`, a `--holdout` tag). Cells are shuffled within
each round, as in v1, and a cell is named
`{model}[_{variant}][_{condition}]_{imgsz}_t{threads}[_b{batch}]_{clock}_r{repeat}`, where `clock`
is the pinned MHz or `dvfs` (for example `yolo11n_int8s_640_t4_2400_r00`,
`yolo11n_640_t4_dvfs_r00`, `yolo11m_soak30_640_t4_2400_r00`). The `ref` cells are named
`v1ref_<key>_<mhz>_rNN` (for example `v1ref_yolo11n_2400_r00`).

| Family | What varies | Cells (default) |
|---|---|---|
| `A` | YOLO models yolov8s, yolov8m, yolo11s, yolo11m at 2400 MHz (5 repeats), 2000 and 1500 MHz (3 each); per-layer benchmarks on | 44 |
| `A-input` | yolo11n on a picture the detector finds nothing in (`inputs/empty.jpg`) and a crowded one (`inputs/crowded.jpg`), 3 repeats each | 6 |
| `B` | INT8 static yolo11n (2400, 2000, 1500 MHz) and yolo11s (2400 MHz), 3 repeats each. FP16, INT8 dynamic and pruned cells exist behind flags (below) | 12 |
| `C` | image size 320 / 480 / 960 (yolo11 n, s, m), batch 2 / 4 / 8, threads 1 / 2 (yolo11 n and s, and yolo11n at 320 and 960 px), 3 repeats each | 60 |
| `D` | free-running clock (DVFS) for yolo11n and yolo11s (5 repeats); fan `auto` and `off`; serving at 1, 5 and 15 requests per second; a stream of 50 different pictures; two models at once (`conc2`) and a model next to a matrix-multiply spinner (`concspin`); 3 repeats each | 34 |
| `soak` | 30 minutes of continuous inference, fan `full`, yolo11n and yolo11m, one run each | 2 |
| `ref` | the v1 cross-check: yolov8n and yolo11n with the **v1 command lines**, 1500 / 2000 / 2400 MHz, 3 repeats, counters **off**, no holdout (so they stay `dataset_version` 1) | 18 |
| `pilot` | one cell of every configuration group, a 5-minute soak and one `ref` cell per model; see [docs/dataset-v2.md](docs/dataset-v2.md#7-phase-1-and-phase-2) | 31 |

Per-layer benchmarks (`--bench 2`) run on repeat 0 of each `A`, `A-input` and `C` configuration only (repeats 1 and up get `--no-bench`), because per-op steady-state power is a per-configuration property while energy and latency per frame come from the trace of every repeat (Ruling R31, to cut matrix time).

All of the families together (`all`) are 176 cells (44 + 6 + 12 + 60 + 34 + 2 + 18). Some cells
are left out of that on purpose, and a flag adds them back:

| Flag (with `--v2`) | Adds | Cells |
|---|---|---|
| `--include-fp16` | `B`: yolo11n in FP16 | +3 |
| `--include-optional` | `C`: yolo11m with 1 and 2 threads | +6 |
| `--include-int8d` | `B`: INT8 dynamic (yolo11n, yolo11s, yolov8n). **Cut:** dynamic quantization only touches Linear layers, and YOLO has none, so it quantizes 0 modules. `run.py` fails such a run instead of labelling float32 data INT8 | +9 |
| `--include-prune` | `B`: pruned yolo11n (30 % and 50 %). **Cut:** pruning without fine-tuning destroys the model (mAP about 0, see [tools/README.md](tools/README.md)) | +6 |
| `--no-concurrent` | *removes* the 6 `D` concurrency cells | -6 |

On the EliteBook, from `~/power-measurement-pi/host`:

```bash
# one family, or "all", or "pilot" (a small first run)
~/.local/bin/uv run python experiment.py plan --v2 pilot --out ../runs/v2_pilot/manifest.jsonl
~/.local/bin/uv run python experiment.py plan --v2 C --out ../runs/v2_C/manifest.jsonl
~/.local/bin/uv run python experiment.py plan --v2 all --no-concurrent --out ../runs/v2/manifest.jsonl
```

Start, watch and stop it exactly like the v1 matrix (section 5.4.1: `experiment.py run`,
`status`, `systemctl --user stop`). `--v2` cannot be combined with `--smoke`, `--probe` or
`--only`. Use a new folder for every plan.

What the orchestrator now does for every cell, v1 and v2 (the soak and `timeout_s` items only
matter for v2):

- **Before every cell** it also checks that the recorder is really sampling (`n_total` grows, and
  is at least an hour clear of its ceiling) and that there is enough free disk: 10 GB on the
  EliteBook runs folder and 5 GB on the Pi project folder (2 GB more for soak cells). A failure
  there stops the whole experiment (stages `recorder` and `disk`; the Pi's `df` is retried once
  after 10 s). It also checks that no `run.py` is still running on the Pi (a leftover one would
  corrupt the cell): if one is, the experiment stops at stage `pi_busy`.
- **Per-cell time limit:** 40 minutes, except a cell that carries its own `timeout_s` (the 30-minute
  soaks get 50 minutes). On a timeout the orchestrator sends SIGINT to `run.py` on the Pi, waits up
  to 60 s for it to exit (so it restores the fan and stops `perf`), and ends the cell at stage
  `timeout` (the error says whether `run.py` exited; a cell that is retried meets `pi_busy` if it
  did not).
- **Under-voltage:** if the cell's `dmesg` log contains "Undervoltage detected", the cell fails at
  stage `undervoltage` (retried like other failures; the experiment goes on).
- **Thermal stop:** `run.py` aborts a run at 85 °C (exit code 3) and stops hard at 90 °C (exit
  code 4). The orchestrator marks that cell as failed at stage `thermal_abort`, shows the Pi's
  `thermal_abort.json`, does **not** retry it, and goes on with the next cell.
- **Extra checks per cell** in `verify_cell.py` (counter coverage, fan mode, DVFS clock range,
  thermal limit, per-workload checks for concurrency cells, mAP reference): see
  [section 6](#6-what-the-output-files-mean) and [docs/dataset-v2.md](docs/dataset-v2.md).
- **Fan `off` and the soaks:** watch the first fan-off cell in person (`./lab run
  yolo11n_fanoff_640_t4 --yes-supervised`), and see
  [docs/dataset-v2.md](docs/dataset-v2.md#6-phase-0-probes) before starting `D` or `soak`.

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
| `derived.json` | Energy and latency per frame, worked out on the EliteBook after alignment ([section 11.8](#118-dataset-v2-files)). | Comparing cells without redoing the arithmetic. |
| `counters.csv`, `sysfs.csv` | Hardware counters (10 Hz) and CPU clock / temperature / fan (1 Hz) from the Pi. Already merged into `power_trace.parquet`; the raw files stay in the cell folder. | Every dataset-version-2 cell, and any newer run (counters are on by default); never the 224 v1 cells. Features for the power predictor. |

### 6.2 `power_trace.parquet` columns

| Column | Meaning |
|---|---|
| `t_s` | Seconds since the run started. |
| `t_pi_ns` | The sample's time on the Pi's (lab-synced) clock, in nanoseconds. |
| `I_A`, `V_V`, `P_W` | Current (A), voltage (V) and power (W) going into the Pi. |
| `phase` | The big chunk: `load_weights`, `first:<mode>`, `warmup`, `sync:start`, `baseline_A`, `forward` / `prefill` / `decode` / `train`, `concurrent_drain`, `serve`, `baseline_B`, `bench:<config>`, `sync:end`, `gap`, `idle`. `concurrent_drain` only exists in `--concurrent` cells: the short moment after the main workload's last frame while the second workload finishes its current iteration. |
| `mode`, `iter` | Inside a measured phase: which mode and which iteration number. Empty elsewhere. |
| `step` | The sub-part of the iteration (see the table below), or empty. In a `--concurrent` cell: the main workload's step. |
| `layer_path` | The innermost network layer running at that instant, or empty. |
| `depth` | How deeply nested that layer is. |
| `ctx_read` | `True` while the Pi was reading its own sensors (`vcgencmd`), which briefly costs a little power. Drop these rows for clean numbers. |

A v2 cell has more columns, added by `align.py` from the Pi's own logs:

| Column | Meaning |
|---|---|
| `ctr_<event>` | The hardware counter `<event>` for the interval (100 ms by default) that contains this sample (`ctr_cycles`, `ctr_instructions`, `ctr_cache_references`, `ctr_cache_misses`, `ctr_branch_misses`, `ctr_stalled_cycles_backend`, `ctr_LLC_load_misses`, `ctr_task_clock`, `ctr_context_switches`, `ctr_page_faults`). Every sample in the same interval has the same value, so this is a count per interval, not per sample. Empty (NaN) outside every interval. |
| `pct_<event>` | The share of that interval (0-100) in which `<event>` was really being counted. Under 100 means the CPU had to share its counters between events and `perf` scaled the count up. The default event set was chosen so this stays at 100. |
| `sys_cpu0_kHz` ... `sys_cpu3_kHz` | Each core's clock, in kHz, from the latest 1 Hz reading (at most 2 s old). |
| `sys_temp_C`, `sys_fan_state` | SoC temperature and fan state (0 = off, up to 4 = full) from the same reading. |
| `sys_uv_alarm` | The Pi's live under-voltage alarm (hwmon `rpi_volt`, `in0_lcrit_alarm`): 1 only while the supply is low, unlike the sticky `throttled` bit. Any 1 fails the run and the cell check. Empty when the Pi has no such device. |
| `tag`, `step_m0`, `step_m1` / `step_spin` | Only in `--concurrent` cells: which workloads were inside a step at that instant (`m0+m1`), and each workload's own step. `step` (and `mode`, `iter`) always follow the main workload `m0`. |

Cells measured before v2 have none of these columns.

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
- `sysfs.csv`: v2 (every `run.py` run writes it): per-core clock, temperature and fan state once a
  second, read straight from `/sys` (cheap, unlike the `vcgencmd` reads of `context.csv`). The same
  reading drives the thermal watchdog.
- `counters.csv`, `counters.raw.txt`: the hardware counters (always in version-2 cells, in any newer run; never in the 224 v1 cells). `counters.raw.txt` is
  `perf`'s own output; `counters.csv` is the same data as one row per 100 ms interval.
- `thermal_abort.json`: only when `run.py` stopped the run for heat (see below).
- `derived.json`: written by `verify_cell.py` for every cell.

**Notes on the v2 files:**

- `layer_table.csv`: for FP16 and INT8 variant cells (`variant` is not plain `yolo11n`, ...) do
  **not** use `bytes_moved`, `macs` or `arith_intensity`. They assume 4-byte float weights, and a
  quantized module has no parameters to count, so they are wrong for those cells.
- Cells with `--concurrent` or `--soak-minutes` have no per-layer begin/end events
  (`layer_events: false` in `run_meta.json`; step events and the mode / iteration labels stay), and their `layer_table.csv` timings are of course slowed by the other workload.
- A cell with `--quant` or `--soak-minutes` has no per-layer power benchmarks (`perop_power.csv` is
  empty): quantized layers cannot be benchmarked in isolation, and a soak must stay bounded.

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

These numbers are dataset v1. **Dataset v2 has only had a three-cell smoke so far** (the harness is built; the
probes and the pilot come first, see [docs/dataset-v2.md](docs/dataset-v2.md#7-phase-1-and-phase-2)).
v1 ran with the Monsoon at 5.25 V on Pi kernel 6.18.50; v2 runs at 5.35 V on kernel 6.18.60, and
the `ref` family exists to measure how much that changes the same workloads. A first v2 FP32 cell
(2026-11-20) drew 9.29 W in `forward` and 5.01 W at baseline, against 9.61 W and 5.15 W for the
matching v1 cell; see [docs/dataset-v2.md](docs/dataset-v2.md#7-phase-1-and-phase-2).

---

## 8. Power safety: turning the Pi on and off, rewiring

The Monsoon is the Pi's only power source. Some of its behaviour is surprising:

- **Stopping the recorder does NOT turn the Pi off.** The output stays on ("keep-on"), so
  restarting the recorder never cuts the Pi's power by accident.
- **A Pi that has been shut down (`sudo poweroff`) stays off** until its power is switched off and
  on again. Restarting the recorder alone does not do that.
- **The only program that switches the output off is `host/vout_off.py`**, and it refuses to run
  without `--confirm-pi-halted`.

**To rewire, change the voltage, or restart the recorder cleanly**, do exactly this, in order
(the recorder refuses a `--vout` outside 4.80-5.40 V; the lab runs at 5.40 V):

1. Shut the Pi down cleanly: `ssh -t maccpi@192.168.0.243 sudo poweroff`. Wait until it stops
   answering `ping 192.168.0.243`.
2. Stop the recorder (the output stays on): `systemctl --user stop monsoon-recorder`.
3. Switch the output off: `cd ~/power-measurement-pi/host && ~/.local/bin/uv run python vout_off.py --confirm-pi-halted`
4. Rewire if needed, and check polarity with a multimeter.
5. Start the recorder ([section 4.7](#47-start-the-recorder)). The output comes on and the Pi boots.

Every recorder start creates a new folder in `host/data/`; old recordings are kept.

**Raised again to 5.40 V (2026-11-21).** At 5.35 V the n-models ran clean but the s/m models still triggered 17 under-voltage events during the on-Pi mAP runs, and no lower-resistance cable is available. At 5.40 V (the recorder's upper limit) a 25 s yolo11m loop at 2400 MHz with the fan at full kept the Pi's input at 4.91-5.26 V (median 5.23 V), with the live under-voltage alarm at 0 and `throttled=0x0` in all 100 samples.

**Why 5.35 V and not 5.25 V.** After a Monsoon restart and a cable reseat on 2026-11-20 the Pi's
input sagged under YOLO load: the Pi reported under-voltage (`throttled` bits `0x50005`, the kernel
logged `Undervoltage detected`) and the firmware throttled it. A YOLO forward pass then drew about
7.5 W where v1 had drawn about 9.6 W at the same pinned clock. That is not a real workload effect,
so `run.py` fails such a run. After raising the Monsoon output to 5.35 V, a YOLO loop on the Pi
showed 5.24 V at idle, a median of 5.17 V under load (lowest of 80 readings 4.74 V) and throttle
bits `0x0` throughout. Dataset v1 was measured at 5.25 V.

**Restart the recorder at least every 9 days.** Besides the 9.9-day library limit, it stalls
*silently* at 2^32 - 1 samples ([section 4.7](#47-start-the-recorder)).

---

## 9. Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| Pi does not answer `ping`/`ssh`, and the Monsoon shows it drawing only about 0.28-0.34 A (how to check: end of this section) | **Pi lockup** (see below). | Wait 3 minutes: sometimes it reboots by itself. If not, follow [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring) steps 2, 3 and 5 (it is already halted). The matrix orchestrator does this itself with `--auto-power-cycle`. |
| Cell failed at `align` with "residual >= 1.0 ms" | The Monsoon lost samples right inside a sync pulse, so the clocks could not be matched. | Nothing. The cell is retried automatically. It happened 4 times in 228 attempts. |
| `slip_warning` in results | The Monsoon lost a few ms of samples during the run (about once a minute). Labels after that point are shifted by a few ms. | Accepted. Repeats average it out. Filter on `alignment.json` → `slip_samples_total` if you need perfect labels. |
| Run fails with "under-voltage occurred during run" | The Pi's input voltage dipped below about 4.63 V. The firmware then throttles the CPU, so the power numbers are not the workload's. | Check the power cable (short and thick, about 0.1 ohm) and the Monsoon voltage (5.35 V, [section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring)). Heavy workloads above about 3 A are at risk. |
| Run fails with "live under-voltage alarm during run (N samples)", or a cell fails at stage `undervoltage` | The live alarm (`sysfs.csv` `uv_alarm`) or the kernel log of the cell ("Undervoltage detected") caught a dip, even if the sticky `throttled` bit was already set from earlier. | As above. If `./lab check` shows `0x50000` since boot, reboot the Pi before a long matrix. |
| Experiment stopped at stage `pi_busy` | A `run.py` was still running on the Pi when the next cell was due (for example after a timeout). | `ssh maccpi@192.168.0.243 pgrep -af src/run.py`; wait for it or stop it with `ssh maccpi@192.168.0.243 pkill -INT -f src/run.py`, then start `run` again. |
| Cell failed at stage `timeout` | `run.py` ran longer than the cell's limit. The orchestrator sent it SIGINT and waited up to 60 s. | The error says whether it exited. Raise `timeout_s`, or check why the cell is slow. |
| Run fails with "baseline drift" or clock errors | The Pi changed speed, or something else ran during the measurement. | Make sure nothing else runs on the Pi. Check `pi-power-pin status`. |
| `Monsoon recorder unreachable` | The recorder is not running. | Start it ([section 4.7](#47-start-the-recorder)). |
| `recorder is not sampling: n_total stayed at ...` or `within an hour of the 32-bit ceiling` | The recorder stalled silently (`/status` still says recording), or is about to. | Restart the recorder the safe way ([section 8](#8-power-safety-turning-the-pi-on-and-off-rewiring)). The orchestrator and `./lab` never do it for you. |
| Cell failed at stage `thermal_abort` | `run.py` stopped at 85 °C (exit 3) or at 90 °C (exit 4). `thermal_abort.json` in the cell folder says when, how hot, and in which phase. | Fan `off` and `auto` cells, soaks and hot rooms can do this. Let the Pi cool. The cell is not retried; the experiment goes on. |
| Experiment stopped at stage `disk` | Less than 10 GB free on the EliteBook runs folder or 5 GB on the Pi (2 GB more for soaks). | Move finished runs elsewhere, then start `run` again (done cells are skipped). |
| `counter coverage 0.xxx < 0.95` in `verify` | Less than 95 % of the measured samples have counter data: `perf` was missing, died, or the cell was short. | Check `perf_counters` in `run_meta.json` (`error`, `min_pct_running`) and that `perf` is installed on the Pi ([section 4.8](#48-extra-setup-for-dataset-v2)). |
| `./lab check` warns `pi wifi` | The Pi's Wi-Fi radio is up and adds power noise. | Turn it off (`ssh -t maccpi@192.168.0.243 'sudo nmcli con down stevens-iot'`) and measure over ethernet. |
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

1. **Pin** the CPU clock and fan (`tools/pi-power-pin`), and record their state. v2 cells can
   instead leave the clock to `ondemand` (`--governor ondemand`, the clock is pinned only around
   the sync pulses) or set the fan to `auto` or `off` (`--fan`). From here on a 1 Hz sampler reads
   clock, temperature and fan state from `/sys` into `sysfs.csv`, and doubles as the thermal
   watchdog: it aborts the run at 85 °C (exit code 3) and stops it at 90 °C (exit code 4), writing
   `thermal_abort.json`. The checks happen at every frame, pass and request.
2. **`load_weights`**: build or load the model. This is measured too. For an INT8 or FP16 variant
   the variant is built or loaded here, and a run that quantized no module fails instead of calling
   float32 data INT8. With `--concurrent <model>.pt` the second model is loaded here as well.
3. **`first:<mode>`**: the very first inference or training step. It is slow (memory allocation,
   kernel setup), so it is measured separately.
4. **`warmup`**: a few more passes, so temperature and caches settle. With `--concurrent`, the
   second workload (a thread of the same process, `src/coload.py`) does its own first call and
   warm-up beside the main one, so every thread exists before the next step.
   Then **`perf stat`** is attached to the run (`perf stat -I 100 -p <pid>`, only for v2 and any
   other run with counters on). It only counts threads that already exist, which is why it starts
   after warm-up. It records `t0_ns` on the lab clock just before starting; `perf` takes a few tens
   of milliseconds to start its own clock (14-37 ms measured on the Pi), and that offset is found
   from the time its first line appears and corrected, so `counters.csv` lines up with the power
   trace.
5. **Sync pulse (start)**, then **`baseline_A`**: 2 s of idle to measure the idle power.
6. **Measured modes** (`forward`, or `prefill` + `decode`, or `train`): N iterations, each split
   into steps, with every layer call logged. With `--soak-minutes M` the `forward` loop runs for M
   minutes instead of N iterations. With `--concurrent`, the second workload is released after the
   main one's first (profiled) frame and runs until the main loop ends; the short wait for it to
   finish is the `concurrent_drain` phase, and step events carry a `tag` (`m0`, `m1`, `spin`).
7. **`serve`** (inference only): single requests with idle gaps, to see how power behaves when
   going between busy and idle. The gap is fixed (200 ms by default), or with `--serve-fps F`
   request k is due at `k/F` seconds after the start, so the idle time is whatever is left (a late
   request starts at once and counts as a missed deadline).
8. **`baseline_B`**: 2 s idle again. The idle power is the mean of A and B; if they differ by more
   than 0.5 W the run fails.
9. **`bench:<config>`** (inference only): each distinct layer type is run alone in a tight loop
   for 1-2 s, giving its steady-state power.
10. **Sync pulse (end)**, stop `perf`, write `counters.csv`, `events.csv`, `run_meta.json` and the
    other tables. (`sysfs.csv` is written last, even after a crash or abort, and a fan-`off` run
    always puts the fan back to `auto`.)
11. **Acceptance checks**: the clock stayed pinned (or, in a DVFS run, stayed within 1500-2400 MHz),
    no throttling or under-voltage, the fan stayed at full (when it was set to full), the model
    produced sane output, an INT8 variant really executed quantized modules, an FP16 variant really
    executed in `float16`, and a concurrent second workload completed iterations. Any failure
    fails the run.

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

**Dataset v2 additions.** After the sync-pulse alignment, `align.py` also joins `counters.csv`
(each sample gets the 100 ms interval that contains it) and `sysfs.csv` (the latest 1 Hz reading,
at most 2 s old) onto the trace on the Pi timestamp `t_pi_ns`; nothing needs the Monsoon clock for
that, because the Pi stamped both with its own lab-synced clock. In a `--concurrent` cell only the
main workload's events label `mode`, `iter` and `step`, and each workload gets its own
`step_<tag>` column. The 1 ms alignment limit is unchanged for every mode. `verify_cell.py` then
adds the v2 checks (section 6) and writes `derived.json`.

### 10.3 The experiment orchestrator (`host/experiment.py`)

- `plan` expands `host/matrix.py` into a manifest (or `host/matrix_v2.py` with `--v2`, section 5.4.2).
- `run` works through it round by round; each attempt is appended to `results.jsonl`.
- **Failure stages:**
  - `recorder`, `disk`, `pi_busy` and `pi_down` stop the whole experiment, because nothing can work
    until a human looks. (`recorder` now also covers a recorder that has stopped sampling silently.)
  - `thermal_abort` (`run.py` stopped at 85 or 90 °C) ends that cell without a retry; the
    experiment goes on.
  - `undervoltage` (the cell's `dmesg` shows "Undervoltage detected") and `timeout` (`run.py` got
    SIGINT after the cell's time limit) are retried like `pi_crash`, `run`, `rsync`, `align` and
    `verify`.
- **Power-cycle guard:** it only switches the power off and on when all three are true:
  - the Pi has not answered for 3 minutes;
  - it does not answer `ping`;
  - the Monsoon shows halted-level current (mean below 0.4 A and max below 0.6 A over 5 s).
- **Last step:** at the very end it restores the Pi's normal clock and fan settings.

---

## 11. Reference

### 11.1 `src/run.py` flags

The flags every run has had since v1:

| Flag | Default | Meaning |
|---|---|---|
| `--task` | `image` | `image` (detection/classification), `lm` (language model), `train` |
| `--model` | `yolo11n.pt` | `yolo11n.pt`, `yolov8n.pt` (any ultralytics weights, v2 also uses `yolo11s.pt`, `yolo11m.pt`, `yolov8s.pt`, `yolov8m.pt`); `torchvision:<name>` (e.g. `torchvision:resnet18`); `minigpt`, `minigpt-explicit`; `hf:<name>` (Hugging Face causal LM) |
| `--sampler` | `pmic` | `monsoon` for real data; `pmic` is the old on-board meter |
| `--recorder-url` | `http://192.168.0.11:8051` | where the recorder is |
| `--cpu-mhz` | none | pin the clock: 1500-2400 in steps of 100. With `--governor ondemand` it is the clock used around the sync pulses |
| `--fan-full` | off | fan at full speed for the whole run (same as `--fan full`) |
| `--frames` | 30 | measured iterations per mode |
| `--warmup` | 10 | warm-up passes |
| `--threads` | 4 | threads for PyTorch, OpenCV and OpenMP together (v2 also uses 1 and 2) |
| `--imgsz` | 640 | image size (image and image-training tasks) |
| `--seq` | 64 | prompt length (LM tasks) |
| `--batch` | none | batch size. Training: 8 when not given (at least 2). Image task: 1 when not given; with B, each `forward` processes B pictures stacked together. Ignored by LM tasks |
| `--optimizer` | `sgd` | `sgd` (momentum 0.9, lr 0.01) or `adam` (lr 1e-3) |
| `--serve` | 0 | number of single requests in the `serve` phase |
| `--serve-gap-ms` | 200 | idle time between serve requests |
| `--bench` | 3.0 | seconds per isolated layer benchmark |
| `--no-bench` | off | skip the layer benchmarks |
| `--check-image` | none | detection sanity check: this image must give a detection with at least `--conf` confidence |
| `--conf` | 0.25 | detection confidence threshold |
| `--live-camera` | off | use real camera frames (Pi camera) instead of a fixed image |
| `--out` | `runs/<timestamp>` | output folder |

Flags added for dataset v2. Every default reproduces the v1 behaviour:

| Flag | Default | Meaning |
|---|---|---|
| `--holdout` | none | `primary-s`, `secondary-m` or `none`. Written to `run_meta.json` (`none` becomes `null`). **Giving the flag at all, even `none`, marks the run `dataset_version` 2** |
| `--governor` | `pinned` | `pinned`: `--cpu-mhz` held all run. `ondemand`: the clock floats (DVFS) for the workload and `--cpu-mhz` (required) is pinned only around the sync-pulse trains, so the alignment still sees a steady clock. Checked: every core stays within 1500-2400 MHz |
| `--fan` | untouched | `full`, `auto` (stock trip points) or `off` (trip points at 85 °C and up, so the fan stays off). Without `--fan` the fan is left alone unless `--fan-full` is given. Cannot be combined with a different `--fan-full` |
| `--serve-fps` | none | with `--serve N`: request k is due at `k / F` seconds; the idle gap is the time left, a request that is already late starts at once and counts as a missed deadline. Replaces `--serve-gap-ms` |
| `--soak-minutes` | none | image task: repeat the measured `forward` loop for this many minutes instead of `--frames` iterations. Turns the benchmarks off |
| `--input-image` | `--check-image` | the picture fed to the model (the sanity check still uses `--check-image`) |
| `--frames-source` | `repeat` | `repeat`: one picture again and again. `stream`: cycle through the sorted `*.jpg` of `--stream-dir`, decoded once at load. Cannot be combined with `--live-camera` or `--input-image` |
| `--stream-dir` | `inputs/stream50` | picture folder for `--frames-source stream` |
| `--quant` | `fp32` | YOLO precision variant (image task): `fp32`; `int8-static` (needs `--quant-weights`); `fp16` (the module and the input are converted to half precision, and the run fails if the convolutions did not really run in float16); `int8-dynamic` (Linear layers only; quantizes 0 modules on YOLO, so the run fails acceptance). Turns the benchmarks off |
| `--quant-weights` | none | the INT8-static file made by `tools/quantize_yolo.py` (e.g. `models/yolo11n_int8s.pt`) |
| `--prune` | none | a pruned-model file made by `tools/prune_yolo.py` (e.g. `models/yolo11n_p30.pt`). Cannot be combined with a `--quant` other than `fp32` |
| `--concurrent` | none | image task: run a second workload as a thread of the same process during the measured `forward` loop: `<model>.pt` (a second model, same size, batch and input; the threads are split `--threads // 2` each) or `spinner` (one thread multiplying matrices; the main workload keeps `--threads - 1`). One sync envelope, step events tagged `m0`, `m1` or `spin`, no per-layer events. Cannot be combined with `--soak-minutes`, `--live-camera`, `--quant` or `--prune` |
| `--perf-counters` | `on` | hardware counters with `perf stat -I` attached to the run (`counters.csv`). On by default for every run, including v1-shaped command lines; forced off with `--sampler pmic`. `off` is what the `ref` family uses |
| `--perf-interval-ms` | 100 | counter sampling interval: 100 (10 Hz) or 200 (5 Hz) |

A run is `dataset_version` 2 when `--holdout` is given, or when any of `--governor ondemand`,
`--fan auto`/`off`, `--serve-fps`, `--soak-minutes`, `--frames-source stream`, `--input-image`,
`--quant` (not `fp32`), `--prune` or `--concurrent` is used. `--threads`, `--batch`, `--imgsz` and
`--perf-counters` alone do not change it. Otherwise it is 1.

**Exit codes of `run.py`:** 0 ok; 3 thermal abort (85 °C); 4 thermal hard stop (90 °C); any other
non-zero is an acceptance failure or a crash.

### 11.2 `host/experiment.py`

| Command | Options |
|---|---|
| `plan --out <manifest>` | `--smoke` (3 cells), `--probe` (each model once at 2400 MHz), `--only a,b` (only these model keys), `--seed N` (shuffle order) |
| `plan --v2 <family> --out <manifest>` | `<family>` is `A`, `A-input`, `B`, `C`, `D`, `soak`, `ref`, `all` or `pilot`; `--seed N`; `--include-fp16`, `--include-optional`, `--include-int8d`, `--include-prune` add cells, `--no-concurrent` removes the concurrency cells. Cannot be combined with `--smoke`, `--probe` or `--only` (section 5.4.2) |
| `run --manifest <manifest>` | `--max-attempts N` (default 2), `--auto-power-cycle N` (default 0), `--results <file>`, `--allow-git-mismatch` |
| `status --manifest <manifest>` | `--results <file>` |

`./lab` (`host/lab.py`) has its own options, all listed in [INSTRUCTIONS.md](INSTRUCTIONS.md). New
since v1: the dataset-v2 presets, the checks `recorder stall`, `helper version`, `pi wifi` and
`perf`, and `./lab run --yes-supervised` for fan-off presets.

### 11.3 `events.csv` (Pi side)

Columns: `t_real_ns` (lab-synced wall clock), `t_mono_ns`, `kind`, `name`, `depth`, `mode`, `iter`,
`layer_path`, `config_id`, `window_id`, `tag` (empty, or `m0`/`m1`/`spin` on step events of a
`--concurrent` run; `experiment.py plan --v2 ... --no-concurrent` leaves those cells out).

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
- **Software:** `torch_version`, `schema_version` (4 for Monsoon runs, whether v1 or v2).

Fields added for dataset v2. Each one is always present, with a v1-style value (`null`, `false`,
`"pinned"`, ...) when the feature was not used:

- **Which dataset:** `dataset_version` (1 or 2, see [section 11.1](#111-srcrunpy-flags) for the
  rule) and `holdout` (`"primary-s"` for yolo11s / yolov8s, `"secondary-m"` for yolo11m / yolov8m,
  `null` otherwise, and `null` for every v1-shaped run).
- **Condition:** `governor` (`pinned` or `ondemand`), `fan_mode` (`full`, `auto`, `off` or `none`
  for a fan that was left alone), `threads_detail` (what torch, OpenCV and OpenMP reported back
  for `--threads`; in a concurrent run it is the main workload's share).
- **Precision variant:** `variant` (`yolo11n`, `yolo11n_int8s`, `yolo11n_fp16`, `yolo11n_int8d`,
  `yolo11n_p30`, ...), `quant`, `prune` (details of the pruned model), `quant_weights_sha256`,
  `quantized_modules` and `quantized_compute_modules` (the same number: real quantized
  Conv / Linear modules, not the stubs around them; it must be above 0 for an INT8 cell),
  `executed_dtype` (the input type the convolutions really ran in, e.g. `float16`, read from the
  layer profile, not from what was asked for), `bench_forced_off` (`"quant"` when the variant
  switched the benchmarks off) and `bench_forced_off_by_soak`.
- **Accuracy:** `map_reference`: the mAP50-95 and mAP50 of this model / variant / image size on
  500 COCO validation pictures, copied from `data/map_reference.json` (also `n_images`, `dataset`,
  `tool_version`, `measured`, and `measured_on` once an entry has been measured on the Pi; `null`
  if there is no entry). It is one number per model, not measured per cell. `verify_cell.py` warns
  until the entry carries `measured_on.machine == "aarch64"`: the current entries are provisional
  numbers measured on the NUC, and the final ones are measured on the Pi
  ([tools/README.md](tools/README.md)).
- **Input and workload:** `inputs` (`source`: `repeat`, `stream` or `camera`, and `files` with the
  path and sha256 of each picture used), `serve_fps`, `serve_missed_deadlines`, `serve_wall_s`,
  `soak_minutes`, `frames_done` (iterations really done per mode: the truth in a soak run, where
  `frames_per_mode` is only the nominal value), `latency_ms_per_frame` (mean, median, p95 and n of
  the measured iterations, frame 0 excluded; only for runs with one measured mode) and
  `latency_ms_per_image` (the same divided by the batch).
- **Concurrency:** `concurrent` (`null`, or: `mode` `model`/`spinner`, `model`, `tags`,
  `threads_split`, `iterations` per tag, `overlapped_primary_frames`, `latency_ms` and
  `first_call_ms` per tag, `load_weights_s`, `sync_envelopes` (always 1), `serve_concurrent`
  (false: `--serve` runs after the second workload has stopped), how it was released and stopped,
  and for the spinner its matrix size) and `layer_events` (only present, as `false`, in a
  concurrent run).
- **Counters:** `perf_counters`: `enabled`, `available` (is `perf` installed), `interval_ms`,
  `events` (the list, in column order), `t0_ns` (the Pi's clock just before `perf` was started),
  `offset_ns` and `offset_method` (`first-line`, or `none` if no first line appeared within 2 s):
  how late `perf` started its own clock, see section 10.1, `paranoid` (the kernel's
  `perf_event_paranoid`) and `scope` (`user`: with the lab's value 2 only the run's user-space
  code is counted, events print as `cycles:u`), `min_pct_running` (the lowest share of an
  interval in which any hardware event was counted; 100 means no sharing of counters), and
  `error` (why counters are missing or incomplete, else `null`).
- **Thermal:** `sysfs_period_s` (1.0), `max_temp_C` (highest temperature read), `thermal_abort`
  (always `null`; an abort writes `thermal_abort.json` and ends the run, normally before `run_meta.json` exists).

### 11.6 `phase_energy.csv`

One row per (`phase`, `iter`, `step`): `n_samples`, `mean_P_W`, `dur_ms`, `E_mJ` (energy in
millijoules = W × ms), `ctx_read_frac`, `t_start_s`, `t_end_s`.

### 11.7 Units

Numeric columns carry their unit in the name: `_W` watts, `_A` amps, `_V` volts, `_ms`
milliseconds, `_ns` nanoseconds, `_mJ` millijoules, `_C` °C, `_MHz`, `_kHz`. Energy is always
`E_mJ = P_W × t_ms`.

### 11.8 Dataset v2 files

**`counters.csv`** (Pi, one row per `perf` interval): `t_start_ns`, `t_end_ns` (the Pi's lab clock,
the same clock as `t_pi_ns` in the trace; the start of the first interval is where counting
began), then `ctr_<event>` (the count in that interval; empty if `perf` could not count the event)
and `pct_<event>` (0-100, the share of the interval in which the event was really counted) for each
event of the default set: `cycles`, `instructions`, `cache-references`, `cache-misses`,
`branch-misses`, `stalled-cycles-backend`, `LLC-load-misses`, `task-clock`, `context-switches` and
`page-faults` (a `-` in the event name becomes `_` in the column). The first seven are hardware
events and fit the Pi 5's CPU counters (one cycle counter plus six programmable ones) at the same
time, so they run 100 % of the time; the last three are software counters. Left out on purpose:
`L1-dcache-loads` and `L1-dcache-load-misses` (they count exactly like `cache-references` and
`cache-misses` on this CPU), and `LLC-loads` and `bus-cycles` (an eighth hardware event would
make every hardware event run only part of the time).

**`counters.raw.txt`**: `perf stat -I 100 -x,` output, untouched. `counters.csv` is made from it.

**`sysfs.csv`** (Pi, once a second): `t_real_ns`, `cpu0_kHz` ... `cpu3_kHz`, `temp_C`,
`fan_state`, `uv_alarm` (0/1, live under-voltage alarm). An unreadable value is empty.

**`thermal_abort.json`** (Pi, only when the run was stopped for heat): `t_real_ns`, `temp_C`,
`level` (`abort` at 85 °C, `hard_stop` at 90 °C) and `phase` (what the run was doing).

**`derived.json`** (EliteBook, from `verify_cell.py`): measured iterations only (frame 0, the
cold first call of a mode, is left out). `energy_per_frame_mJ` (energy of one iteration, all
samples in it), `dyn_energy_per_frame_mJ` (the same minus the idle power times the iteration's
duration), `latency_ms_per_frame`, `energy_per_image_mJ` (per frame divided by the batch),
`n_frames`, `batch`, `baseline_mean_P_W`, `by_mode` (the same figures for each mode), and a copy
of `variant`, `holdout` and `map_reference`. A concurrent cell adds `concurrent`; its energy per
frame covers both workloads together, not just the main one.

**What `verify_cell.py` checks beyond v1** (a dataset v2 cell, `dataset_version` 2):

| Check | Fails / warns when |
|---|---|
| counters on | `perf_counters.enabled` is false (fails) |
| counter coverage | less than 95 % of the samples inside the measured phases have counter data (fails; the value is in the result as `counter_coverage`) |
| thermal | `thermal_abort.json` exists or `max_temp_C` is 85 °C or more (fails) |
| fan mode | `fan_mode` is `none` (an untouched fan), or `full` without `fan_full` (fails; `auto` and `off` are accepted, the fan is no longer required to be full) |
| DVFS clock | in a `governor: ondemand` cell, any core outside 1500-2400 MHz (+/-25) in the measured phases (fails) |
| concurrent cells | a workload completed no iterations or left no `step_<tag>` column (fails); the second workload's expected steps are missing, or its steps cover under 50 % of its active window (fails; warns under 90 %); the main workload was inside a step during under 50 % of the second workload's window, meaning they did not really overlap (fails; warns under 90 %) |
| accuracy reference | a YOLO cell has no `map_reference` entry, or the entry was not measured on the Pi (warns) |

The alignment check (`max_residual_ms` < 1.0) is the same for every cell, DVFS and concurrent
ones included.

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
# NUC tools (run on the NUC, own environment)
cd ~/Desktop/power-measurement-pi/tools && ~/.local/bin/uv run pytest tests
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

- **Tracked:** code, tests, docs, `uv.lock`, and two small result files that the Pi needs:
  `data/map_reference.json` (mAP per model / variant) and `inputs/MANIFEST.json` (which COCO
  pictures `inputs/` holds, with checksums).
- **Not tracked** (see `.gitignore`):
  - `runs/` (measurement data, back it up separately);
  - `host/data/` (raw Monsoon recordings, about 0.15 GB per hour, 3.5 GB per day);
  - `.venv/`, model weights (`*.pt`), logs;
  - `tools/data/` (COCO downloads and subsets), the pictures in `inputs/`, and `models/` (INT8
    and pruned model files): rebuild them with `tools/` and copy them to the Pi
    ([tools/README.md](tools/README.md)).

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
  sysfs_sampler.py    1 Hz clock / temperature / fan reads and the 85 / 90 C thermal watchdog
  perfcounters.py     perf stat hardware counters attached to the run (counters.csv)
  yolo_variants.py    FP16 / INT8 / pruned YOLO variants and their honesty checks
  coload.py           second workload thread for --concurrent
  sync_check.py       emit sync pulses only (alignment self-test)
lab                   one-command launcher (wrapper for host/lab.py)
host/                 EliteBook side
  lab.py              single runs and tests: check / run / test / presets
  recorder.py         always-on Monsoon recorder + HTTP API
  align.py            line up clocks, label every sample
  verify_cell.py      check one cell, write phase_energy.csv
  experiment.py       run the matrix (plan / run / status)
  matrix.py           which models, clocks, repeats (dataset v1)
  matrix_v2.py        the dataset-v2 families (plan --v2)
  vout_off.py         manual safety tool: switch the Monsoon output off
  store.py, clockfit.py   recording storage and clock fitting
tools/pi-power-pin    root helper for clock and fan pinning
tools/*.py            NUC-side dataset tools: COCO subsets, mAP, INT8 and pruned models (tools/README.md)
data/                 map_reference.json: mAP per model / variant (committed)
inputs/               pictures the Pi reads (empty, crowded, a 50-picture stream); MANIFEST.json is committed
tests/, host/tests/, tools/tests/   unit tests
docs/                 design docs, plans, logs
```

Further reading:

- Dataset v2: what it adds, the v1/v2 counter split, the holdout, the one-time setup and the
  Phase 0 probe checklist: [docs/dataset-v2.md](docs/dataset-v2.md)
- NUC tools (COCO subsets, mAP, INT8, pruning): [tools/README.md](tools/README.md)
- Dataset v2 harness plan (the tasks behind it): `docs/superpowers/plans/2026-11-19-dataset-v2-harness.md`
- Design spec: `docs/superpowers/specs/2026-10-06-monsoon-experiment-design.md`
- Measurement-plane bring-up log (what was measured and decided, SP1):
  `docs/superpowers/plans/sp1-bringup-log.md`
- Workloads and matrix plan (SP2 and SP3): `docs/superpowers/plans/2026-10-07-sp2-sp3-workloads-and-matrix.md`
- The old on-board PMIC harness (schema v3), `collect.py`, model-agnostic adapters:
  `docs/legacy-pmic-harness.md`. Its code (`collect.py`, `verify.py`, `analyze_trends.py`, `modeling/`,
  `trace_forecast/`, `analysis/`) and its September runs (`runs/`, `test_eff.pth`) were removed from `main` on
  2026-11-25. They are still in the history: `git checkout fc331b0 -- <path>`.
- **Next step (not built yet), SP4:** one index across all cells, for training the power
  predictor.
