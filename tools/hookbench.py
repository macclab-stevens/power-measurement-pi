"""Latency-only: LayerProfiler hooks (+ EventLog layer events, as in measured cells) on vs off, ABAB in one process.
Run from the harness checkout on the Pi (it imports src/ relative to the working directory):
    uv run --frozen python tools/hookbench.py OUT.csv model:imgsz ...
Writes one row per 15-frame block (median/mean ms, hooks 0/1, n_hooked modules); no power is measured."""
import csv, statistics, sys, time
sys.path.insert(0, "src")
import torch
from run import build_parser
from tasks import make_task
from layers import LayerProfiler
from eventlog import EventLog

def clear_hooks(mod):
    for m in mod.modules():
        m._forward_hooks.clear(); m._forward_pre_hooks.clear()

out = open(sys.argv[1], "w", newline=""); w = csv.writer(out); w.writerow(["model", "imgsz", "block", "hooks", "median_ms", "mean_ms", "n_hooked"])
torch.set_num_threads(4)
for spec in sys.argv[2:]:
    model, imgsz = spec.rsplit(":", 1)
    args = build_parser().parse_args(["--task", "image", "--model", model, "--imgsz", imgsz, "--threads", "4"])
    task = make_task("image", model, args)
    nh = 0
    for _ in range(8):
        task.run_one("forward")                     # warm-up
    for b, hooks in enumerate([False, True] * 4):
        clear_hooks(task.module)
        if hooks:
            prof = LayerProfiler(task.module); prof.events = EventLog(); prof.set_mode("forward"); nh = len(prof._modules)
        ts = []
        for k in range(15):
            if hooks:
                prof.set_frame(k + 1)
            t0 = time.perf_counter(); task.run_one("forward"); ts.append((time.perf_counter() - t0) * 1000)
        w.writerow([model, imgsz, b, int(hooks), round(statistics.median(ts), 3), round(statistics.mean(ts), 3), nh]); out.flush()
        print(model, imgsz, b, hooks, round(statistics.median(ts), 2), flush=True)
    clear_hooks(task.module); del task
