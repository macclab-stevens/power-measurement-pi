import csv, glob, os, json
from collections import Counter

d = sorted(glob.glob("runs/*/"))[-1]
print("DIR:", d)
for f in ["samples.csv", "context.csv", "layer_table.csv", "op_table.csv",
          "perop_power.csv", "annotated.jpg", "run_meta.json"]:
    print("  ", f, "OK" if os.path.exists(d + f) else "MISSING")

perop = list(csv.DictReader(open(d + "perop_power.csv")))
print("\nperop rows:", len(perop),
      "skipped:", sum(1 for r in perop if r["SKIPPED"] == "True"))

big = sorted((r for r in perop if r["P_delta_W"]), key=lambda r: -float(r["P_delta_W"]))[:5]
print("top P_delta_W ops:")
for r in big:
    print(f"  {r['class']:12s} P_delta={float(r['P_delta_W']):.4f}W  "
          f"E={r['E_per_call_mJ']}mJ  t={r['t_per_call_ms']}ms  samples={r['samples_used']}")

for cls in ("Identity", "Concat", "Conv2d"):
    rows = [r for r in perop if r["class"] == cls]
    nd = sum(1 for r in rows if r["P_delta_W"])
    print(f"  {cls}: {len(rows)} rows, {nd} with power, skipped={sum(1 for r in rows if r['SKIPPED']=='True')}")

s = list(csv.DictReader(open(d + "samples.csv")))
phs = Counter(r["phase"].split(":")[0] for r in s)
print("\nsamples:", len(s), "phase coverage:", dict(phs))

ops = list(csv.DictReader(open(d + "op_table.csv")))
top = sorted(ops, key=lambda r: -float(r["self_total_us"]))[:4]
print("op_table:", len(ops), "top by self time:")
for o in top:
    print(f"  {o['op'][:40]:42s} cnt={o['count']:>4} mean={o['self_mean_us']}us")

c = perop[0]
print("\nperop env bracket sample: temp", c["temp_start_C"], "->", c["temp_end_C"],
      " freq", c["freq_start_MHz"], "->", c["freq_end_MHz"])

meta = json.load(open(d + "run_meta.json"))
print("meta: baseline_W", meta["baseline_W"], "hz", meta["achieved_sample_hz"],
      "fps", meta["mean_fps"], "throttle", meta["throttle_bits"])

n_kbd = [r for r in csv.DictReader(open(d + "layer_table.csv"))]
print("\nlayer_table rows:", len(n_kbd))