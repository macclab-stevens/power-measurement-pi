"""Sanity-check the latest run's dataset (new generalized schema).

Checks the model-agnostic contract:
  - all expected files present (run_meta now has op_sequence + arch_fingerprint)
  - perop: no SKIPPED, schema columns (input_dtypes, macs, bytes_moved), a
    dominant op class with positive P_delta + sufficient samples
  - layer_table: leaf rows carry config_id / input_dtypes / macs / bytes /
    arith_intensity / mode; measurement identity is shape+dtype aware
  - op_table non-empty; samples/phase coverage present
  - run_meta: baseline, hz, task, modes, dominant-op summary
Usage: uv run python verify.py [runs/<ts>]
"""
import csv, glob, json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/*/"))[-1]
print("DIR:", path)


def _cols(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


ok_all = True


def _has(name):
    global ok_all
    p = os.path.join(path, name)
    ok = os.path.exists(p)
    ok_all = ok_all and ok
    print("  ", name, "OK" if ok else "MISSING")
    return ok


for f in ["samples.csv", "context.csv", "layer_table.csv", "op_table.csv",
          "perop_power.csv", "run_meta.json"]:
    _has(f)

perop = _cols(os.path.join(path, "perop_power.csv"))
skipped = sum(1 for r in perop if r["SKIPPED"] == "True")
print(f"\nperop rows: {len(perop)}   SKIPPED: {skipped}  ({'GREEN' if skipped == 0 else 'FATAL'})")

# dominant op acceptance (mirrors run.py)
vol = {}
for r in perop:
    if r.get("N_calls"):
        vol[r["class"]] = vol.get(r["class"], 0) + int(r["N_calls"])
dom = max(vol, key=vol.get) if vol else None
if dom:
    bad = [r for r in perop if r["class"] == dom and not r["SKIPPED"] == "True"
           and (not r["P_delta_W"] or float(r["P_delta_W"]) <= 0.01)]
    low = [r for r in perop if r["class"] == dom and not r["SKIPPED"] == "True"
           and int(r["samples_used"]) < 30]
    print(f"dominant op: {dom}  rows={vol.get(dom)}  P_delta_bad={len(bad)}  "
          f"low_samples={len(low)} {'GREEN' if not bad and not low else 'FAIL'}")

big = sorted((r for r in perop if r.get("P_delta_W")),
             key=lambda r: -float(r["P_delta_W"]))[:5]
print("top P_delta_W ops:")
for r in big:
    print(f"  {r['class']:14s} P_delta={float(r['P_delta_W']):.4f}W  "
          f"E={r['E_per_call_mJ']}mJ  t={r['t_per_call_ms']}ms  "
          f"macs={r.get('macs')}  bytes={r.get('bytes_moved')}  samples={r['samples_used']}")

# measurement identity: same (class + ATTRS + shapes + dtype) must be ONE config_id
mu = {}
for r in _cols(os.path.join(path, "perop_power.csv")):
    if not r["SKIPPED"] == "True":
        k = (r["class"], r["config_json"], r["input_shapes"], r["input_dtypes"])
        mu.setdefault(k, set()).add(r["config_id"])
collide = {k: v for k, v in mu.items() if len(v) > 1}
print(f"identity collision (same class+attrs+shapes+dtype, diff config_id): "
      f"{len(collide)} {'GREEN' if not collide else 'FAIL'}")

lt = _cols(os.path.join(path, "layer_table.csv"))
leaves = [r for r in lt if r["leaf"] == "True"]
print(f"\nlayer_table rows: {len(lt)}  leaves: {len(leaves)}  "
      f"modes: {sorted({r['mode'] for r in lt if r['mode']})}")
allfmt = all(r["macs"] != "" or r["bytes_moved"] != "" for r in leaves)
has_ai = sum(1 for r in leaves if r["arith_intensity"])
print(f"  leaves with macs/bytes set: {allfmt}  arith_intensity non-empty: {has_ai}")

ops = _cols(os.path.join(path, "op_table.csv"))
top = sorted(ops, key=lambda r: -float(r["self_total_us"]))[:4]
print(f"\nop_table rows: {len(ops)}  modes: {sorted({r['mode'] for r in ops})}  top:")
for o in top:
    print(f"  {o['op'][:44]:46s} cnt={o['count']:>4} mean={o['self_mean_us']}us  {o['mode']}")

s = _cols(os.path.join(path, "samples.csv"))
phs = {}
for r in s:
    ph = r["phase"].split(":")[0]
    phs[ph] = phs.get(ph, 0) + 1
print(f"\nsamples: {len(s)}  phase coverage: {phs}")

meta = json.load(open(os.path.join(path, "run_meta.json")))
print("meta: task", meta.get("task"), "model", meta.get("model"), "modes", meta.get("modes"))
print("      baseline_W", meta["baseline_W"], "hz", meta["achieved_sample_hz"],
      "throttle", meta.get("throttle_bits"))
print("      arch_fingerprint:", meta.get("arch_fingerprint"))
print("      op_sequence by mode:",
      {m: len(v) for m, v in meta.get("op_sequence", {}).items()})

pk = perop[0]
print("\nperop env bracket: temp", pk["temp_start_C"], "->", pk["temp_end_C"],
      " freq", pk["freq_start_MHz"], "->", pk["freq_end_MHz"])
print("ALL FILES:", "GREEN" if ok_all else "FAIL (missing files)")