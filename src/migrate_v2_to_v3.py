# src/migrate_v2_to_v3.py
"""Mark v2 runs tainted, backfill safe fields. Usage: uv run src/migrate_v2_to_v3.py"""
import csv, glob, os, json
for d in sorted(glob.glob("runs/*/")):
    meta_p = os.path.join(d, "run_meta.json")
    if not os.path.exists(meta_p): continue
    meta = json.load(open(meta_p))
    if meta.get("schema_version") == 3: continue
    # flag
    open(os.path.join(d, "_TAINTED_V2"), "w").write("freq_MHz actually Hz; input_dtypes empty; bias in config_id\n")
    print(f"tainted {d} baseline={meta.get('baseline_W')}")
print("done. Do NOT train on tainted runs.")
