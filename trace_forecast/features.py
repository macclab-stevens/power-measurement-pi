import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from layers import LayerProfiler
def dry_sums(module, fixture_shapes, dtype="torch.float32"):
    prof = LayerProfiler(module)
    with torch.no_grad():
        prof.set_mode("dry"); prof.set_frame(1)
        if dtype == "torch.int64":
            x = torch.randint(0, 32, tuple(fixture_shapes[0]), dtype=torch.int64)
        else:
            x = torch.rand(*fixture_shapes[0])
        module(x)
    rows = prof.aggregated(exclude_frame=0)
    leaves = [r for r in rows if r["leaf"]]
    op_mix = {}
    for r in leaves: op_mix[r["class"]] = op_mix.get(r["class"], 0)+1
    tot = max(1, len(leaves))
    return {"sum_macs": sum(r["macs"] or 0 for r in leaves),
            "sum_bytes": sum(r["bytes_moved"] or 0 for r in leaves),
            "sum_teff": sum(r.get("t_eff") or 0 for r in leaves),
            "n_leaves": len(leaves),
            "op_mix": {k: v/tot for k, v in op_mix.items()},
            "T": fixture_shapes[0][-1] if len(fixture_shapes[0]) == 2 else (fixture_shapes[0][-2]*fixture_shapes[0][-1] if len(fixture_shapes[0]) >= 4 else 1)}
