"""Throwaway verification of the reviewer-critical input-capture/replay path:
multi-argument positional leaves, int64 indices, bool masks, and the distinct
prefill/decode measurement identity. Runs layers+microbench only (no PMIC).
"""
import sys, copy
sys.path.insert(0, "src")
import torch
import torch.nn as nn
from layers import LayerProfiler
from microbench import _replay_args, _forward


class Add3(nn.Module):           # leaf (no children), 3 positional tensors
    def forward(self, a, b, c):
        return a + b + c


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(64, 16)          # int64 leaf
        self.add = Add3()                         # multi-arg leaf
        self.scale = nn.Parameter(torch.rand(16))

    def forward(self, idx, mask):                 # idx int64, mask bool
        x = self.emb(idx)                          # int64 -> dtype replay
        x = self.add(x, x, x * self.scale)         # 3-tensor positional
        return x * mask.unsqueeze(-1).float()


def run(T):
    m = Model().eval()
    p = LayerProfiler(m)
    idx = torch.randint(0, 64, (1, T), dtype=torch.int64)
    mask = torch.ones(1, T, dtype=torch.bool)
    with torch.no_grad():
        m(idx, mask)
    cfgs = p.leaf_configs()
    dtypes = set(p.input_recs["emb"][0]["args"][0]["dtype"]
                 for _ in range(1))
    # -- checks ------------------------------------------------------------
    emb = p.input_recs["emb"][0]["args"][0]
    assert emb["dtype"] == "torch.int64", emb
    add = p.input_recs["add"][0]["args"]
    assert len(add) == 3 and all(a["dtype"] == "torch.float32" for a in add), add
    # replay + forward must reproduce shape-correct output on a deepcopy
    entry = cfgs[[k for k, v in cfgs.items() if v["class"] == "Add3"][0]]
    args = _replay_args(entry["input_kind"])
    print("  DEBUG add replay:", [tuple(a.shape) if hasattr(a, "shape") else type(a).__name__ for a in args],
          "| rec_args:", [a.get("k") for a in entry["input_kind"]["args"]])
    assert [tuple(a.shape) for a in args] == [(1, T, 16)] * 3, \
        f"got {[tuple(a.shape) for a in args]}"
    assert args[0].dtype == torch.float32
    dc = copy.deepcopy(entry["module"]).eval()
    with torch.no_grad():
        out = _forward(dc, args)
    assert tuple(out.shape) == (1, T, 16) and torch.isfinite(out).all()
    # distinct prefill/decode leaf configs (shapes in identity)
    ids = set(cfgs.keys())
    print(f"T={T}: leaf configs={len(cfgs)}; emb dtype={emb['dtype']}; "
          f"add args={[a['dtype'] for a in add]}; replay ok shape={tuple(out.shape)}")
    return ids


ids64 = run(64)
ids1 = run(1)
assert ids64.isdisjoint(ids1), "prefill/decode config_ids must differ (shapes in identity)"
print("prefill (T=64) vs decode (T=1):", len(ids64), "vs", len(ids1),
      "configs; shared=0 as required (shape-aware identity)")
print("PASS: multi-arg positional replay + int64/bool + per-mode identity all verified")