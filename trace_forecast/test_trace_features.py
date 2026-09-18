import sys, os; sys.path.insert(0, os.path.dirname(__file__))
def test_dry_sums_keys():
    from features import dry_sums
    import torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3,4,3), nn.ReLU())
    s = dry_sums(m, [[1,3,16,16]])
    assert s["sum_macs"] > 0 and s["sum_bytes"] > 0 and "Conv2d" in s["op_mix"]
def test_dry_sums_lm_shapes():
    from features import dry_sums
    import torch.nn as nn
    m = nn.Sequential(nn.Embedding(32,8), nn.LayerNorm(8))
    s = dry_sums(m, [[1,16]], dtype="torch.int64")
    assert s["T"] == 16 and s["sum_bytes"] > 0
