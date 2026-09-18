import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch.nn as nn
from layers import _macs, _bytes_moved
def test_norm_macs_not_none():
    m = nn.LayerNorm(64)
    v = _macs(m, [[1,64,64]], [[1,64,64]])
    assert v == 2*64*64, f"LayerNorm MACs should be 2*numel, got {v}"
def test_bytes_uses_real_dtype():
    rec = {"args": [{"k":"tensor","shape":[1,64],"dtype":"torch.int64","value":"zeros","bytes":512}]}
    b = _bytes_moved(rec, [[1,64,64]], out_dtype="torch.float32")
    assert b == 512 + 1*64*64*4, f"got {b}"
