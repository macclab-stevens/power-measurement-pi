import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn
from layers import module_config, config_id
def test_bias_values_excluded():
    m1 = nn.Conv2d(3, 16, 3, bias=True)
    m2 = nn.Conv2d(3, 16, 3, bias=True)
    with torch.no_grad():
        m2.bias.fill_(99.0)
    c1, c2 = module_config(m1), module_config(m2)
    assert "bias" not in str(c1).lower() or c1.get("has_bias") in (True, False)
    assert "99" not in str(c1) and "99" not in str(c2)
    assert config_id({**c1, "shapes": [[1,3,32,32]], "dtypes": ["torch.float32"]}) == config_id({**c2, "shapes": [[1,3,32,32]], "dtypes": ["torch.float32"]})
def test_bn_does_not_dump_parameter():
    m = nn.BatchNorm2d(8)
    c = module_config(m)
    assert "Parameter" not in str(c)
    assert "containing" not in str(c)
