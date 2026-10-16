import pytest

import vout_off


def test_vout_off_requires_confirmation():
    with pytest.raises(SystemExit) as e:
        vout_off.main([])          # no flag: exits before any Monsoon import / USB access
    assert e.value.code == 2
