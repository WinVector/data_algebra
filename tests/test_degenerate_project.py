
import pytest

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *


def test_degenerate_project():
    d = data_algebra.pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "a", "b", "b"]})

    ops_good = describe_table(d).project({"x2": "x.max()"})
    res = ops_good.transform(d)
    expect = data_algebra.pd.DataFrame({"x2": [4],})
    assert data_algebra.test_util.equivalent_frames(res, expect)

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "x.max() + x.max()"})

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "x"})

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "1"})
