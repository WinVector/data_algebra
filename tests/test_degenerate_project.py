
import pandas

import pytest

import data_algebra.util
from data_algebra.data_ops import *

def test_degenerate_project():
    d = pandas.DataFrame({
        'x': [1, 2, 3, 4],
        'y': ['a', 'a', 'b', 'b']
    })

    ops_good = describe_table(d). \
        project({'x2': 'x.max()'})
    res = ops_good.transform(d)
    expect = pandas.DataFrame({
        'x2': [4],
        })
    assert data_algebra.util.equivalent_frames(res, expect)

    with pytest.raises(Exception):
        ops_bad = describe_table(d). \
            project({'x2': 'x'})
        ops_bad.transform(d)
