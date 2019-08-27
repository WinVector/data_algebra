
import pandas
import data_algebra.util
from data_algebra.data_ops import *


def test_equiv():
    d1 = pandas.DataFrame({'x': [1, 2], 'y': [3, 4]})
    d2 = pandas.DataFrame({'x': [1, 2], 'z': ['a', 'b']})
    assert data_algebra.util.equivalent_frames(d1, d1)
    assert data_algebra.util.equivalent_frames(d2, d2)
    assert not data_algebra.util.equivalent_frames(d1, d2)


def test_simple():
    q = 4
    x = 2
    var_name = 'y'

    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription('d', ['x', 'y']).extend({'z': '1/q + x'})

    d_local = pandas.DataFrame({'x': [1, 2], 'y': [3, 4]})
    res = ops.eval_pandas({'d': d_local})
    expect = pandas.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [1.25, 2.25]})
    assert data_algebra.util.equivalent_frames(res, expect)
