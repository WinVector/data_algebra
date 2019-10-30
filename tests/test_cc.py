
import pandas

import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.connected_components import connected_components, partitioned_connected_components


def test_cc():
    f = [1, 4, 6, 2, 1]
    g = [2, 5, 7, 3, 7]
    res = connected_components(f, g)
    expect = [1, 4, 1, 1, 1]
    assert res == expect

def test_cc_partitioned():
    f = [1, 4, 6, 2, 1]
    g = [2, 5, 7, 3, 7]
    p = [1, 2, 1, 2, 1]
    res = partitioned_connected_components(f, g, p)
    expect = [1, 4, 1, 2, 1]
    assert res == expect


def test_cc2():
    d = pandas.DataFrame({
        'f': [1, 4, 6, 2, 1],
        'g': [2, 5, 7, 3, 7],
    })

    ops = describe_table(d). \
        extend({'c': 'f.co_equalizer(g)'})
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'f': [1, 4, 6, 2, 1],
        'g': [2, 5, 7, 3, 7],
        'c': [1, 4, 1, 1, 1],
        })

    assert data_algebra.util.equivalent_frames(res, expect)
