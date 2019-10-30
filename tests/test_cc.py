
import pandas

import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.connected_components import connected_components


def test_cc():
    f = [1, 4, 6, 2, 1]
    g = [2, 5, 7, 3, 7]
    res = connected_components(f, g)
    expect = [1, 4, 1, 1, 1]
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


# def test_cc3():
#     d = pandas.DataFrame({
#         'f': [1, 4, 6, 2, 1],
#         'g': [2, 5, 7, 3, 7],
#         'x': [1, 1, 1, 1, 2],
#     })
#
#     ops = describe_table(d). \
#         extend({'c': 'f.co_equalizer(g)'},
#                partition_by=['x'])
#     res = ops.transform(d)
#
#     expect = pandas.DataFrame({
#         'f': [1, 4, 6, 2, 1],
#         'g': [2, 5, 7, 3, 7],
#         'c': [1, 4, 1, 1, 1],
#         })
#
#     assert data_algebra.util.equivalent_frames(res, expect)
