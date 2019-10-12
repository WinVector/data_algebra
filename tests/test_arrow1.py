
import pandas

import pytest

import data_algebra.diagram
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.arrow


def test_arrow1():
    d = pandas.DataFrame({
        'g': ['a', 'b', 'b', 'c', 'c', 'c'],
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10.0, 40.0, 50.0, 70.0, 80.0, 90.0],
        'i': [True, True, False, False, False, False],
    })

    d

    table_description = TableDescription('d', ['g', 'x', 'v', 'i'])

    id_ops_a = table_description. \
        project(group_by=['g']). \
        extend({
        'ngroup': '_row_number()',
    },
        order_by=['g'])

    id_ops_a.transform(d)

    id_ops_b = table_description. \
        natural_join(id_ops_a, by=['g'], jointype='LEFT')

    id_ops_b.transform(d)

    # needs
    id_ops_b.columns_used()

    # %%

    # produced
    id_ops_b.column_names

    a1 = data_algebra.arrow.DataOpArrow(id_ops_b)

    # print(a1)

    a1.fit(d)

    # print(a1)

    # print(a1.__repr__())

    a1.transform(d)

    cols2_too_small = [c for c in (set(id_ops_b.column_names) - set(['i']))]
    ordered_ops = TableDescription('d2', cols2_too_small). \
        extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
        order_by=['x'],
        partition_by=['g'])
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
    # print(a2)

    # %%

    with pytest.raises(ValueError):
        a1 >> a2

    # %%

    cols2_too_large = id_ops_b.column_names + ['q']
    ordered_ops = TableDescription('d2', cols2_too_large). \
        extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
        order_by=['x'],
        partition_by=['g'])
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
    # print(a2)

    # %%

    with pytest.raises(ValueError):
        a1 >> a2

    # %%

    ordered_ops = TableDescription('d2', id_ops_b.column_names). \
        extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
        order_by=['x'],
        partition_by=['g'])
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
    # print(a2)

    # %%

    # print(a2)

    # %%

    # print(a1 >> a2)

    wrong_example = pandas.DataFrame({
        'g': ['a'],
        'v': [1.0],
        'x': ['b'],
        'i': [True],
        'ngroup': [1]
    })

    a2.fit(wrong_example)
    # print(a2)

    # %%

    with pytest.raises(ValueError):
        a1 >> a2

    # %%

    a2.fit(a1.transform(d))

    # %%

    unordered_ops = TableDescription('d3', ordered_ops.column_names). \
        extend({
        'size': '_size()',
        'max_v': 'v.max()',
        'min_v': 'v.min()',
        'sum_v': 'v.sum()',
        'mean_v': 'v.mean()',
        'count_v': 'v.count()',
        'size_v': 'v.size()',
    },
        partition_by=['g'])
    a3 = data_algebra.arrow.DataOpArrow(unordered_ops)
    #print(a3)

    # %%

    a3.fit(a2.transform(a1.transform(d)))

    # %%

    f0 = ( a3.transform(a2.transform(a1)) ).pipeline.__repr__()
    f1  = ( a1 >> a2 >> a3 ).pipeline.__repr__()

    assert f1 == f0

    # %%

    f2 = ( (a1 >> a2) >> a3 ).pipeline.__repr__()

    assert f2 == f1

    # %%

    f3 = ( a1 >> (a2 >> a3) ).pipeline.__repr__()

    assert f3 == f1

    # %%

    a1 >> (a2 >> a3)

    r1 = (a1 >> a2 >> a3).transform(d)

    # Python default associates left to right so this is:
    # ((d >> a1) >> a2) >> a3
    r1 = d >> a1 >> a2 >> a3

    # the preferred notation, work in operator space
    r2 = d >> (a1 >> a2 >> a3)

    assert data_algebra.util.equivalent_frames(r1, r2)

    # check pipelines compose
    p1 = a3.transform(a2.pipeline.transform(a1.pipeline)).__repr__()

    # p2 = ( a1.pipeline >> a2.pipeline >> a3.pipeline ).__rep__()
    #
    # assert p2 == p1
