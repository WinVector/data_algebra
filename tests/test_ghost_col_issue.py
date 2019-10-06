
import pandas

from data_algebra.data_ops import *
import data_algebra.util

def test_ghost_col_issue():
    d2 = pandas.DataFrame({
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10, 40, 50, 70, 80, 90],
        'g': [1, 2, 2, 3, 3, 3],
        'ngroup': [1, 2, 2, 3, 3, 3],
        'row_number': [1, 1, 2, 1, 2, 3],
        'shift_v': [None, None, 40.0, None, 70.0, 80.0],
    })
    o2 = describe_table(d2).extend(
        {'size': '_size()', 'max_v': 'v.max()', 'min_v': 'v.min()', 'sum_v': 'v.sum()', 'mean_v': 'v.mean()',
         'count_v': 'v.count()', 'size_v': 'v.size()'}, partition_by=['g'])
    res = o2.transform(d2)
    expect = pandas.DataFrame({
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10, 40, 50, 70, 80, 90],
        'g': [1, 2, 2, 3, 3, 3],
        'ngroup': [1, 2, 2, 3, 3, 3],
        'row_number': [1, 1, 2, 1, 2, 3],
        'shift_v': [None, None, 40.0, None, 70.0, 80.0],
        'size': [1, 2, 2, 3, 3, 3],
        'max_v': [10, 50, 50, 90, 90, 90],
        'min_v': [10, 40, 40, 70, 70, 70],
        'sum_v': [10, 90, 90, 240, 240, 240],
        'mean_v': [10, 45, 45, 80, 80, 80],
        'count_v': [1, 2, 2, 3, 3, 3],
        'size_v': [1, 2, 2, 3, 3, 3],
        })
    assert data_algebra.util.equivalent_frames(res, expect)