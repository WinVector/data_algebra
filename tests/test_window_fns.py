
import sqlite3
import pandas
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.SQLite
import data_algebra.util


# https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html

def test_window_fns():

    d = pandas.DataFrame({
        'g': [1, 2, 2, 3, 3, 3],
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10, 40, 50, 70, 80, 90],
    })

    ops = describe_table(d). \
        extend({
            'row_number': '_row_number()',
            'max_v': 'v.max()',
            'min_v': 'v.min()',
            'sum_v': 'v.sum()',
            'mean_v': 'v.mean()',
            'shift_v': 'v.shift()',
            'count_v': 'v.count()',
            'size_v': 'v.size()',
            'ngroup_v': 'v.ngroup()',
        },
        order_by=['x'],
        partition_by=['g'])

    res1 = ops.transform(d)