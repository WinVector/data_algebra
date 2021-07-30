import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util


def test_coalesce_one():
    d = data_algebra.default_data_model.pd.DataFrame({
        'a': range(5),
        'b': 0,
        'c': 1,
        'u': 2,
        'v': 3,
        'w': 4,
        'x': 5,
        'y': 6,
        'z': 7,
    })
    ops = (
        describe_table(d, table_name='d')
            .extend({
                    'res': 'a*b*c + u*v*w + x*y*z'
                })
    )
    expect = data_algebra.default_data_model.pd.DataFrame({
        'a': [0, 1, 2, 3, 4],
        'b': [0, 0, 0, 0, 0],
        'c': [1, 1, 1, 1, 1],
        'u': [2, 2, 2, 2, 2],
        'v': [3, 3, 3, 3, 3],
        'w': [4, 4, 4, 4, 4],
        'x': [5, 5, 5, 5, 5],
        'y': [6, 6, 6, 6, 6],
        'z': [7, 7, 7, 7, 7],
        'res': [234, 234, 234, 234, 234],
        })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
