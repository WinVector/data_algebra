
import pandas


from data_algebra.data_ops import *
import data_algebra.test_util


def test_mimimum_1():
    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'g': [1, 1, 1, 2, 2, 2],
    })

    ops = describe_table(d, table_name='d') .\
        extend({
            'x_g_max': 'x.max()',
              },
            partition_by=['g']
            ) .\
        extend({
            'xl': 'x.minimum(x_g_max - 1)'
        })

    res_pandas = ops.transform(d)  # throws deep in Pandas!

    expect = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'g': [1, 1, 1, 2, 2, 2],
        'x_g_max': [3, 3, 3, 6, 6, 6],
        'xl': [1, 2, 2, 4, 5, 5],
    })

    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
