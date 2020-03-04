
import numpy

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util


def test_user_fn_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
        'g': [1, 1, 2, 2],
    })

    def normalize(x):
        x = x - numpy.mean(x)
        x = x/numpy.std(x)
        return x

    def sumsq(x):
        return numpy.sum(x*x)

    ops_g = describe_table(d, table_name='d'). \
        project({'x': user_fn(sumsq, 'x')},
                group_by=['g'])
    res_g = ops_g.transform(d)
    expect_g = data_algebra.default_data_model.pd.DataFrame({
        'g': [1, 2],
        'x': [5, 25],
        })
    assert data_algebra.test_util.equivalent_frames(res_g, expect_g)
