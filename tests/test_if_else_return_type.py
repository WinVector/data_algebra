
import numpy

import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util


def test_if_else_return_type():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [True, False, None],
    })
    ops = (
        descr(d=d)
            .extend({
                'w': 'x.where(1.0, 2.0)',
                'i': 'x.if_else(1.0, 2.0)',
            })
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'x': [True, False, None],
        'w': [1.0, 2.0, 2.0],
        'i': [1.0, 2.0, numpy.nan],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    assert str(res['w'].dtype) == 'float64'
    assert str(res['i'].dtype) == 'float64'
