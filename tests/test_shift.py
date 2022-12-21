
import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import descr
import pytest


def test_shift():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'v': ['a', 'b', 'c', 'd', 'e'],
    })
    ops = (
        descr(d=d)
            .extend({
                'v_m2': 'v.shift(-2)',
                'v_m1': 'v.shift(-1)',
                'v_s': 'v.shift()',
                'v_p1': 'v.shift(1)',
                'v_p2': 'v.shift(2)',
                },
                order_by=['x']
                )
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'v': ['a', 'b', 'c', 'd', 'e'],
        'v_m2': ['c', 'd', 'e', None, None],
        'v_m1': ['b', 'c', 'd', 'e', None],
        'v_s': [None, 'a', 'b', 'c', 'd'],
        'v_p1': [None, 'a', 'b', 'c', 'd'],
        'v_p2': [None, None, 'a', 'b', 'c'],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect,
    )


def test_shift_assert_on_0():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'v': ['a', 'b', 'c', 'd', 'e'],
    })
    with pytest.raises(ValueError):
        # shift by zero not allowed
        ops = (
            descr(d=d)
                .extend({
                    'v_s0': 'v.shift(0)',
                    },
                    order_by=['x']
                    )
        )
