
import pandas


from data_algebra.data_ops import *

import pytest


def test_sign_parse_1():
    d = pandas.DataFrame({
        'x': [1, 2, 3],
        'g': [False, True, False],
    })

    ops = describe_table(d, table_name='d'). \
        extend({
            'x_g': 'g.if_else(x, -3)',
            }
        )

    # we don't want this error, but it is something we can't work around
    # yet. It is also the kind of thing we are trying to avoid in the
    # test_if_else_complex() test.
    with pytest.raises(AttributeError):
        res_pandas = ops.transform(d)

    # # Pandas mis-parses -3 is unary_op(3) and then throws!
    # # TODO: look towards non-eval implementations for non-windowed situations?
    # # TODO: look into pre-landing columns in many cases?
    # res_pandas = ops.transform(d)
    #
    # expect = pandas.DataFrame({
    #     'x': [1, 2, 3],
    #     'g': [False, True, False],
    #     'x_g': [1, -3, 3],
    # })
    # assert data_algebra.test_util.equivalent_frames(expect, res_pandas)


def test_if_else_complex():
    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'g': [1, 1, 1, 2, 2, 2],
    })

    # need to block this structure, as Pandas eval fails on it
    with pytest.raises(ValueError):
        ops = describe_table(d, table_name='d') .\
            extend({'r': '(g>1).if_else(1, 2)'})
