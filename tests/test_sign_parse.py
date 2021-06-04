
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
