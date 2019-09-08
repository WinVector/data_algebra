
import numpy
import numpy
import math

import data_algebra.util
from data_algebra.data_ops import *


def test_null_bad():
    ops =  TableDescription("d", ["x"]).extend({
        "x_is_null": "x.is_null()",
        "x_is_bad": "x.is_bad()"
    })

    d = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 2]
    })

    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 2],
        'x_is_null': [False, True, False, False, True, False],
        'x_is_bad': [False, True, True, True, True, False]
    })

    assert all(d2['x_is_null'] == expect['x_is_null'])
    assert all(d2['x_is_bad'] == expect['x_is_bad'])