
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

    is_null = lambda x: pandas.isnull(x)
    is_bad = lambda x: pandas.isnull(x)

    with data_algebra.env.Env(locals()) as env:
        d2 = ops.transform(d)
