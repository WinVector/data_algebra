import math
import numpy
import sqlite3

import data_algebra
import data_algebra.util
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.PostgreSQL


def test_null_bad():
    ops = TableDescription("d", ["x"]).extend(
            {
                "x_is_null": "x.is_null()",
                "x_is_bad": "x.is_bad()"}
        ) .\
        drop_columns(['x'])

    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, numpy.nan, math.inf, -math.inf, None, 0]}
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "x_is_null": [False, True, False, False, True, False],
            "x_is_bad": [False, True, True, True, True, False],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        allow_pretty=False)  # pretty printer was changing capitalization of data to DATA
