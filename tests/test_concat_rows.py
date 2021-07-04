import numpy
import sqlite3

import data_algebra
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util


def test_concat_rows():
    db_model = data_algebra.SQLite.SQLiteModel()

    d1 = data_algebra.default_data_model.pd.DataFrame(
        {"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]}
    )

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"x": [0, 4, None], "y": [2, 7, None]}
    )

    ops4 = describe_table(d1, "d1").concat_rows(b=describe_table(d2, "d2"))

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, None, 0.0, 4.0, None],
            "y": [1.0, 2.0, None, 3.0, 2.0, 7.0, None],
            "source_name": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops4,
        data={'d1': d1, 'd2': d2},
        expect=expect)
