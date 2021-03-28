import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite
import sqlite3


def test_natural_join_columns():
    db_model = data_algebra.SQLite.SQLiteModel()

    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]}
    )

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"qq": [10, 20, 30], "y": [1.0, 2.0, 3.0], "x": [4, 5, 7]}
    )

    ops4 = describe_table(d, "d").natural_join(
        b=describe_table(d2, "d2"), by=["y"], jointype="LEFT"
    )

    res_pandas = ops4.eval(data_map={"d": d, "d2": d2})

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 7.0],
            "y": [1.0, 2.0, numpy.nan, 3.0],
            "qq": [10.0, 20.0, numpy.nan, 30.0],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    sql = ops4.to_sql(db_model, pretty=True)

    conn = sqlite3.connect(":memory:")
    db_model.insert_table(conn, d, "d")
    db_model.insert_table(conn, d2, "d2")
    res_sqlite = db_model.read_query(conn, sql)
    # neaten up
    conn.close()

    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
