
import math
import numpy
import sqlite3

import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite

def test_null_bad():
    ops = TableDescription("d", ["x"]).extend({
        "x_is_null": "x.is_null()",
        "x_is_bad": "x.is_bad()"
    })

    d = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 0]
    })

    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 0],
        'x_is_null': [False, True, False, False, True, False],
        'x_is_bad': [False, True, True, True, True, False]
    })

    assert all(d2['x_is_null'] == expect['x_is_null'])
    assert all(d2['x_is_bad'] == expect['x_is_bad'])

    db_model = data_algebra.SQLite.SQLiteModel()

    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    conn = sqlite3.connect(":memory:")
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d, 'd')

    res = db_model.read_query(conn, sql)

    conn.close()

    expectr = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 0],
        'x_is_null': [0, 1, 0, 0, 1, 0],
        'x_is_bad': [0, 1, 1, 1, 1, 0]
    })

    assert all(res['x_is_null'] == expectr['x_is_null'])
    assert all(res['x_is_bad'] == expectr['x_is_bad'])
