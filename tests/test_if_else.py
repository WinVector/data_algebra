
import sqlite3

import pandas

from data_algebra.data_ops import *
from data_algebra.SQLite import SQLiteModel
import data_algebra.util


def test_if_else():
    d = pandas.DataFrame({
        'a': [True, False],
        'b': [1 ,2],
        'c': [3, 4]
    })

    ops = TableDescription('d', ['a', 'b', 'c']). \
        extend({'d': 'a.if_else(b, c)'})

    expect = pandas.DataFrame({
        'a': [True, False],
        'b': [1, 2],
        'c': [3, 4],
        'd': [1, 4],
        })

    res_pandas = ops.transform(d)

    assert data_algebra.util.equivalent_frames(res_pandas, expect)

    db_model = SQLiteModel()

    ops_sql = ops.to_sql(db_model)

    conn = sqlite3.connect(":memory:")
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d, 'd')

    res_db = db_model.read_query(conn, ops_sql)

    conn.close()

    assert data_algebra.util.equivalent_frames(res_db, expect)
