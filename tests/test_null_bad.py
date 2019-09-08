
import math
import numpy
import sqlite3

import psycopg2

import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.PostgreSQL

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

    expect_r = pandas.DataFrame({
        'x': [1, numpy.nan, math.inf, -math.inf, None, 0],
        'x_is_null': [0, 1, 0, 0, 1, 0],
        'x_is_bad': [0, 1, 1, 1, 1, 0]
    })

    assert all(res['x_is_null'] == expect_r['x_is_null'])
    assert all(res['x_is_bad'] == expect_r['x_is_bad'])

    # can't copy NA/None into db through current path
    d = pandas.DataFrame({
        'x': [1, 2, math.inf, -math.inf, 2, 0]
    })

    db_model_p = data_algebra.PostgreSQL.PostgreSQLModel()
    sql_p = ops.to_sql(db_model_p, pretty=True)

    test_postgresql = False
    if test_postgresql:
        conn_p = psycopg2.connect(
            database="johnmount",
            user="johnmount",
            host="localhost",
            password=""
        )
        conn_p.autocommit=True

        db_model_p.insert_table(conn_p, d, 'd')

        res_p = db_model_p.read_query(conn_p, sql_p)

        conn_p.close()

        expect_p = pandas.DataFrame({
            'x': [1, numpy.nan, math.inf, -math.inf, None, 0],
            'x_is_null': [False, False, False, False, False, False],
            'x_is_bad': [False, False, True, True, False, False]
        })

        assert all(res_p['x_is_null'] == expect_p['x_is_null'])
        assert all(res_p['x_is_bad'] == expect_p['x_is_bad'])
