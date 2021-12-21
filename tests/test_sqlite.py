import re
import sqlite3

import numpy

import data_algebra
import data_algebra.db_model
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.SQLite
import data_algebra.util


def test_sqlite():
    db_model = data_algebra.SQLite.SQLiteModel()
    conn = sqlite3.connect(":memory:")
    db_model.prepare_connection(conn)
    cur = conn.cursor()

    # From:
    #   https://docs.python.org/3.5/library/sqlite3.html

    # noinspection SqlNoDataSourceInspection
    cur.execute(
        """CREATE TABLE stocks
                 (date text, trans text, symbol text, qty real, price real)"""
    )

    # Insert a row of data
    # noinspection SqlNoDataSourceInspection
    cur.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    conn.commit()
    # work a simple example

    ops = TableDescription(
        table_name="stocks", column_names=["date", "trans", "symbol", "qty", "price"]
    ).extend({"cost": "qty * price"})

    pp = ops.to_python(pretty=True)

    sql = ops.to_sql(db_model)

    res = db_model.read_query(conn, sql)

    # clean up
    conn.close()

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "symbol": ["RHAT"],
            "date": ["2006-01-05"],
            "price": 35.14,
            "qty": 100.0,
            "trans": "BUY",
            "cost": 3514.0,
        }
    )

    assert data_algebra.test_util.equivalent_frames(res, expect)

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {
            "symbol": ["RHAT"],
            "date": ["2006-01-05"],
            "price": 35.14,
            "qty": 100.0,
            "trans": "BUY",
        }
    )

    res2 = ops.eval(data_map={"stocks": d2})

    assert data_algebra.test_util.equivalent_frames(res2, expect)


def test_sqllite_g2():
    ops = (
        TableDescription(table_name="d", column_names=["col1", "col2", "col3"])
        .extend({"sum23": "col2 + col3"})
        .extend({"x": 1.0})
        .extend({"x": 2.0})
        .extend({"x": 3.0})
        .extend({"x": 4.0})
        .extend({"x": 5.0})
        .project({"x": "x.max()"}, group_by=["sum23"])
        .extend(
            {"ratio": "x / sum23", "sum": "x + sum23", "diff": "x - sum23", "neg": "-x"}
        )
        .select_columns(["ratio", "sum23", "diff"])
        .select_rows("sum23 > 8")
        .drop_columns(["sum23"])
        .rename_columns({"rat": "ratio"})
        .rename_columns({"rat": "diff", "diff": "rat"})
        .order_rows(["rat"])
        .extend({"z": "-rat"})
    )

    d = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 2], "col2": [3, 4, 3], "col3": [4, 5, 7]}
    )

    res_pandas = ops.transform(d)

    sql_model = data_algebra.SQLite.SQLiteModel()

    q = ops.to_sql(db_model=sql_model)

    with sqlite3.connect(":memory:") as conn:
        sql_model.prepare_connection(conn)
        db_handle = sql_model.db_handle(conn)
        db_handle.insert_table(d, table_name="d")

        # conn.execute('CREATE TABLE res AS ' + q)
        # res_sql = sql_model.read_table(conn, 'res')
        res_sql = db_handle.read_query(q)

    assert data_algebra.test_util.equivalent_frames(
        res_pandas, res_sql, check_row_order=True
    )


def test_join_g2():
    d1 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 4], "col2": [3, None, 6], "col3": [4, 5, 7]}
    )
    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [3, 4, None], "col4": [4, 5, 7]}
    )

    ops = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=["col1"], jointype="LEFT"
    )

    res_pandas = ops.eval({"d1": d1, "d2": d2})

    sql_model = data_algebra.SQLite.SQLiteModel()

    q = ops.to_sql(db_model=sql_model)

    conn = sqlite3.connect(":memory:")
    sql_model.prepare_connection(conn)
    sql_model.insert_table(conn, d1, table_name="d1")
    sql_model.insert_table(conn, d2, table_name="d2")

    # conn.execute('CREATE TABLE res AS ' + q)
    # res_sql = sql_model.read_table(conn, 'res')
    res_sql = sql_model.read_query(conn, q)

    conn.close()

    assert data_algebra.test_util.equivalent_frames(
        res_pandas, res_sql, check_row_order=True
    )


def test_unionall_g2():
    d1 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 4], "col2": [3, None, 6], "col3": [4, 5, 7]}
    )
    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [3, 4, None], "col3": [4, 5, 7]}
    )

    ops = describe_table(d1, table_name="d1").concat_rows(
        b=describe_table(d2, table_name="d2")
    )

    res_pandas = ops.eval({"d1": d1, "d2": d2})

    sql_model = data_algebra.SQLite.SQLiteModel()

    q = ops.to_sql(db_model=sql_model)

    conn = sqlite3.connect(":memory:")
    sql_model.prepare_connection(conn)
    db_handle = data_algebra.db_model.DBHandle(db_model=sql_model, conn=conn)

    tbl_map = {
        "d1": db_handle.insert_table(d1, table_name="d1"),
        "d2": db_handle.insert_table(d2, table_name="d2"),
    }

    res_sql = sql_model.read_query(conn, q)

    assert data_algebra.test_util.equivalent_frames(
        res_pandas, res_sql, check_row_order=False
    )

    conn.close()


def test_sqlite_sign():
    sqlite_handle = data_algebra.SQLite.example_handle()
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [-2, -1, 0, 1, None, numpy.nan]
    })
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    ops = (
        descr(d=d)
            .extend({'xs': 'x.sign()'})
    )
    sqlite_handle.read_query(ops)
    sqlite_handle.close()
