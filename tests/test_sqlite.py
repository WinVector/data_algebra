from dataclasses import dataclass
import re
import sqlite3

import numpy
import pytest

import data_algebra
import data_algebra.db_model
import data_algebra.test_util
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.SparkSQL
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

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
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

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
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

    d = data_algebra.data_model.default_data_model().pd.DataFrame(
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
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"col1": [1, 2, 4], "col2": [3, None, 6], "col3": [4, 5, 7]}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
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
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"col1": [1, 2, 4], "col2": [3, None, 6], "col3": [4, 5, 7]}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
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
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [-2, -1, 0, 1, None, numpy.nan]
    })
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    ops = (
        descr(d=d)
            .extend({'xs': 'x.sign()'})
    )
    sqlite_handle.read_query(ops)
    sqlite_handle.close()


def test_sqlite_arccosh():
    sqlite_handle = data_algebra.SQLite.example_handle()
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [.1, .2, .3, .4],
    })
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    ops = (
        descr(d=d)
            .extend({'xs': 'x.arccosh()'})
    )
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(res_db, res_pandas)


def test_sqlite_median():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [.1, .2, .3, .4],
        'g': ['a', 'a', 'a', 'b'],
    })
    ops = (
        descr(d=d)
            .project(
                {'xs': 'x.median()'},
                group_by=['g'],
            )
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'xs': [0.2, 0.4],
        'g': ['a', 'b'],
        })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_int_div():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1, 2, 3, 5],
        'y': [2, 2, 3, 3],
    })
    ops = (
        descr(d=d)
            .extend({'r': 'x // y'},)
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1, 2, 3, 5],
        'y': [2, 2, 3, 3],
        'r': [0, 1, 1, 1],
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_concat():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1, 2, 3, 5],
        'y': [2, 2, 3, 3],
    })
    ops = (
        descr(d=d)
            .extend({'r': 'x %+% "_" %+% y'},)
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1, 2, 3, 5],
        'y': [2, 2, 3, 3],
        'r': ['1_2', '2_2', '3_3', '5_3'],
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_around():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1.11, 2.22, 3.33, 5.55],
    })
    ops = (
        descr(d=d)
            .extend({
                'r0': 'x.around(0)',
                'r1': 'x.around(1)',
                'r2': 'x.around(2)',
              })
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x': [1.11, 2.22, 3.33, 5.55],
        'r0': [1.0, 2.0, 3.0, 6.0],
        'r1': [1.1, 2.2, 3.3, 5.6],
        'r2': [1.11, 2.22, 3.33, 5.55],
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_any_all_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x0': [False, False],
        'x1': [False, True],
        'x2': [True, False],
        'x3': [True, True],
    })
    ops = (
        descr(d=d)
            .project({
                'r0n': 'x0.any()',
                'r0l': 'x0.all()',
                'r1n': 'x1.any()',
                'r1l': 'x1.all()',
                'r2n': 'x2.any()',
                'r2l': 'x2.all()',
                'r3n': 'x3.any()',
                'r3l': 'x3.all()',
              },
            group_by=[])
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'r0n': [False],
        'r0l': [False],
        'r1n': [True],
        'r1l': [False],
        'r2n': [True],
        'r2l': [False],
        'r3n': [True],
        'r3l': [True],
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_any_all_extend():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x0': [False, False],
        'x1': [False, True],
        'x2': [True, False],
        'x3': [True, True],
        'g': 'a',
    })
    ops = (
        descr(d=d)
            .extend({
                'r0n': 'x0.any()',
                'r0l': 'x0.all()',
                'r1n': 'x1.any()',
                'r1l': 'x1.all()',
                'r2n': 'x2.any()',
                'r2l': 'x2.all()',
                'r3n': 'x3.any()',
                'r3l': 'x3.all()',
              },
            partition_by=['g'])
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'x0': [False, False],
        'x1': [False, True],
        'x2': [True, False],
        'x3': [True, True],
        'g': 'a',
        'r0n': False,
        'r0l': False,
        'r1n': True,
        'r1l': False,
        'r2n': True,
        'r2l': False,
        'r3n': True,
        'r3l': True,
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    # # not a workable direction (composite expression in aggregate issue?)
    # sqlite_handle = data_algebra.SQLite.example_handle()
    # sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    # res_db = sqlite_handle.read_query(ops)
    # sqlite_handle.close()
    # assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_floor():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'z': [1.6, None, -2.1, 0],
    })
    ops = (
        descr(d=d)
            .extend({'r': 'z.floor()'},)
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'z': [1.6, None, -2.1, 0],
        'r': [1, None, -3, 0],
    })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    db_handle = data_algebra.SQLite.example_handle()
    # db_handle = data_algebra.SparkSQL.example_handle()
    db_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = db_handle.read_query(ops)
    db_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_sqlite_default_to_sql():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2], "z": ["q"]})
    td = describe_table(d, table_name="d")
    ops = td.rename_columns({"y1": "y", "x2": "x"})
    ops_str = ops.to_sql()
    assert isinstance(ops_str, str)
    ops_str_2 = ops.to_sql(data_algebra.SQLite.SQLiteModel())
    assert ops_str_2 == ops_str
    with data_algebra.SQLite.example_handle() as hdl:
        ops_str_3 = ops.to_sql(hdl)
    assert ops_str_3 == ops_str
