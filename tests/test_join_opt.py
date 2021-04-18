import pytest

import sqlite3

import pandas

from data_algebra.data_ops import *
import data_algebra.SQLite

import data_algebra.test_util
import data_algebra.util


def test_join_opt_1():
    d1 = pandas.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [5, 6, 7, 8],
        'z': [9, 10, None, None]
    })
    td1 = describe_table(d1, table_name='d1')

    d2 = pandas.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [5, 6, 7, 8],
        'z': [90, None, 110, None]
    })
    td2 = describe_table(d2, table_name='d2')

    ops_sel = td1 .\
        natural_join(
            b=td2.select_columns(['x', 'z']),
            by=['x'],
            jointype='left')
    # check column control doesn't get optimized out
    assert "select_columns" in str(ops_sel)

    ops_drop = td1 .\
        natural_join(
            b=td2.drop_columns(['y']),
            by=['x'],
            jointype='left')
    # check column control doesn't get optimized out
    assert "drop_columns" in str(ops_drop)

    ops_sel_2 = td1 . \
        select_columns(['x', 'z']) .\
        natural_join(
            b=td2,
            by=['x'],
            jointype='left')
    # check column control doesn't get optimized out
    assert "select_columns" in str(ops_sel_2)

    ops_drop_2 = td1 .\
        drop_columns(['y']) .\
        natural_join(
            b=td2,
            by=['x'],
            jointype='left')
    # check column control doesn't get optimized out
    assert "drop_columns" in str(ops_drop_2)

    tables = {'d1': d1, 'd2': d2}

    ops_list = [
        ops_sel, ops_drop, ops_sel_2, ops_drop_2
    ]

    expect = pandas.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [5, 6, 7, 8],
        'z': [9, 10, 110, None]
    })

    for ops in ops_list:
        res_pandas = ops.eval(tables)
        assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    sql_model = data_algebra.SQLite.SQLiteModel()

    # see that we get exactly 2 selects in derived sql
    for ops in ops_list:
        sql = ops.to_sql(sql_model, pretty=True)
        assert sql.count('SELECT') == 2

    with sqlite3.connect(":memory:") as conn:
        sql_model.prepare_connection(conn)
        db_handle = data_algebra.db_model.DBHandle(db_model=sql_model, conn=conn)
        for k, v in tables.items():
            db_handle.insert_table(v, table_name=k)
        for ops in ops_list:
            res_db = db_handle.read_query(ops)
            assert data_algebra.test_util.equivalent_frames(res_db, expect)
