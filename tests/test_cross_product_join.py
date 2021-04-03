
import sqlite3

import pandas

from data_algebra.data_ops import *

import data_algebra.SQLite
import data_algebra.test_util

import pytest


def test_cross_project_join_1():
    d1 = pandas.DataFrame({
        'x': [1, 2, 3],
    })
    d2 = pandas.DataFrame({
        'y': ['a', 'b', 'c', 'd'],
    })
    ops = describe_table(d1, table_name='d1') .\
            natural_join(b=describe_table(d2, table_name='d2'),
                         by=[],
                         jointype='CROSS')
    res_pandas = ops.eval({'d1': d1, 'd2': d2})

    expect = pandas.DataFrame({
        'x': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'y': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd',],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
 
    sql_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(sql_model)
    with sqlite3.connect(":memory:") as conn:
        sql_model.prepare_connection(conn)
        db_handle = data_algebra.db_model.DBHandle(db_model=sql_model, conn=conn)
        db_handle.insert_table(d1, table_name='d1')
        db_handle.insert_table(d2, table_name='d2')
        res_db = db_handle.read_query(sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_db)
