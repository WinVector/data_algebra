import pytest

import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.SQLite


def test_join_check():
    d1 = data_algebra.default_data_model.pd.DataFrame({"key": ["a", "b"], "x": [1, 2],})
    td1 = describe_table(d1, table_name='d1')

    d2 = data_algebra.default_data_model.pd.DataFrame({"key": ["b", "c"], "y": [3, 4],})
    td2 = describe_table(d2, table_name='d2')

    # okay
    ops = td1.natural_join(b=td2, by=["key"], jointype='INNER')
    res_pandas = ops.eval({'d1': d1, 'd2': d2})
    expect = data_algebra.default_data_model.pd.DataFrame({"key": ["b"], "x": [2], "y": [3]})
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    # also okay
    sql_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as conn:
        sql_model.prepare_connection(conn)
        db_handle = sql_model.db_handle(conn)
        db_handle.insert_table(d1, table_name=td1.table_name)
        db_handle.insert_table(d2, table_name=td2.table_name)
        res_sql = db_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sql)

    # same table name with two different definitions error
    with pytest.raises(ValueError):
        describe_table(d1).natural_join(b=describe_table(d2), by=["key"], jointype='INNER')

    # wrong key error
    with pytest.raises(KeyError):
        td1.natural_join(b=td2, by=["keyzzz"], jointype='INNER')

    # wrong table defs error
    with pytest.raises(ValueError):
        ops.eval({'d1': d2, 'd2': d1})

    # bad join type error
    with pytest.raises(KeyError):
        td1.natural_join(b=td2, by=["key"], jointype='weird')
