
import sqlite3

import numpy
import pandas

from data_algebra.data_ops import *
import data_algebra.util
import data_algebra.test_util
import data_algebra.SQLite

import pytest

# TODO: SQL tests for all non-failing steps

def test_or_1():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        select_rows('(ID == 3) | (ID == 4)')
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [3, 4, 4, 4, 4],
        'OP': ['D', 'C', 'A', 'D', 'B'],
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)


def test_in_1():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        extend({'v': 'ID.is_in([3, 4])'})
    ops_str = str(ops)  # see if this throws
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'v': [False]*3 + [True]*5 + [False]*3,
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    with sqlite3.connect(':memory:') as con:
        db_model.prepare_connection(con)
        d.to_sql(name='d', con=con)
        res_db = pandas.read_sql(sql, con=con)

    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_in_1b():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        extend({'v': 'ID.is_in((3, 4))'})
    ops_str = str(ops)  # see if this throws
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'v': [False]*3 + [True]*5 + [False]*3,
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    with sqlite3.connect(':memory:') as con:
        db_model.prepare_connection(con)
        d.to_sql(name='d', con=con)
        res_db = pandas.read_sql(sql, con=con)

    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_in_2():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 34, 44, 44, 44, 44, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        extend({'v': 'ID.is_in([34, 44])'}). \
        select_rows('v')
    ops_str = str(ops)  # see if this throws
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [34, 44, 44, 44, 44,],
        'OP': ['D', 'C', 'A', 'D', 'B'],
        'v': [True]*5,
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert sql.find("'34'") < 0
    assert sql.find("34") > 0
    with sqlite3.connect(':memory:') as con:
        db_model.prepare_connection(con)
        d.to_sql(name='d', con=con)
        res_db = pandas.read_sql(sql, con=con)

    assert data_algebra.test_util.equivalent_frames(expect, res_db)


def test_in_3():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name='d'). \
            select_rows('ID.is_in([3, 4])')

    # d2 = ops.transform(d)  # this will throw on user fn in a confusing way, so throw where the expr is made
