
import datetime
import sqlite3

import pandas

from data_algebra.data_ops import *

import data_algebra.SQLite
import data_algebra.BigQuery
import data_algebra.test_util


def test_TRIMSTR():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': db_handle.fns.trimstr('x', start=0, stop=5)
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
        'nx': ['01234', 'abcde'],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops)

    # see if the query works in SQLite
    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, 'd')
        res_sqlite = sqllite_model.read_query(sqllite_conn, bigquery_sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)


def test_AS_INT64():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': db_handle.fns.as_int64('x')
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
        'nx': [123456, 66]
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops)

    # see if the query works in SQLite
    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, 'd')
        res_sqlite = sqllite_model.read_query(sqllite_conn, bigquery_sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)


def test_DATE():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': pandas.to_datetime([1490196805, 1490195835], unit='s'),
        'y': ['012345', '77'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': db_handle.fns.datetime_to_date('x')
        })
    res = ops.transform(d)

    expect = d.copy()
    expect['nx'] = expect.x.dt.date.copy()
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops)
    # can't test on SQLite as SQLite loses date types


def test_COALESCE_0():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': [1, None, 3]
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': db_handle.fns.coalesce_0('x')
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': [1, None, 3],
        'nx': [1, 0, 3]
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops)

    # see if the query works in SQLite
    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, 'd')
        res_sqlite = sqllite_model.read_query(sqllite_conn, bigquery_sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)


def test_PARSE_DATE():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': ['2001-01-01', '2020-04-02']
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': db_handle.fns.parse_date('x')
        })
    res = ops.transform(d)
    assert isinstance(res.nx[0], datetime.date)

    expect = pandas.DataFrame({
        'x': ['2001-01-01', '2020-04-02']
    })
    expect['nx'] = pandas.to_datetime(d.x, format="%Y-%m-%d")
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops)


def test_DATE_PARTS():
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'x': ['2001-01-01', '2020-04-02'],
        't': ['2001-01-01 01:33:22', '2020-04-02 13:11:10'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
            'nx': db_handle.fns.parse_date('x', format="%Y-%m-%d"),
            'nt': db_handle.fns.parse_datetime('x', format="%Y-%m-%d %H:%M:%S"),
        }) .\
        extend({
            'date2': db_handle.fns.datetime_to_date('nt'),
            'day_of_week': db_handle.fns.dayofweek('nx'),
            'day_of_year': db_handle.fns.dayofyear('nx'),
            'month': db_handle.fns.month('nx'),
            'day_of_month': db_handle.fns.dayofmonth('nx'),
            'quarter': db_handle.fns.quarter('nx'),
            'year': db_handle.fns.year('nx'),
        })
    res = ops.transform(d)
    assert isinstance(res.nx[0], datetime.date)

    expect = pandas.DataFrame({
        'x': ['2001-01-01', '2020-04-02'],
        't': ['2001-01-01 01:33:22', '2020-04-02 13:11:10'],
        'day_of_week': [1, 4],
        'day_of_year': [1, 93],
        'month': [1, 4],
        'day_of_month': [1, 2],
        'quarter': [1, 2],
        'year': [2001, 2020],
    })
    expect['nx'] = pandas.to_datetime(d.x, format="%Y-%m-%d").dt.date.copy()
    expect['nt'] = pandas.to_datetime(d.x, format="%Y-%m-%d %H:%M:%S")
    expect['date2'] = expect.nt.dt.date.copy()
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_sql = db_handle.to_sql(ops, pretty=True)
