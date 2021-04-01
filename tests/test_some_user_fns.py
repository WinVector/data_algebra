

import pandas

from data_algebra.data_ops import *

import data_algebra.BigQuery
import data_algebra.test_util


def test_TRIMSTR():
    trim_0_5 = data_algebra.BigQuery.TRIMSTR(start=0, stop=5)

    d = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': trim_0_5('x')
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
        'nx': ['01234', 'abcde'],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=bigquery_model, conn=None)
    bigquery_sql = handle.to_sql(ops)


def test_AS_INT64():
    d = pandas.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': data_algebra.BigQuery.AS_INT64('x')
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
        'nx': [123456, 66]
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=bigquery_model, conn=None)
    bigquery_sql = handle.to_sql(ops)


def test_DATE():
    d = pandas.DataFrame({
        'x': pandas.to_datetime([1490196805, 1490195835], unit='s'),
        'y': ['012345', '77'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': data_algebra.BigQuery.DATE('x')
        })
    res = ops.transform(d)

    expect = d.copy()
    expect['nx'] = expect.x.dt.date.copy()
    assert data_algebra.test_util.equivalent_frames(res, expect)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=bigquery_model, conn=None)
    bigquery_sql = handle.to_sql(ops)

