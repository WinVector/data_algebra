
import sqlite3

import pandas

import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.BigQuery
import data_algebra.SQLite


def test_bigquery_1():
    d = pandas.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    ops = describe_table(d, table_name='d'). \
        extend(
            {'med_val': 'median(val)'},
            partition_by=['group']). \
        project(
            {'med_val': 'mean(med_val)'},  # pseudo-aggregator
            group_by=['group'])

    res_1 = ops.transform(d)

    expect = pandas.DataFrame({
        'group': ['a', 'b'],
        'med_val': [1.5, 3.5],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res_1)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    bigquery_sql = ops.to_sql(bigquery_model, pretty=True)

    # run through std sqllite style code as an example
    ops_natural = describe_table(d, table_name='d'). \
        project(
            {'med_val': 'median(val)'},
            group_by=['group'])
    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, 'd')
        sqllite_sql = ops_natural.to_sql(sqllite_model, pretty=True)
        res_sqlite = sqllite_model.read_query(sqllite_conn, sqllite_sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)


def test_bigquery_2():
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
    })

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    # refs on BigQuery window fn horeshit:
    #  https://iamhectorotero.github.io/median-and-group-by/
    #  https://chartio.com/resources/tutorials/how-countdistinct-field-works-in-google-bigquery/
    ops = describe_table(d, table_name='d'). \
        extend({
            'med_1': 'v1.median()',  # median is only a window fn in Big Query
            'med_2': 'v2.median()',  # median is only a window fn in Big Query
            },
            partition_by=['group']). \
        project({
            'med_1': 'med_1.mean()',  # pseudo aggregator
            'med_2': 'med_2.mean()',  # pseudo aggregator
            'mean_1': 'v1.mean()',
            'mean_2': 'v2.mean()',
            'nu_1': 'v1.nunique()',
            'nu_2': 'v2.nunique()',
            },
            group_by=['group'])

    res_1 = ops.transform(d)

    expect = pandas.DataFrame({
        'group': ['a', 'b'],
        'med_1': [2, 0],
        'med_2': [2.0, 4.5],
        'mean_1': [1.66666666667, 0.0],
        'mean_2': [2.0, 4.5],
        'nu_1': [2, 1],
        'nu_2': [3, 2],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res_1)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    bigquery_sql = ops.to_sql(bigquery_model, pretty=True)


def test_bigquery_date_1():
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
        'dt': pandas.to_datetime([1490195805, 1490195815, 1490295805, 1490196805, 1490195835], unit='s')
    })
    d['dt_str'] = d.dt.astype(str)

    trim_0_10 = data_algebra.BigQuery.TRIMSTR(start=0, stop=10)

    ops = describe_table(d, table_name='d') .\
        extend({
            'date': data_algebra.BigQuery.DATE('dt'),
            'date_str': trim_0_10('dt_str'),
         }) . \
        extend({
            'mean_v1': 'v1.mean()',
            'count': '_size()',
            },
            partition_by='group')
    res_1 = ops.transform(d)

    expect = d.copy()
    expect['date'] = expect.dt.dt.date.copy()
    expect['date_str'] = expect.dt_str.str.slice(start=0, stop=10)
    expect['mean_v1'] = [5/3, 5/3, 5/3, 0, 0]
    expect['count'] = [3, 3, 3, 2, 2]
    assert data_algebra.test_util.equivalent_frames(expect, res_1)

    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    bigquery_sql = ops.to_sql(bigquery_model, pretty=True)

    handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=bigquery_model, conn=None)


def test_big_query_table_step():
    bigquery_model = data_algebra.BigQuery.BigQueryModel()
    handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=bigquery_model, conn=None)
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
        'dt': pandas.to_datetime([1490195805, 1490195815, 1490295805, 1490196805, 1490195835], unit='s')
    })
    # build a description that looks like the BigQuery db handle built it.
    td = describe_table(d, table_name='big.honking.dt')
    td.sql_meta = pandas.DataFrame()
    td.qualifiers['table_catalog'] = 'big'
    td.qualifiers['table_schema'] = 'honking'
    td.qualifiers['table_name'] =  'dt'
    td.qualifiers['full_name'] = td.table_name
    # see if we can use this locally
    td.transform(d)


def test_big_query_and():
    bigquery_handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=data_algebra.BigQuery.BigQueryModel(), conn=None)
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 2],
        'v2': [1, 2, 3, 4, 5],
        'dt': pandas.to_datetime([1490195805, 1490195815, 1490295805, 1490196805, 1490195835], unit='s')
    })
    # build a description that looks like the BigQuery db handle built it.
    ops = describe_table(d, table_name='d') .\
        select_rows("(group == 'a') & (v1 == 2)")

    # see & gets translated to AND
    sql = bigquery_handle.to_sql(ops)
    assert sql.find('&') < 0
    assert sql.find('AND') > 0

    # see if we can use this locally
    res = ops.transform(d)
    expect = pandas.DataFrame({
        'group': ['a', 'a'],
        'v1': [2, 2],
        'v2': [2, 3],
        'dt': pandas.to_datetime([1490195815, 1490295805], unit='s')
    })
    assert data_algebra.test_util.equivalent_frames(expect, res)

    # see if the query works in SQLite
    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, 'd')
        res_sqlite = sqllite_model.read_query(sqllite_conn, sql)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
