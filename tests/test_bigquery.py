
import os

import pandas
from google.cloud import bigquery

import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.BigQuery
import data_algebra.SQLite

import pytest

# running bigquery tests depends on environment variable
# example:
#  export GOOGLE_APPLICATION_CREDENTIALS="/Users/johnmount/big_query/big_query_jm.json"
# to clear:
# unset GOOGLE_APPLICATION_CREDENTIALS
@pytest.fixture(scope='module')
def get_bq_handle():
    bq_client = None
    gac = None
    try:
        gac = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        bq_client = bigquery.Client()
    except KeyError:
        pass
    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(bq_client)
    data_catalog = 'data-algebra-test'
    data_schema = 'test_1'
    yield {
        'bq_client': bq_client,
        'bq_handle': bq_handle,
        'data_catalog': data_catalog,
        'data_schema': data_schema,
    }
    bq_handle.close()


def test_bigquery_1(get_bq_handle):
    d = pandas.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    table_name = f'{data_catalog}.{data_schema}.d'

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    ops = describe_table(d, table_name=table_name). \
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


    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_bigquery_2(get_bq_handle):
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
    })

    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    table_name = f'{data_catalog}.{data_schema}.d'

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    # refs on BigQuery window fn horeshit:
    #  https://iamhectorotero.github.io/median-and-group-by/
    #  https://chartio.com/resources/tutorials/how-countdistinct-field-works-in-google-bigquery/
    ops = describe_table(d, table_name=table_name). \
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

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_bigquery_date_1(get_bq_handle):
    db_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
        'dt': pandas.to_datetime([1490195805, 1490195815, 1490295805, 1490196805, 1490195835], unit='s')
    })
    d['dt_str'] = d.dt.astype(str)

    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    table_name = f'{data_catalog}.{data_schema}.d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'date': db_handle.fns.datetime_to_date('dt'),
            'date_str': db_handle.fns.trimstr('dt_str', start=0, stop=10),
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

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        # # big query adding timezones to timestamps, so can't compare
        # assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


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


def test_big_query_and(get_bq_handle):
    bigquery_handle = data_algebra.BigQuery.BigQuery_DBHandle(
        db_model=data_algebra.BigQuery.BigQueryModel(), conn=None)
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 2],
        'v2': [1, 2, 3, 4, 5],
    })

    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    table_name = f'{data_catalog}.{data_schema}.d'

    # build a description that looks like the BigQuery db handle built it.
    ops = describe_table(d, table_name=table_name) .\
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
    })
    assert data_algebra.test_util.equivalent_frames(expect, res)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_big_query_notor(get_bq_handle):
    bigquery_handle = data_algebra.BigQuery.BigQuery_DBHandle(conn=None)
    d = pandas.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 2],
        'v2': [1, 2, 3, 4, 5],
    })

    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    table_name = f'{data_catalog}.{data_schema}.d'

    # build a description that looks like the BigQuery db handle built it.
    ops = describe_table(d, table_name=table_name) .\
        select_rows("not ((group == 'a') or (v1 == 2))")

    # see & gets translated to AND
    sql = bigquery_handle.to_sql(ops)
    assert sql.find('|') < 0
    assert sql.find('OR') > 0

    # see if we can use this locally
    res = ops.transform(d)
    expect = pandas.DataFrame({
        'group': ['b'],
        'v1': [0],
        'v2': [4],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)
