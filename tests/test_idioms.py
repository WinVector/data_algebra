
import os
import sqlite3

from google.cloud import bigquery

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.BigQuery
import data_algebra.SQLite

import pytest


# running bigquery tests depends on environment variable
# example:
#  export GOOGLE_APPLICATION_CREDENTIALS="/Users/johnmount/big_query/big_query_jm.json"
# to clear:
#  unset GOOGLE_APPLICATION_CREDENTIALS
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
    tables_to_delete = set()
    yield {
        'bq_client': bq_client,
        'bq_handle': bq_handle,
        'data_catalog': data_catalog,
        'data_schema': data_schema,
        'tables_to_delete': tables_to_delete,
    }
    # back from yield, clean up
    if bq_client is not None:
        for tn in tables_to_delete:
            bq_handle.drop_table(tn)
    bq_handle.close()


def test_ideom_extend_one_count(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name)

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'one': 1
        }) .\
        project({
            'count': 'one.sum()'
        })
    expect = data_algebra.pd.DataFrame({
        'count': [4]
    })

    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_ideom_extend_special_count(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name)

    ops = describe_table(d, table_name=table_name) .\
        project({
            'count': '_count()'
        })
    expect = data_algebra.pd.DataFrame({
        'count': [4]
    })

    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_ideom_forbind_extend_test_trinary():
    d = data_algebra.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    # forbid this form early, as Pandas throws on ops.transform(d)
    with pytest.raises(Exception):
        ops = describe_table(d, table_name='d') .\
            extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                'select': '(val > 2.5).if_else("high", "low")'
            })


def test_ideom_extend_test_trinary(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })

    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name)

    ops = describe_table(d, table_name=table_name) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'select': '(val > 2.5)'
        }) .\
        extend({
            'select': 'select.if_else("high", "low")'
        })
    expect = data_algebra.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
        'select': ['low', 'low', 'high', 'high']
    })

    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_ideom_simulate_cross_join(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    e = data_algebra.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })

    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name_d)
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'
    tables_to_delete.add(table_name_e)

    ops = describe_table(d, table_name=table_name_d) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'one': 1
        }) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e) . \
                extend({  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    'one': 1
                }),
            by=['one'],
            jointype='left'
        ) .\
        drop_columns(['one'])
    expect = data_algebra.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    res_pandas = ops.eval({table_name_d: d , table_name_e: e})
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name_d)
        sqlite_handle.insert_table(e, table_name=table_name_e)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name_d, allow_overwrite=True)
        bq_handle.insert_table(e, table_name=table_name_e, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_ideom_simulate_cross_join_select(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    e = data_algebra.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })

    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name_d)
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'
    tables_to_delete.add(table_name_e)

    ops = describe_table(d, table_name=table_name_d) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'one': 1
        }) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e) . \
                extend({  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    'one': 1
                }),
            by=['one'],
            jointype='left'
        ) .\
        select_columns(['x', 'y'])
    expect = data_algebra.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    res_pandas = ops.eval({table_name_d: d , table_name_e: e})
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name_d)
        sqlite_handle.insert_table(e, table_name=table_name_e)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name_d, allow_overwrite=True)
        bq_handle.insert_table(e, table_name=table_name_e, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)


def test_ideom_cross_join(get_bq_handle):
    bq_client = get_bq_handle['bq_client']
    bq_handle = get_bq_handle['bq_handle']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']
    tables_to_delete = get_bq_handle['tables_to_delete']

    d = data_algebra.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    e = data_algebra.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })

    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'
    tables_to_delete.add(table_name_d)
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'
    tables_to_delete.add(table_name_e)

    ops = describe_table(d, table_name=table_name_d) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e),
            by=[],
            jointype='cross'
        )
    expect = data_algebra.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    res_pandas = ops.eval({table_name_d: d , table_name_e: e})
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    db_model = data_algebra.SQLite.SQLiteModel()
    sql = ops.to_sql(db_model, pretty=True)
    assert isinstance(sql, str)

    with db_model.db_handle(sqlite3.connect(":memory:")) as sqlite_handle:
        db_model.prepare_connection(sqlite_handle.conn)
        sqlite_handle.insert_table(d, table_name=table_name_d)
        sqlite_handle.insert_table(e, table_name=table_name_e)
        res_sqlite = sqlite_handle.read_query(ops)
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)

    bigquery_sql = bq_handle.to_sql(ops, pretty=True)
    if bq_client is not None:
        bq_handle.insert_table(d, table_name=table_name_d, allow_overwrite=True)
        bq_handle.insert_table(e, table_name=table_name_e, allow_overwrite=True)
        bigquery_res = bq_handle.read_query(bigquery_sql)
        assert data_algebra.test_util.equivalent_frames(expect, bigquery_res)
