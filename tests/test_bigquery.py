
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
            {'med_val': 'mean(med_val)'},
            group_by=['group'])

    res_1 = ops.transform(d)

    expect = pandas.DataFrame({
        'group': ['a', 'b'],
        'med_val': [1.5, 3.5],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res_1)

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


