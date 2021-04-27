
import pandas

from data_algebra.data_ops import *
import data_algebra.BigQuery


def test_set_quoting_1():
    d = pandas.DataFrame({
        'x': [1, -2, 3, -4]
    })
    targets = [-2, -5]

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name='d') .\
        extend({
            'select': f'x.is_in({targets})'
        })

    sql = bq_handle.to_sql(ops, pretty=True)
    assert "'" not in sql
    assert '"' not in sql


def test_set_quoting_2():
    d = pandas.DataFrame({
        'x': [1, -2, 3, -4]
    })

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name='d') .\
        extend({
            'select': f'x.is_in({-5, 1+2})'
        })

    sql = bq_handle.to_sql(ops, pretty=True)
    assert "'" not in sql
    assert '"' not in sql
    assert '3' in sql
