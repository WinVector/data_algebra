
import sqlite3

import pandas

import pytest

from data_algebra.data_ops import *
import data_algebra.test_util


def test_forbidden_calculation():
    td = TableDescription(table_name='d', column_names=['a', 'b', 'c'])

    # test using undefined column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'x'})

    # test colliding with know column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'b'})

    # test swaps don't show up in forbidden
    ops1 = td.rename_columns({'a': 'b', 'b': 'a'})
    f1 = ops1.forbidden_columns()
    assert set(f1['d']) == set()

    # test new column creation triggers forbidden annotation
    ops2 = td.rename_columns({'e': 'a'})
    f2 = ops2.forbidden_columns()
    assert set(['e']) == f2['d']

    # test merge
    ops3 = td.rename_columns({'e': 'a'}).rename_columns({'f': 'b'})
    f3 = ops3.forbidden_columns()
    assert set(['e', 'f']) == f3['d']

    # test composition
    ops4 = td.rename_columns({'e': 'a'}).rename_columns({'a': 'b'})
    f4 = ops4.forbidden_columns()
    assert set(['e']) == f4['d']


def test_calc_interface():
    ops = TableDescription(table_name='d', column_names=['a']).rename_columns({'b': 'a'})

    d_good = pandas.DataFrame({'a': [1]})
    d_bad = pandas.DataFrame({'a': [1], 'b': [2]})
    expect = pandas.DataFrame({'b': [1]})

    res1 = ops.transform(d_good)
    assert data_algebra.test_util.equivalent_frames(res1, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d_good, expect=expect)

    with pytest.raises(ValueError):
        ops.transform(d_bad)

    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d_bad, table_name='d')

    sql = ops.to_sql(db_model, pretty=True)
    res_db_bad = db_model.read_query(conn, sql)

    conn.close()
