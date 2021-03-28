

import sqlite3

import numpy
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.SQLite import SQLiteModel
import data_algebra.util


def test_if_else():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4]}
    )

    ops = TableDescription("d", ["a", "b", "c"]).extend({"d": "a.if_else(b, c)"})

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [1, 4],}
    )

    res_pandas = ops.transform(d)

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = SQLiteModel()

    ops_sql = ops.to_sql(db_model)

    conn = sqlite3.connect(":memory:")
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d, "d")

    res_db = db_model.read_query(conn, ops_sql)

    conn.close()

    assert data_algebra.test_util.equivalent_frames(res_db, expect)


def test_if_else_2():
    d = pandas.DataFrame({
        'group': ['A', 'B', None, 'A', numpy.NAN, 'C'],
        'c1': [1, 2, 3, 4, 5, 6],
        'c2': [-1, -2, -3, -4, -5, -6],
    })

    ops = describe_table(d, table_name='d'). \
        extend({'choice': "group=='A'"}). \
        extend({'choice_fixed': 'choice.is_bad().if_else(0, choice)'}). \
        extend({'rc': 'choice_fixed.if_else(c1, c2)'}). \
        select_columns(['choice_fixed', 'rc'])

    res1 = ops.transform(d)

    expect = pandas.DataFrame({
        'choice_fixed': [1, 0, 0, 1, 0, 0],
        'rc': [1, -2, -3, 4, -5, -6]
    })

    assert data_algebra.test_util.equivalent_frames(expect, res1)

    db_model = SQLiteModel()

    sql = ops.to_sql(db_model, pretty=True)

    with sqlite3.connect(':memory:') as con:
        db_model.prepare_connection(con)
        d.to_sql(name='d', con=con)
        res2 = pandas.read_sql(sql, con=con)

    assert data_algebra.test_util.equivalent_frames(res1, res2)
