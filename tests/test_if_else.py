

import sqlite3

import numpy

import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.SQLite import SQLiteModel
import data_algebra.util

import pytest


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
    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['A', 'B', None, 'A', numpy.NAN, 'C'],
        'c1': [1, 2, 3, 4, 5, 6],
        'c2': [-1, -2, -3, -4, -5, -6],
    })

    # ! not used
    with pytest.raises(Exception):
        ops = describe_table(d, table_name='d'). \
            extend({'choice': "group=='A'"}). \
            extend({'choice_fixed': 'choice.is_bad().if_else(0, choice)'}). \
            extend({'not_c_2': '! choice_fixed'})

    # ~ not used
    with pytest.raises(Exception):
        ops = describe_table(d, table_name='d'). \
            extend({'choice': "group=='A'"}). \
            extend({'choice_fixed': 'choice.is_bad().if_else(0, choice)'}). \
            extend({'not_c_2': '~ choice_fixed'})

    ops = describe_table(d, table_name='d'). \
        extend({'choice': "group=='A'"}). \
        extend({'choice_fixed': 'choice.is_bad().if_else(0, choice)'}). \
        extend({'not_c_1': 'choice_fixed == False'}). \
        extend({'rc': 'choice_fixed.if_else(c1, c2)'}). \
        select_columns(['choice_fixed', 'rc', 'not_c_1'])

    res1 = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'choice_fixed': [1, 0, 0, 1, 0, 0],
        'rc': [1, -2, -3, 4, -5, -6],
        'not_c_1': [False, True, True, False, True, True],
    })

    assert data_algebra.test_util.equivalent_frames(expect, res1)

    db_model = SQLiteModel()

    sql = ops.to_sql(db_model, pretty=True)

    with sqlite3.connect(':memory:') as con:
        db_model.prepare_connection(con)
        d.to_sql(name='d', con=con)
        res2 = data_algebra.default_data_model.pd.read_sql(sql, con=con)

    assert data_algebra.test_util.equivalent_frames(res1, res2)


def test_maximum_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        "a": [1, 2, 3, 5],
        "b": [-1, 3, -7, 6],
    })

    ops = describe_table(d, table_name='d') .\
        extend({
            "c": "a.maximum(b)",
            "d": "a.minimum(b)"})

    res_pandas = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({
        "a": [1, 2, 3, 5],
        "b": [-1, 3, -7, 6],
    })
    expect['c'] = numpy.maximum(expect.a, expect.b)
    expect['d'] = numpy.minimum(expect.a, expect.b)

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = SQLiteModel()
    ops_sql = ops.to_sql(db_model)
    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_model.insert_table(conn, d, "d")
        res_db = db_model.read_query(conn, ops_sql)

    assert data_algebra.test_util.equivalent_frames(res_db, expect)


def test_if_else_complex():
    d = data_algebra.default_data_model.pd.DataFrame({
        "a": [-4, 2],
        "b": [1, 2],
        "c": [3, 4]}
    )

    ops = describe_table(d, table_name='d') .\
        extend({"d": "((a + 2).sign() > 0).if_else(b+1, c-2)"})

    res_pandas = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({
        "a": [-4, 2],
        "b": [1, 2],
        "c": [3, 4],
        "d": [1, 3]}
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = SQLiteModel()

    ops_sql = ops.to_sql(db_model)

    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_model.insert_table(conn, d, "d")
        res_db = db_model.read_query(conn, ops_sql)

    assert data_algebra.test_util.equivalent_frames(res_db, expect)
