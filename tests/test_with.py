import numpy

import pytest
import sqlite3

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util
import data_algebra.SQLite


def test_with_query_example_1():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2, 3]})
    ops = (
        describe_table(d, table_name="d")
        .extend({"z": "x + 1"})
        .extend({"q": "z + 2"})
        .extend({"h": "q + 3"})
    )

    res_pandas = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3], "z": [2, 3, 4], "q": [4, 5, 6], "h": [7, 8, 9]}
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = data_algebra.SQLite.SQLiteModel()
    db_handle = db_model.db_handle(conn=None)
    example_sql = db_handle.to_sql(ops, use_with=True)
    assert isinstance(example_sql, str)

    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_handle = db_model.db_handle(conn)
        db_handle.insert_table(d, table_name="d")
        sql_regular = db_handle.to_sql(
            ops, use_with=False, annotate=False
        )
        res_regular = db_handle.read_query(sql_regular)
        sql_regular_a = db_handle.to_sql(
            ops, use_with=False, annotate=True
        )
        res_regular_a = db_handle.read_query(sql_regular_a)
        sql_with = db_handle.to_sql(
            ops, use_with=True, annotate=False
        )
        res_with = db_handle.read_query(sql_with)
        sql_with_a = db_handle.to_sql(
            ops, use_with=True, annotate=True
        )
        res_with_a = db_handle.read_query(sql_with_a)

    assert data_algebra.test_util.equivalent_frames(res_regular, expect)
    assert data_algebra.test_util.equivalent_frames(res_with, expect)
    assert data_algebra.test_util.equivalent_frames(res_regular_a, expect)
    assert data_algebra.test_util.equivalent_frames(res_with_a, expect)
    assert "--" in sql_regular_a
    assert "--" in sql_with_a
    assert "--" not in sql_regular
    assert "--" not in sql_with


def test_with_query_example_2():
    d1 = data_algebra.default_data_model.pd.DataFrame(
        {"k": [1, 2, 3], "x": [5, 10, 15],}
    )

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"k": [1, 2, 3], "y": [-3, 2, 1],}
    )

    ops = (
        describe_table(d1, table_name="d1")
        .extend({"z": "x + 1"})
        .natural_join(
            b=describe_table(d2, table_name="d2").extend({"q": "y - 1"}),
            by=["k"],
            jointype="left",
        )
        .extend({"m": "(x + y) / 2"})
    )

    res_pandas = ops.eval({"d1": d1, "d2": d2})

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "k": [1, 2, 3],
            "x": [5, 10, 15],
            "z": [6, 11, 16],
            "y": [-3, 2, 1],
            "q": [-4, 1, 0],
            "m": [1.0, 6.0, 8.0],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = data_algebra.SQLite.SQLiteModel()

    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_handle = db_model.db_handle(conn)
        db_handle.insert_table(d1, table_name="d1")
        db_handle.insert_table(d2, table_name="d2")
        sql_regular = db_handle.to_sql(
            ops, use_with=False, annotate=False
        )
        res_regular = db_handle.read_query(sql_regular)
        sql_regular_a = db_handle.to_sql(
            ops, use_with=False, annotate=True
        )
        res_regular_a = db_handle.read_query(sql_regular_a)
        sql_with = db_handle.to_sql(
            ops, use_with=True, annotate=False
        )
        res_with = db_handle.read_query(sql_with)
        sql_with_a = db_handle.to_sql(
            ops, use_with=True, annotate=True
        )
        res_with_a = db_handle.read_query(sql_with_a)

    assert data_algebra.test_util.equivalent_frames(res_regular, expect)
    assert data_algebra.test_util.equivalent_frames(res_with, expect)
    assert data_algebra.test_util.equivalent_frames(res_regular_a, expect)
    assert data_algebra.test_util.equivalent_frames(res_with_a, expect)
    assert "--" in sql_regular_a
    assert "--" in sql_with_a
    assert "--" not in sql_regular
    assert "--" not in sql_with
