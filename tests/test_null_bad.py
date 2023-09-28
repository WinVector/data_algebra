
import math
import numpy

import pytest

import data_algebra
import data_algebra.util
import data_algebra.test_util
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex


def test_null_bad():
    ops = (
        TableDescription(table_name="d", column_names=["x"])
        .extend({"x_is_null": "x.is_null()", "x_is_bad": "x.is_bad()"})
        .drop_columns(["x"])
    )

    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, numpy.nan, math.inf, -math.inf, None, 0]}
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x_is_null": [False, True, False, False, True, False],
            "x_is_bad": [False, True, True, True, True, False],
        }
    )

    models_to_skip = set()
    models_to_skip.add(
        str(data_algebra.MySQL.MySQLModel())
    )  # can't insert infinity into MySQL
    models_to_skip.add(
        str(data_algebra.SparkSQL.SparkSQLModel())
    )  # None/Null/Non handled differently
    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, models_to_skip=models_to_skip,
    )


def test_null_bad_no_compare():
    # similar in intent to not allowing None in sets in this grammar
    # Null/None/NaN should be checked by a method, not an expression
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, numpy.nan, math.inf, -math.inf, None, 0]}
    )

    # good, compare x to number
    ops = describe_table(d, table_name="d").extend({"c": "x == 5"})

    # good, check if x is missing
    ops = describe_table(d, table_name="d").extend({"c": "x.is_null()"})

    with pytest.raises(Exception):
        # bad, compare to None
        ops = describe_table(d, table_name="d").extend({"c": "x != None"})


def test_is_inf():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'a': [1.0, numpy.inf, numpy.nan, None, 0.0, -1.0, -numpy.inf],
    })
    ops = (
        descr(d=d)
            .extend({
                'is_inf': 'a.is_inf().if_else(1, 0)',
                'is_nan': 'a.is_nan().if_else(1, 0)',
                'is_bad': 'a.is_bad().if_else(1, 0)',
                'is_null': 'a.is_null().if_else(1, 0)',
                })
    )
    res_pandas = ops.transform(d)
    expect = pd.DataFrame({
        'a': [1.0, numpy.inf, numpy.nan, None, 0.0, -1.0, -numpy.inf],
        'is_inf': [0, 1, 0, 0, 0, 0, 1],
        'is_nan': [0, 0, 1, 1, 0, 0, 0],
        'is_bad': [0, 1, 1, 1, 0, 0, 1],
        'is_null': [0, 0, 1, 1, 0, 0, 0],   # Pandas can't represent the difference, so this is the wrong answer in general
        })
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_sqlite = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        try_on_DBs=False,
        models_to_skip={
            data_algebra.MySQL.MySQLModel(),  # sqlalchemy won't insert inf
            data_algebra.SparkSQL.SparkSQLModel(),  # probably not inserting values
            },
        try_on_Polars=False,  # Polars is correct in not confusing null and Nan, so excluding Polars on this example
        )


def test_is_null():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'a': ["a", None],
    })
    ops = (
        descr(d=d)
            .extend({
                'is_null': 'a.is_null().if_else(1, 0)',
                })
    )
    res_pandas = ops.transform(d)
    expect = pd.DataFrame({
        'a': ["a", None],
        'is_null': [0, 1],
    })
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_sqlite = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        )
