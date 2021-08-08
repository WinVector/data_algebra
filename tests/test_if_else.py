import sqlite3

import numpy

import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.SQLite import SQLiteModel


import data_algebra
import data_algebra.util

import pytest


def test_if_else():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4]}
    )

    ops = TableDescription(table_name="d", column_names=["a", "b", "c"]).extend(
        {"d": "a.if_else(b, c)"}
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [1, 4],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_if_else_2():
    d = data_algebra.default_data_model.pd.DataFrame(
        {
            "group": ["A", "B", None, "A", None, "C"],
            "c1": [1, 2, 3, 4, 5, 6],
            "c2": [-1, -2, -3, -4, -5, -6],
        }
    )

    # ! not used
    with pytest.raises(Exception):
        ops = (
            describe_table(d, table_name="d")
            .extend({"choice": "group=='A'"})
            .extend({"choice_fixed": "choice.is_null().if_else(False, choice)"})
            .extend({"not_c_2": "! choice_fixed"})
        )

    # ~ not used
    with pytest.raises(Exception):
        ops = (
            describe_table(d, table_name="d")
            .extend({"choice": "group=='A'"})
            .extend({"choice_fixed": "choice.is_null().if_else(False, choice)"})
            .extend({"not_c_2": "~ choice_fixed"})
        )

    ops = (
        describe_table(d, table_name="d")
        .extend({"choice": "group=='A'"})
        .extend({"choice_fixed": "choice.is_null().if_else(False, choice)"})
        .extend({"not_c_1": "choice_fixed == False"})
        .extend({"rc": "choice_fixed.if_else(c1, c2)"})
        .select_columns(["choice_fixed", "rc", "not_c_1"])
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "choice_fixed": [1, 0, 0, 1, 0, 0],
            "rc": [1, -2, -3, 4, -5, -6],
            "not_c_1": [False, True, True, False, True, True],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_maximum_1():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"a": [1, 2, 3, 5], "b": [-1, 3, -7, 6],}
    )

    ops = describe_table(d, table_name="d").extend(
        {"c": "a.maximum(b)", "d": "a.minimum(b)"}
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"a": [1, 2, 3, 5], "b": [-1, 3, -7, 6],}
    )
    expect["c"] = numpy.maximum(expect.a, expect.b)
    expect["d"] = numpy.minimum(expect.a, expect.b)

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_if_else_complex():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"a": [-4.0, 2.0], "b": [1.0, 2.0], "c": [3.0, 4.0]}
    )

    ops = describe_table(d, table_name="d").extend(
        {"d": "((a + 2).sign() > 0).if_else(b+1, c-2)"}
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"a": [-4.0, 2.0], "b": [1.0, 2.0], "c": [3.0, 4.0], "d": [1.0, 3.0]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
