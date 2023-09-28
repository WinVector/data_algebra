import pytest

import sqlite3

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite

import data_algebra.test_util
import data_algebra.util


def test_join_opt_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8], "z": [9, 10, None, None]}
    )
    td1 = describe_table(d1, table_name="d1")

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8], "z": [90, None, 110, None]}
    )
    td2 = describe_table(d2, table_name="d2")

    ops_sel = td1.natural_join(
        b=td2.select_columns(["x", "z"]), by=["x"], jointype="left"
    )
    # check column control doesn't get optimized out
    assert "select_columns" in str(ops_sel)

    ops_drop = td1.natural_join(b=td2.drop_columns(["y"]), by=["x"], jointype="left")
    # check column control doesn't get optimized out
    assert "drop_columns" in str(ops_drop)

    ops_sel_2 = td1.select_columns(["x", "z"]).natural_join(
        b=td2, by=["x"], jointype="left"
    )
    # check column control doesn't get optimized out
    assert "select_columns" in str(ops_sel_2)

    ops_drop_2 = td1.drop_columns(["y"]).natural_join(b=td2, by=["x"], jointype="left")
    # check column control doesn't get optimized out
    assert "drop_columns" in str(ops_drop_2)

    tables = {"d1": d1, "d2": d2}

    ops_list = [ops_sel, ops_drop, ops_sel_2, ops_drop_2]

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8], "z": [9, 10, 110, None]}
    )

    for ops in ops_list:
        data_algebra.test_util.check_transform(ops=ops, data=tables, expect=expect)
