import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.BigQuery
import data_algebra.SQLite
import data_algebra.MySQL

import pytest


def test_idiom_count_empty_frame_a():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"group": [], "val": [],})
    table_name = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name).project({"count": "(1).sum()"})

    res = ops.transform(d)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"count": [0],})

    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_idiom_count_empty_frame_b():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"group": [], "val": [],})
    table_name = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name)
        .extend({"one": 1})
        .project({"count": "one.sum()"})
    )

    res = ops.transform(d)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"count": [0],})

    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_idiom_extend_one_count():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4],}
    )
    table_name = "pytest_temp_d"
    ops = (
        describe_table(d, table_name=table_name)
        .extend({"one": 1})
        .project({"count": "one.sum()"})
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"count": [4]})
    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, valid_for_empty=False,
    )
    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, empty_produces_empty=False,
    )


def test_idiom_extend_special_count():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4],}
    )
    table_name = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name).project({"count": "_count()"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"count": [4]})

    with pytest.warns(UserWarning):
        # warning is db adapter saying to not use this fn
        data_algebra.test_util.check_transform(
            ops=ops, data=d, expect=expect, empty_produces_empty=False,
        )


# previously forbidden
def test_idiom_forbidden_extend_test_trinary():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4],}
    )
    table_name = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name).extend(
        {  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            "select": '(val > 2.5).if_else("high", "low")'
        }
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "val": [1, 2, 3, 4],
            "select": ["low", "low", "high", "high"],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_idiom_extend_test_trinary():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4],}
    )
    table_name = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name)
        .extend({"select": "(val > 2.5)"})
        .extend({"select": 'select.if_else("high", "low")'})
    )

    db_handle = data_algebra.MySQL.MySQLModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)
    # print(sql)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "val": [1, 2, 3, 4],
            "select": ["low", "low", "high", "high"],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_idiom_extend_test_trinary_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4],}
    )
    table_name = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name).extend(
        {"select": '(val > 2.5).if_else("high", "low")'}
    )

    db_handle = data_algebra.MySQL.MySQLModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)
    # print(sql)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "val": [1, 2, 3, 4],
            "select": ["low", "low", "high", "high"],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_idiom_simulate_cross_join():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3, 4],})
    table_name_d = "pytest_temp_d"

    e = data_algebra.data_model.default_data_model().pd.DataFrame({"y": ["a", "b", "c"],})
    table_name_e = "pytest_temp_e"

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend(
            {  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                "one": 1
            }
        )
        .natural_join(
            b=describe_table(e, table_name=table_name_e).extend(
                {  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    "one": 1
                }
            ),
            by=["one"],
            jointype="left",
        )
        .drop_columns(["one"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "y": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data={table_name_d: d, table_name_e: e}, expect=expect
    )


def test_idiom_simulate_cross_join_select():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3, 4],})
    table_name_d = "pytest_temp_d"

    e = data_algebra.data_model.default_data_model().pd.DataFrame({"y": ["a", "b", "c"],})
    table_name_e = "pytest_temp_e"

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend(
            {  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                "one": 1
            }
        )
        .natural_join(
            b=describe_table(e, table_name=table_name_e).extend(
                {  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    "one": 1
                }
            ),
            by=["one"],
            jointype="left",
        )
        .select_columns(["x", "y"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "y": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data={table_name_d: d, table_name_e: e}, expect=expect
    )


def test_idiom_cross_join():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3, 4],})
    table_name_d = "pytest_temp_d"
    e = data_algebra.data_model.default_data_model().pd.DataFrame({"y": ["a", "b", "c"],})
    table_name_e = "pytest_temp_e"
    ops = describe_table(d, table_name=table_name_d).natural_join(
        b=describe_table(e, table_name=table_name_e), by=[], jointype="cross"
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "y": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"],
        }
    )
    data_algebra.test_util.check_transform(
        ops=ops, data={table_name_d: d, table_name_e: e}, expect=expect, valid_for_empty=False
    )
    data_algebra.test_util.check_transform(
        ops=ops, data={table_name_d: d, table_name_e: e}, expect=expect,
        try_on_Polars=False,  # TODO: get empty case to match and turn this on
    )


# Note: switching away from _row_number and _count
def test_idiom_row_number():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 3, 2, 4, 5], "g": [1, 2, 2, 1, 1],}
    )
    table_name_d = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend({"one": 1})
        .extend({"n": "one.cumsum()"}, partition_by=["g"], order_by=["i"],)
        .drop_columns(["one"])
        .order_rows(["i"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 2, 3, 4, 5], "g": [1, 2, 2, 1, 1], "n": [1, 1, 2, 2, 3],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


# Note: switching away from _row_number and _count
def test_idiom_row_number_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 3, 2, 4, 5], "g": [1, 2, 2, 1, 1],}
    )
    table_name_d = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend({"n": "(1).cumsum()"}, partition_by=["g"], order_by=["i"],)
        .order_rows(["i"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 2, 3, 4, 5], "g": [1, 2, 2, 1, 1], "n": [1, 1, 2, 2, 3],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


# Note: switching away from _row_number and _count
def test_idiom_row_number_3():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 3, 2, 4, 5], "g": [1, 2, 2, 1, 1],}
    )
    table_name_d = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend({"n": "_row_number()"}, partition_by=["g"], order_by=["i"],)
        .order_rows(["i"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 2, 3, 4, 5], "g": [1, 2, 2, 1, 1], "n": [1, 1, 2, 2, 3],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )


def test_idiom_sum_cumsum():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 2, 3, 4, 5], "o": [1, 1, 1, 1, 1], "g": [1, 2, 2, 1, 1],}
    )
    table_name_d = "pytest_temp_d"

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name=table_name_d).extend(
            {"s2": "o.sum()",}, partition_by=["g"], order_by=["i"],
        )

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name=table_name_d).extend(
            {"s2": "o.cumsum()",}, partition_by=["g"],
        )

    ops = (
        describe_table(d, table_name=table_name_d)
        .extend({"s": "(1).cumsum()",}, partition_by=["g"], order_by=["i"],)
        .extend(
            {
                "n": "s.max()",  # max over cumsum to get sum!
                "n2": "(1).sum()",  # no order present, so meaning is non-cumulative.
            },
            partition_by=["g"],
        )
        .order_rows(["i"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "i": [1, 2, 3, 4, 5],
            "o": [1, 1, 1, 1, 1],
            "g": [1, 2, 2, 1, 1],
            "n": [3, 2, 2, 3, 3],
            "n2": [3, 2, 2, 3, 3],
            "s": [1, 1, 2, 2, 3],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_idiom_project_sum():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"i": [1, 2, 3, 4, 5], "g": [1, 2, 2, 1, 1],}
    )
    table_name_d = "pytest_temp_d"

    ops = (
        describe_table(d, table_name=table_name_d)
        .project({"s": "(1).sum()",}, group_by=["g"],)
        .order_rows(["g"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"g": [1, 2], "s": [3, 2],})

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_idiom_concat_op():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["a", "b", "c"], "y": ["1", "2", "3"],}
    )
    table_name_d = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name_d).extend({"z": "x %+% y %+% + x"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["a", "b", "c"], "y": ["1", "2", "3"], "z": ["a1a", "b2b", "c3c"]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )


def test_idiom_coalesce_op():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["a", "b", None, None], "y": ["1", None, "3", None],}
    )
    table_name_d = "pytest_temp_d"

    ops = describe_table(d, table_name=table_name_d).extend({"z": "x %?% y"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": ["a", "b", None, None],
            "y": ["1", None, "3", None],
            "z": ["a", "b", "3", None],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )
