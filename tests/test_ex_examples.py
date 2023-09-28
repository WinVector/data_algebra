import pytest

import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex, table
import data_algebra.util
import data_algebra.SQLite
import data_algebra.test_util


def test_ex_example_ex_descr():
    ops = TableDescription(table_name="d", column_names=["x"]).extend({"z": "x + 1"})
    with pytest.raises(Exception):
        ops.ex()


def test_ex_examples_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8],}
    )

    ops = describe_table(d, keep_all=True).drop_columns(["z"])

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3],}
    )

    res_1 = ops.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect, res_1)

    res_2 = ops.ex()

    assert data_algebra.test_util.equivalent_frames(expect, res_2)


def test_ex_examples_catch_partial():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8],}
    )

    ops = describe_table(d, table_name="d", row_limit=2).drop_columns(["z"])

    with pytest.raises(AssertionError):
        ops.ex()


def test_ex_examples_join():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "z": [6, 7, 8],}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "q": ["a", "b", "c"],}
    )

    ops = table(d1, table_name="d1").natural_join(
        b=table(d2, table_name="d2"), by=["x", "y"], jointype="inner"
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "z": [6, 7, 8], "q": ["a", "b", "c"],}
    )

    res = ops.eval({"d1": d1, "d2": d2})

    assert data_algebra.test_util.equivalent_frames(expect, res)

    res2 = ops.ex()

    assert data_algebra.test_util.equivalent_frames(expect, res2)


def test_ex_examples_join_catch_partial():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "z": [6, 7, 8],}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "q": ["a", "b", "c"],}
    )

    ops = describe_table(d1, table_name="d1", row_limit=2).natural_join(
        b=describe_table(d2, table_name="d2", row_limit=2),
        by=["x", "y"],
        jointype="inner",
    )

    # failed to explicitly capture all rows
    with pytest.raises(Exception):
        ops.ex()

    res = ops.ex(allow_limited_tables=True)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2,], "y": [3, 3,], "z": [6, 7,], "q": ["a", "b",],}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_ex_examples_join_catch_unnamed_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "z": [6, 7, 8],}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "q": ["a", "b", "c"],}
    )

    # failed to explicitly name a table
    with pytest.raises(ValueError):
        (
            describe_table(d1).natural_join(
                b=describe_table(d2), by=["x", "y"], jointype="inner"
            )
        )


def test_ex_examples_join_catch_unnamed_2():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "z": [6, 7, 8],}
    )
    descr1 = describe_table(d1, table_name="d1", keep_all=True)
    assert descr1.table_name_was_set_by_user
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 2], "y": [3, 3, 4], "q": ["a", "b", "c"],}
    )
    descr2 = describe_table(d2, keep_all=True)
    assert not descr2.table_name_was_set_by_user
    ops = descr1.natural_join(
        b=descr2, by=["x", "y"], jointype="inner"
    )
    with pytest.raises(AssertionError):
        ops.ex()


def test_ex_examples_fn_notation_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8],}
    )

    res = ex(data(d=d).drop_columns(["z"]))

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3],}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_ex_examples_fn_notation_1_unnamed():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8],}
    )

    res = ex(data(d).drop_columns(["z"]))

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3],}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_ex_examples_fn_notation_1_cant_ex_short_descr():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": range(20), "z": 7,})

    with pytest.raises(Exception):
        res = ex(descr(d=d).drop_columns(["z"]))


def test_ex_examples_fn_notation_1_short_descr():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": range(20), "z": 7,})

    ops = descr(d=d).drop_columns(["z"])
    res = ops.transform(d)
    expect = d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": range(20)})
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_ex_examples_fn_notation_data_named():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8]}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "q": [10, 11, 12]}
    )

    res = ex(data(d1=d1).natural_join(b=data(d2=d2), by=["x", "y"], jointype="inner",))

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 1, 2], "y": [5, 4, 3], "z": [6, 7, 8], "q": [10, 11, 12]}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)
