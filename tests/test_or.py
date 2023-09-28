
import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import data_algebra.test_util
import data_algebra.SQLite


def test_or_1():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    ops = describe_table(d, table_name="d").select_rows("(ID == 3) or (ID == 4)")

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"ID": [3, 4, 4, 4, 4], "OP": ["D", "C", "A", "D", "B"],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_in_1():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    ops = describe_table(d, table_name="d").extend({"v": "ID.is_in([3, 4])"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
            "v": [False] * 3 + [True] * 5 + [False] * 3,
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )


def test_in_1b():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    ops = describe_table(d, table_name="d").extend({"v": "ID.is_in((3, 4))"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
            "v": [False] * 3 + [True] * 5 + [False] * 3,
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )


def test_in_2():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 34, 44, 44, 44, 44, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    ops = (
        describe_table(d, table_name="d")
        .extend({"v": "ID.is_in([34, 44])"})
        .select_rows("v")
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"ID": [34, 44, 44, 44, 44,], "OP": ["D", "C", "A", "D", "B"], "v": [True] * 5,}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )


def test_in_3():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    ops = describe_table(d, table_name="d").select_rows("ID.is_in([3, 4])")

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"ID": [3, 4, 4, 4, 4], "OP": ["D", "C", "A", "D", "B"],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )
