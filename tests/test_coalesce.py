import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util


def test_coalesce_one():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": ["c", None, "d", None],
            "x": [1, 2, None, None],
            "y": [3, None, 4, None],
        }
    )

    ops = describe_table(d, table_name="d").extend({"z": "x %?% y", "c": "a %?% b"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": ["c", None, "d", None],
            "c": ["a", "b", "d", None],
            "x": [1, 2, None, None],
            "y": [3, None, 4, None],
            "z": [1, 2, 4, None],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )
