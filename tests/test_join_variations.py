import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util

import pytest


def test_join_variations_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "x", "y"], "z": [4, 5, 6],}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "y", "y"], "q": [7, 8, 9],}
    )

    ops1 = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=["a"], jointype="inner",
    )
    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 2],
            "b": ["x", "x", "x", "x", "y"],
            "z": [4, 4, 5, 5, 6],
            "q": [7, 8, 7, 8, 9],
        }
    )
    data_algebra.test_util.check_transform(
        ops=ops1, data={"d1": d1, "d2": d2}, expect=expect1
    )

    ops2 = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=["a", "b"], jointype="inner",
    )
    expect2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "x", "y"], "z": [4, 5, 6], "q": [7, 7, 9],}
    )
    data_algebra.test_util.check_transform(
        ops=ops2, data={"d1": d1, "d2": d2}, expect=expect2
    )

    ops2.eval({"d1": d1, "d2": d2})

    with pytest.raises(KeyError):
        ops3 = describe_table(d1, table_name="d1").natural_join(
            b=describe_table(d2, table_name="d2"),
            by=["a"],
            jointype="inner",
            check_all_common_keys_in_equi_spec=True,
        )
