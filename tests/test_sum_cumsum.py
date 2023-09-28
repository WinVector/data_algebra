import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import data_algebra.test_util

import pytest


def test_sum_cumsum_1():
    #  google.api_core.exceptions.BadRequest: 400 Partitioning by expressions of type FLOAT64 is not allowed
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["1.0", "2.0", "2.0", "1.0", "1.0"], "s": ["c", "d", "a", "b", "e"],}
    )

    with pytest.raises(ValueError):
        describe_table(d, table_name="d").extend(
            {"total_count": "(1).cumsum()",},  # wrong method
        )

    with pytest.raises(ValueError):
        describe_table(d, table_name="d").extend(
            {"total_order": "(1).sum()",}, order_by=["s"],  # wrong method
        )

    ops = (
        describe_table(d, table_name="d")
        .extend({"total_count": "(1).sum()",},)
        .extend({"group_count": "(1).sum()",}, partition_by=["x"],)
        .extend({"total_order": "(1).cumsum()",}, order_by=["s"],)
        .extend({"group_order": "(1).cumsum()",}, partition_by=["x"], order_by=["s"],)
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": ["1.0", "2.0", "2.0", "1.0", "1.0"],
            "s": ["c", "d", "a", "b", "e"],
            "total_count": [5, 5, 5, 5, 5],
            "group_count": [3, 2, 2, 3, 3],
            "total_order": [3, 4, 1, 2, 5],
            "group_order": [2, 2, 1, 1, 3],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
