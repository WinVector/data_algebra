import pytest

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.SQLite


# https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html


def test_window_fns():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    ops = table_desciption.extend(
        {
            "row_number": "_row_number()",
            # "shift_v": "v.shift()",
        },
        order_by=["x"],
        partition_by=["g"],
    ).extend(
        {
            # "ngroup": "_ngroup()",
            "size": "_size()",
            "size2": "(1).sum()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )

    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "row_number": [1, 1, 2, 1, 2, 3],
            # "ngroup": [0, 1, 1, 2, 2, 2],
            "size": [1, 2, 2, 3, 3, 3],
            "size2": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10, 45, 45, 80, 80, 80],
            # "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect1,
    )


def test_window_fns_pandas_only():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    ops = table_desciption.extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    ).extend(
        {
            "ngroup": "_ngroup()",
            "size": "_size()",
            "size2": "(1).sum()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )

    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "row_number": [1, 1, 2, 1, 2, 3],
            "ngroup": [0, 1, 1, 2, 2, 2],
            "size": [1, 2, 2, 3, 3, 3],
            "size2": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10, 45, 45, 80, 80, 80],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect1)


def test_window_fns_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    ops = table_desciption.extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    ).project(
        {
            # "ngroup": "_ngroup()",
            "size": "_size()",
            "size2": "(1).sum()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "size_v": "v.size()",
        },
        group_by=["g"],
    )

    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 3],
            "size": [1, 2, 3],
            "size2": [1, 2, 3],
            "max_v": [10, 50, 90],
            "min_v": [10, 40, 70],
            "sum_v": [10, 90, 240],
            "mean_v": [10, 45, 80],
            "size_v": [1, 2, 3],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect1,
    )


def test_window_fns_project_no_ngroup_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    with pytest.raises(ValueError):
        table_desciption.extend(
            {"row_number": "_row_number()", "shift_v": "v.shift()",},
            order_by=["x"],
            partition_by=["g"],
        ).project(
            {
                "ngroup": "_ngroup()",
                "size": "_size()",
                "size2": "(1).sum()",
                "max_v": "v.max()",
                "min_v": "v.min()",
                "sum_v": "v.sum()",
                "mean_v": "v.mean()",
                "size_v": "v.size()",
            },
            group_by=["g"],
        )


def test_window_fns_project_pandas_only():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    ops = table_desciption.extend(
        {"row_number": "_row_number()", "ngroup": "_ngroup()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    ).project(
        {
            "ng_max": "ngroup.max()",
            "ng_min": "ngroup.max()",
            "size": "_size()",
            "size2": "(1).sum()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "size_v": "v.size()",
        },
        group_by=["g"],
    )

    res = ops.transform(d)

    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [1, 2, 3],
            "ng_max": [0, 1, 2],
            "ng_min": [0, 1, 2],
            "size": [1, 2, 3],
            "size2": [1, 2, 3],
            "max_v": [10, 50, 90],
            "min_v": [10, 40, 70],
            "sum_v": [10, 90, 240],
            "mean_v": [10, 45, 80],
            "size_v": [1, 2, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res, expect1)
