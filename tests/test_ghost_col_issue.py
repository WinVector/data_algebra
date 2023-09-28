import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util


def test_ghost_col_issue():
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10.0, 40.0, 50.0, 70.0, 80.0, 90.0],
            "g": [1, 2, 2, 3, 3, 3],
            "ngroup": [1, 2, 2, 3, 3, 3],
            "row_number": [1, 1, 2, 1, 2, 3],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
        }
    )

    o2 = describe_table(d2).extend(
        {
            "size": "(1).sum()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "g": [1, 2, 2, 3, 3, 3],
            "ngroup": [1, 2, 2, 3, 3, 3],
            "row_number": [1, 1, 2, 1, 2, 3],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "size": [1, 2, 2, 3, 3, 3],
            "max_v": [10.0, 50.0, 50.0, 90.0, 90.0, 90.0],
            "min_v": [10.0, 40.0, 40.0, 70.0, 70.0, 70.0],
            "sum_v": [10.0, 90.0, 90.0, 240.0, 240.0, 240.0],
            "mean_v": [10.0, 45.0, 45.0, 80.0, 80.0, 80.0],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    data_algebra.test_util.check_transform(ops=o2, data=d2, expect=expect,
    )
