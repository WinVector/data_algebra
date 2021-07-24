
import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util


# https://github.com/WinVector/data_algebra/blob/master/Examples/WindowFunctions/WindowFunctions.ipynb
def test_window2():
    d = data_algebra.default_data_model.pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    ops = (
        describe_table(d)
            .extend(
                {
                    "row_number": "_row_number()",
                    "shift_v": "v.shift()",
                },
                order_by=["x"],
                partition_by=["g"],
            )
            .extend(
                {
                    "size": "_size()",
                    "max_v": "v.max()",
                    "min_v": "v.min()",
                    "sum_v": "v.sum()",
                    "mean_v": "v.mean()",
                    "count_v": "v.count()",
                    "size_v": "v.size()",
                },
                partition_by=["g"],
            )
        )

    res1 = ops.transform(d)

    expect1 = data_algebra.default_data_model.pd.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "row_number": [1, 1, 2, 1, 2, 3],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "size": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10, 45, 45, 80, 80, 80],
            "count_v": [1, 2, 2, 3, 3, 3],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect1, res1)

    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect1
    )
