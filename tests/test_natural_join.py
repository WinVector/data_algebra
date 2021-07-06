import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite


def test_natural_join_columns():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [-1, 0, 1, None], "y": [1, 2, None, 3]}
    )

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"qq": [10, 20, 30], "y": [1.0, 2.0, 3.0], "x": [4, 5, 7]}
    )

    ops4 = describe_table(d, "d").natural_join(
        b=describe_table(d2, "d2"), by=["y"], jointype="LEFT"
    )

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 7.0],
            "y": [1.0, 2.0, None, 3.0],
            "qq": [10.0, 20.0, None, 30.0],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops4,
        data={"d": d, "d2": d2},
        expect=expect)
