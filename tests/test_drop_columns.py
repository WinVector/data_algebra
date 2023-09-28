import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex


def test_drop_columns():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2]})

    ops = describe_table(d, "d").drop_columns(["x"])

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [2]})

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
