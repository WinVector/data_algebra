import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_order_limit_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": range(4), "y": ["a", "b", "c", "d"],}
    )

    ops = describe_table(d, table_name="d").order_rows(["x"], reverse=["x"], limit=2)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [3, 2], "y": ["d", "c"],}
    )

    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)
