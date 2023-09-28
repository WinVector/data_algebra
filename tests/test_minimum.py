import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_minimum_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "g": [1, 1, 1, 2, 2, 2],}
    )

    ops = (
        describe_table(d, table_name="d")
        .extend({"x_g_max": "x.max()",}, partition_by=["g"])
        .extend({"xl": "x.minimum(x_g_max - 1)"})
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "g": [1, 1, 1, 2, 2, 2],
            "x_g_max": [3, 3, 3, 6, 6, 6],
            "xl": [1, 2, 2, 4, 5, 5],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, 
        data=d, 
        expect=expect,
        valid_for_empty=False,
    )
