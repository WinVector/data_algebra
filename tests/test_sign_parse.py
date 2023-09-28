import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_sign_parse_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "g": [False, True, False],}
    )

    # Pandas mis-parses -3 is unary_op(3) and then throws!
    # we are now avoiding that by not calling eval() on Pandas DataFrame
    ops = describe_table(d, table_name="d").extend({"x_g": "g.if_else(x, -3)",})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "g": [False, True, False], "x_g": [-3, 2, -3],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_if_else_complex():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "g": [1, 1, 1, 2, 2, 2],}
    )

    ops = describe_table(d, table_name="d").extend({"r": "(g>1).if_else(1, 2)"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "g": [1, 1, 1, 2, 2, 2], "r": [2, 2, 2, 1, 1, 1]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
