import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_select_rows_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    ops = describe_table(d, table_name="d").select_rows("x == 1")

    d_sel = ops.transform(d)
    # note type(d.iloc[0, :]) is pandas.core.series.Series

    assert isinstance(d_sel, data_algebra.data_model.default_data_model().pd.DataFrame)


def test_select_rows_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [-2.0, 0.0, 3.0], "y": [1.0, 2.0, 3.0]}
    )

    ops = describe_table(d, table_name="d").select_rows("x.sign() == 1")

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [3.0], "y": [3.0]})

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_select_columns_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    ops = describe_table(d, table_name="d").select_columns(["x"])

    d_sel = ops.transform(d)
    # note type(d.iloc[:, 0]) is pandas.core.series.Series

    assert isinstance(d_sel, data_algebra.data_model.default_data_model().pd.DataFrame)
