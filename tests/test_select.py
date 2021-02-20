from data_algebra.data_ops import *


def test_select_rows_1():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    ops = describe_table(d, table_name="d").select_rows("x == 1")

    d_sel = ops.transform(d)
    # note type(d.iloc[0, :]) is pandas.core.series.Series

    assert isinstance(d_sel, data_algebra.default_data_model.pd.DataFrame)


def test_select_columns_1():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    ops = describe_table(d, table_name="d").select_columns(["x"])

    d_sel = ops.transform(d)
    # note type(d.iloc[:, 0]) is pandas.core.series.Series

    assert isinstance(d_sel, data_algebra.default_data_model.pd.DataFrame)
