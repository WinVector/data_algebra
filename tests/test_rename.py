
import pytest
import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite
import data_algebra.test_util
import data_algebra.util


def test_rename_columns_swap():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2]})
    td = describe_table(d, table_name="d")
    swap = td.rename_columns({"y": "x", "x": "y"})
    res_pandas = swap.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [1], "x": [2],})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(swap, data={"d": d}, expect=expect)
    # name collision is an error
    with pytest.raises(ValueError):
        td.rename_columns({"y": "x"})


def test_rename_columns_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2], "z": ["q"]})
    td = describe_table(d, table_name="d")
    ops = td.rename_columns({"y1": "y", "x2": "x"})
    res_pandas = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x2": [1], "y1": [2], "z": ["q"]})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)


def test_map_columns_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2], "z": ["q"]})
    td = describe_table(d, table_name="d")
    ops = td.map_columns({"y": "y1", "x": "x2"})
    res_pandas = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x2": [1], "y1": [2], "z": ["q"]})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)


def test_map_columns_swap():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2]})
    td = describe_table(d, table_name="d")
    swap = td.map_columns({"y": "x", "x": "y"})
    res_pandas = swap.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [1], "x": [2],})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(swap, data={"d": d}, expect=expect)
    # name collision is an error
    with pytest.raises(ValueError):
        td.map_columns({"y": "x"})


def test_map_columns_del():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2], "z": ["q"]})
    td = describe_table(d, table_name="d")
    ops = td.map_columns({"y": "y1", "x": None})
    assert set(ops.columns_produced()) == set(["y1", "z"])
    res_pandas = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y1": [2], "z": ["q"]})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect,
    )


def test_map_columns_swap_del():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    td = describe_table(d, table_name="d")
    swap = td.map_columns({"y": "x", "x": "y", "z": None})
    assert set(swap.columns_produced()) == set(["y", "x"])
    res_pandas = swap.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [1], "x": [2],})
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(swap, data={"d": d}, expect=expect,
    )
