import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex


def test_can_convert_v_to_numeric():
    data_model = data_algebra.data_model.default_data_model()
    assert data_model.can_convert_col_to_numeric(0)
    assert data_model.can_convert_col_to_numeric(1.0)
    assert not data_model.can_convert_col_to_numeric("hi")
    assert data_model.can_convert_col_to_numeric(numpy.asarray([1, 2]))
    assert data_model.can_convert_col_to_numeric(
        data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2]})["x"]
    )
    assert data_model.can_convert_col_to_numeric(
        data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, numpy.nan]})["x"]
    )
    assert not data_model.can_convert_col_to_numeric(
        data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a", "b"]})["x"]
    )
    assert not data_model.can_convert_col_to_numeric(
        data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a", numpy.nan]})["x"]
    )
    assert not data_model.can_convert_col_to_numeric(
        data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, "a"]})["x"]
    )


def test_equiv():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2], "y": [3, numpy.nan]}
    )
    d1b = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [2, 1], "y": [numpy.nan, 3]}
    )
    d1c = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4.0001]})
    d1d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2], "y": [3.0001, numpy.nan]}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "z": ["a", "b"]})
    d3 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    d4 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2]})
    d5 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 0], "y": [3, numpy.nan, 0]}
    )
    assert data_algebra.test_util.equivalent_frames(d1, d1)
    assert data_algebra.test_util.equivalent_frames(d2, d2)
    assert data_algebra.test_util.equivalent_frames(d1, d1[["y", "x"]])
    assert not data_algebra.test_util.equivalent_frames(
        d1, d1[["y", "x"]], check_column_order=True
    )
    assert not data_algebra.test_util.equivalent_frames(d1, d2)
    assert data_algebra.test_util.equivalent_frames(d1, d1b)
    assert not data_algebra.test_util.equivalent_frames(d1, d1b, check_row_order=True)
    assert not data_algebra.test_util.equivalent_frames(d1, d1c, float_tol=1e-3)
    assert not data_algebra.test_util.equivalent_frames(d1, d1c, float_tol=1e-8)
    assert data_algebra.test_util.equivalent_frames(d1, d1d, float_tol=1e-3)
    assert not data_algebra.test_util.equivalent_frames(d1, d1d, float_tol=1e-8)
    assert not data_algebra.test_util.equivalent_frames(d1, d3)
    assert not data_algebra.test_util.equivalent_frames(d1, d4)
    assert not data_algebra.test_util.equivalent_frames(d1, d5)


def test_simple():
    q = 4.0
    x = 2.0
    var_name = "y"

    ops = TableDescription(table_name="d", column_names=["x", "y"]).extend(
        {"z": f"1/({q}) + x"}
    )

    d_local = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0]}
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [1.25, 2.25]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d_local, expect=expect)


def test_pandas_to_example():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "record_id": [1, 1, 1, 2, 2, 2],
            "column_label": [
                "rec_col1",
                "rec_col2",
                "rec_col3",
                "rec_col1",
                "rec_col2",
                "rec_col3",
            ],
            "c_row1": [1.0, None, 3.0, 11.0, None, 13.0],
            "c_row2": [4, 5, 6, 14, 15, 16],
            "c_row3": [7, 8, 9, 17, 18, 19],
        }
    )
    d_str = data_algebra.util.pandas_to_example_str(d)
    d_back = eval(d_str, globals(), {"pd": data_algebra.data_model.default_data_model().pd})
    assert data_algebra.test_util.equivalent_frames(d, d_back)
