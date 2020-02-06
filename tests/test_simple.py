
import data_algebra
import numpy
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.util import od


def test_od():
    res = od(y=1, x="b")
    expect = collections.OrderedDict()
    expect["y"] = 1
    expect["x"] = "b"
    assert res == expect


def test_can_convert_v_to_numeric():
    assert data_algebra.util.can_convert_v_to_numeric(0)
    assert data_algebra.util.can_convert_v_to_numeric(1.0)
    assert not data_algebra.util.can_convert_v_to_numeric("hi")
    assert data_algebra.util.can_convert_v_to_numeric(numpy.asarray([1, 2]))
    assert data_algebra.util.can_convert_v_to_numeric(
        data_algebra.pd.DataFrame({"x": [1, 2]})["x"]
    )
    assert data_algebra.util.can_convert_v_to_numeric(
        data_algebra.pd.DataFrame({"x": [1, numpy.nan]})["x"]
    )
    assert not data_algebra.util.can_convert_v_to_numeric(
        data_algebra.pd.DataFrame({"x": ["a", "b"]})["x"]
    )
    assert not data_algebra.util.can_convert_v_to_numeric(
        data_algebra.pd.DataFrame({"x": ["a", numpy.nan]})["x"]
    )
    assert not data_algebra.util.can_convert_v_to_numeric(
        data_algebra.pd.DataFrame({"x": [1, "a"]})["x"]
    )


def test_equiv():
    d1 = data_algebra.pd.DataFrame({"x": [1, 2], "y": [3, numpy.nan]})
    d1b = data_algebra.pd.DataFrame({"x": [2, 1], "y": [numpy.nan, 3]})
    d1c = data_algebra.pd.DataFrame({"x": [1, 2], "y": [3, 4.0001]})
    d1d = data_algebra.pd.DataFrame({"x": [1, 2], "y": [3.0001, numpy.nan]})
    d2 = data_algebra.pd.DataFrame({"x": [1, 2], "z": ["a", "b"]})
    d3 = data_algebra.pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    d4 = data_algebra.pd.DataFrame({"x": [1, 2]})
    d5 = data_algebra.pd.DataFrame({"x": [1, 2, 0], "y": [3, numpy.nan, 0]})
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
    q = 4
    x = 2
    var_name = "y"

    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    d_local = data_algebra.pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    res = ops.eval(data_map={"d": d_local})
    expect = data_algebra.pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [1.25, 2.25]})
    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_pandas_to_example():
    d = data_algebra.pd.DataFrame(
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
    d_back = eval(d_str)
    assert data_algebra.test_util.equivalent_frames(d, d_back)
