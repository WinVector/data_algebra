import numpy

import data_algebra
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_concat_rows():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]}
    )

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [0, 4, None], "y": [2, 7, None]}
    )

    ops4 = describe_table(d1, "d1").concat_rows(b=describe_table(d2, "d2"))

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, None, 0.0, 4.0, None],
            "y": [1.0, 2.0, None, 3.0, 2.0, 7.0, None],
            "source_name": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops4, data={"d1": d1, "d2": d2}, expect=expect
    )


def test_if_concat_is_stoopid_1():
    # look if concat is concating by position instead of name
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a"], "y": ["b"]})

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({"y": ["c"], "x": ["d"]})

    ops = describe_table(d1, "d1").concat_rows(
        b=describe_table(d2, "d2"), id_column=None
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["a", "d"], "y": ["b", "c"]}
    )

    data_algebra.test_util.check_transform(
        ops=ops, data={"d1": d1, "d2": d2}, expect=expect,
    )


def test_if_concat_is_stoopid_2():
    # look if concat is concating by position intead of name
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a"], "y": ["b"]})

    descr = describe_table(d1, "d1")
    ops = descr.concat_rows(b=descr.select_columns(["y", "x"]), id_column=None)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": ["a", "a"], "y": ["b", "b"]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d1, expect=expect,
    )
