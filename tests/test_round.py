import pytest

import numpy

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite
import data_algebra.test_util
import data_algebra.util


def test_round_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [0.2, 0.6, 1.2, 1.7]})
    td = describe_table(d, table_name="d")

    ops = td.extend({"v": "x.round()"})
    res_pandas = ops.transform(d)

    expect = d.copy()
    expect["v"] = numpy.round(expect["x"])

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect,
    )


def test_floor_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [0.2, 0.6, 1.2, 1.7]})
    td = describe_table(d, table_name="d")

    ops = td.extend({"v": "x.floor()"})
    res_pandas = ops.transform(d)

    expect = d.copy()
    expect["v"] = numpy.floor(expect["x"])

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)


def test_ceil_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [0.2, 0.6, 1.2, 1.7]})
    td = describe_table(d, table_name="d")

    ops = td.extend({"v": "x.ceil()"})
    res_pandas = ops.transform(d)

    expect = d.copy()
    expect["v"] = numpy.ceil(expect["x"])

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)
