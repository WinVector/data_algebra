import numpy

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_simple_expr_1():
    d_orig = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1.0, 2.0, -3.0, 4.0]})
    d = d_orig.copy()

    ops = describe_table(d, table_name="d").extend(
        {
            "z": "x + 1",
            "sin_x": "x.sin()",  # triggers numpy path
            "xm": "-x",
            "xs": "2 * (x-2).sign()",
        }
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1.0, 2.0, -3.0, 4.0]})
    expect["z"] = expect["x"] + 1
    expect["sin_x"] = numpy.sin(expect["x"])
    expect["xm"] = -expect["x"]
    expect["xs"] = 2 * numpy.sign(d["x"] - 2)

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)

    assert d.equals(d_orig)
