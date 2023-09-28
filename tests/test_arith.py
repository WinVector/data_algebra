import numpy


import data_algebra
from data_algebra.data_ops import describe_table
import data_algebra.util
import data_algebra.test_util


def test_arith_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [6.0, 9.0, 3.0, 8.0],}
    )

    ops = describe_table(d, table_name="d").extend(
        {
            "a": "x + y",
            "b": "x - y",
            "c": "x * y",
            "d": "x / y",
            "e": "x + y / 2",
            "f": "x*x + y*y",
            "g": "(x*x + y*y).sqrt()",
            'h': 'x*x == y**2',
        }
    )

    expect = d.copy()
    expect["a"] = expect.x + expect.y
    expect["b"] = expect.x - expect.y
    expect["c"] = expect.x * expect.y
    expect["d"] = expect.x / expect.y
    expect["e"] = expect.x + (expect.y / 2)
    expect["f"] = (expect.x * expect.x) + (expect.y * expect.y)
    expect["g"] = numpy.sqrt(((expect.x * expect.x) + (expect.y * expect.y)))
    expect['h'] = (expect.x * expect.x) == (expect.y)**2

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
