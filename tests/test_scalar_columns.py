import numpy

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_scalar_columns():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3, 4]})

    ops = describe_table(d, table_name="d").extend(
        {
            "n1": 1,
            "z1": numpy.nan,
            "z2": None,
            "z3": "None",  # none
            "c1": "'a'",  # string
            "b1": True,
            "f1": 2.1,
            "f2": "3.5",  # number
        }
    )
    res = ops.transform(d)

    expect = d.copy()
    expect["n1"] = 1
    expect["z1"] = numpy.nan
    expect["z2"] = None
    expect["z3"] = None
    expect["c1"] = "a"
    expect["b1"] = True
    expect["f1"] = 2.1
    expect["f2"] = 3.5

    assert data_algebra.test_util.equivalent_frames(res, expect)

    # Note: leaving parse and db checks off as this pipeline
    # wasn't formed purely from text and nan/None/logical is handled
    # differently in dbs.
    # data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect, check_parse=False)
