import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.data_pipe import *
import data_algebra.PostgreSQL


def test_apply():
    data_algebra.env.push_onto_namespace_stack(locals())

    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]}
    )

    expect_1 = data_algebra.default_data_model.pd.DataFrame(
        {"x": [0.0], "y": [2.0], "z": [0.0]}
    )
    expect_2 = data_algebra.default_data_model.pd.DataFrame(
        {"x": [0.0], "y": [2.0], "z": [0.0], "q": [2.0]}
    )

    ops0 = (
        describe_table(d, table_name="t1").extend({"z": "x / y"}).select_rows("z >= 0")
    )

    res_0_0 = ops0.eval(data_map={"t1": d})

    assert data_algebra.test_util.equivalent_frames(expect_1, res_0_0)

    res_0_1 = ops0.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect_1, res_0_1)

    res_0_2 = d >> ops0

    assert data_algebra.test_util.equivalent_frames(expect_1, res_0_2)

    ops1b = (
        TableDescription("t1", ["x", "y"])
        .add(Extend({"z": "x / y"}))
        .add(SelectRows("z >= 0"))
    )

    res_1b = ops1b.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect_1, res_1b)
