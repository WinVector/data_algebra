import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.data_pipe import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml


def test_apply():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    data_algebra.env.push_onto_namespace_stack(locals())

    d = data_algebra.pd.DataFrame(
        {"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]}
    )

    expect_1 = data_algebra.pd.DataFrame({"x": [0.0], "y": [2.0], "z": [0.0]})
    expect_2 = data_algebra.pd.DataFrame(
        {"x": [0.0], "y": [2.0], "z": [0.0], "q": [2.0]}
    )

    ops0 = (
        TableDescription("t1", ["x", "y"]).extend({"z": "x / y"}).select_rows("z >= 0")
    )

    res_0_0 = ops0.eval_pandas(data_map={"t1": d})

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
