import numpy

import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SparkSQL
import data_algebra.test_util


def test_mapv_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": ["a", "b", "c", None, numpy.nan, "b"],})
    ops = (
        data(d=d)
        .extend({"x_mapped": "0.0"})  # not deleted, as it is a constant
        .extend(
            {"z": "1 + -3", "x_mapped": 'x.mapv({"a": 1.0, "b": 2.0, "q": -3}, 0.5)'}
        )
    )
    ops_str = str(ops)
    assert isinstance(ops_str, str)

    transformed = ops.transform(d)
    expect = pd.DataFrame(
        {
            "x": ["a", "b", "c", None, numpy.nan, "b"],
            "z": -2,
            "x_mapped": [1.0, 2.0, 0.5, 0.5, 0.5, 2.0],
        }
    )
    assert data_algebra.test_util.equivalent_frames(transformed, expect)

    # problems insert None into Spark numeric columns, so use different test for spark
    # also Spark converts the result to strings
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={str(data_algebra.SparkSQL.SparkSQLModel())},
    )


def test_mapv_1_spark():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": ["a", "b", "c", "z", "z", "b"],})
    ops = (
        data(d=d)
        .extend({"x_mapped": "0.0"})  # not deleted, as it is a constant
        .extend(
            {"z": "1 + -3", "x_mapped": 'x.mapv({"a": 1.0, "b": 2.0, "q": -3}, 0.5)'}
        )
    )
    ops_str = str(ops)
    assert isinstance(ops_str, str)

    transformed = ops.transform(d)
    expect = pd.DataFrame(
        {
            "x": ["a", "b", "c", "z", "z", "b"],
            "z": -2,
            "x_mapped": [1.0, 2.0, 0.5, 0.5, 0.5, 2.0],
        }
    )
    assert data_algebra.test_util.equivalent_frames(transformed, expect)

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )
