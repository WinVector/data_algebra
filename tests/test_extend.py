
import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.test_util import formats_to_self

import pytest


def test_extend_0():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({}, partition_by=["c", "g"])

    assert isinstance(ops, TableDescription)
    assert formats_to_self(ops)

    res = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(d, res)


def test_extend_p():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"}, partition_by=["g"])

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {"g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4], "c": [3, 4, 3, 4],}
    )
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_extend_p0():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"})

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {"g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4], "c": [4, 4, 4, 4],}
    )
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_extend_shrink_1():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"}).extend({"d": "y.min()"})

    assert formats_to_self(ops)
    assert isinstance(ops.sources[0], TableDescription)  # check does combine nodes in this case

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "g": ["a", "b", "a", "b"],
            "y": [1, 2, 3, 4],
            "c": [4, 4, 4, 4],
            "d": [1, 1, 1, 1],
        }
    )
    assert data_algebra.test_util.equivalent_frames(expect, res)

    ops2 = describe_table(d, "d").extend({"c": "y.max()", "d": "y.min()"})

    assert str(ops) == str(ops2)

    ops2b = describe_table(d, "d").extend({"c": "y"}).extend({"d": "c"})

    assert isinstance(ops2b.sources[0], ExtendNode)

    ops2c = describe_table(d, "d").extend({"c": "1"}).extend({"c": "2"})

    assert isinstance(ops2c.sources[0], TableDescription)

    ops3 = describe_table(d, "d").extend({"c": "y.max()"}).extend({"d": "y"})

    assert isinstance(ops3.sources[0], ExtendNode)


def test_extend_shrink_2():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"}).extend({"d": "c.min()"})

    assert formats_to_self(ops)
    assert isinstance(ops.sources[0], ExtendNode)  # check doesn't combine nodes in this case

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "g": ["a", "b", "a", "b"],
            "y": [1, 2, 3, 4],
            "c": [4, 4, 4, 4],
            "d": [4, 4, 4, 4],
        }
    )
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_extend_catch_nonagg():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    with pytest.raises(ValueError):
        ops = describe_table(d, "d").extend({'y': 'y'}, partition_by=["c", "g"])
