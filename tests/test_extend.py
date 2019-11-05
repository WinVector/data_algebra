import data_algebra.util
import pandas
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml
from data_algebra.test_util import formats_to_self

import pytest


def test_extend_0():
    d = pandas.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({}, partition_by=["c", "g"])

    assert isinstance(ops, TableDescription)
    assert formats_to_self(ops)

    res = ops.transform(d)
    assert data_algebra.util.equivalent_frames(d, res)


def test_extend_p():
    d = pandas.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"}, partition_by=["g"])

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = pandas.DataFrame(
        {"g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4], "c": [3, 4, 3, 4],}
    )
    assert data_algebra.util.equivalent_frames(expect, res)


def test_extend_p0():
    d = pandas.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").extend({"c": "y.max()"})

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = pandas.DataFrame(
        {"g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4], "c": [4, 4, 4, 4],}
    )
    assert data_algebra.util.equivalent_frames(expect, res)
