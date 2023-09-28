import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.test_util import formats_to_self

import pytest


def test_project0():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(group_by=["c", "g"])

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1], "g": ["a", "b"]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_project_z():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project({"c": "c.max()"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"c": [1]})

    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, empty_produces_empty=False,
    )


def test_project_zz():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    with pytest.raises(ValueError):
        describe_table(d, "d").project()


def test_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(
        {"ymax": "y.max()", "ymin": "y.min()"}, group_by=["c", "g"]
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1], "g": ["a", "b"], "ymax": [3, 4], "ymin": [1, 2]}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_project_catch_nonagg():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    with pytest.raises(ValueError):
        ops = describe_table(d, "d").project({"y": "y"}, group_by=["c", "g"])
