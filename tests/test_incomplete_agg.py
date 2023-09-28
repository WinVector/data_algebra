import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex

import data_algebra.test_util

import pytest


def test_incomplete_agg_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "genus": (1, 1, 1, 2),
            "group": ("a", "a", "b", "b"),
            "x": (1, 2, 3, 4),
            "y": (10, 20, 30, 40),
        }
    )

    ops_1 = describe_table(d, table_name="d").project(
        {"x": "x.mean()", "y": "y.mean()",}, group_by=["genus", "group"]
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "genus": (1, 1, 2),
            "group": ("a", "b", "b"),
            "x": (1.5, 3, 4),
            "y": (15, 30, 40),
        }
    )

    data_algebra.test_util.check_transform(ops=ops_1, data=d, expect=expect)

    with pytest.raises(ValueError):
        ops_bad = describe_table(d, table_name="d").project(
            {"x": "x.mean()", "y": "y",},  # error: forgoat aggregator!
            group_by=["genus", "group"],
        )
