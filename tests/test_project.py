import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.test_util import formats_to_self

import pytest


def test_project0():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(group_by=["c", "g"])

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1], "g": ["a", "b"]}
    )
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_project_z():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project({"c": "c.max()"})

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame({"c": [1]})
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_project_zz():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    with pytest.raises(ValueError):
        describe_table(d, "d").project()


def test_project():
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(
        {"ymax": "y.max()", "ymin": "y.min()"}, group_by=["c", "g"]
    )

    sql = ops.to_sql(db_model)

    res = ops.eval(data_map={"d": d})

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"c": [1, 1], "g": ["a", "b"], "ymax": [3, 4], "ymin": [1, 2]}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)
