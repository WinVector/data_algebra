import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml
from data_algebra.test_util import formats_to_self

import pytest


def test_R_yaml():
    d = data_algebra.pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4],})

    ops1 = describe_table(d).project({"sum_y": "y.sum()"})
    res1 = ops1.transform(d)
    expect1 = data_algebra.pd.DataFrame({"sum_y": [10],})
    assert data_algebra.test_util.equivalent_frames(res1, expect1)

    ops2 = describe_table(d).project({"sum_y": "y.sum()"}, group_by=["x"])
    res2 = ops2.transform(d)
    expect2 = data_algebra.pd.DataFrame({"x": [1, 2], "sum_y": [3, 7],})
    assert data_algebra.test_util.equivalent_frames(res2, expect2)


def test_project0():
    d = data_algebra.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(group_by=["c", "g"])

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.pd.DataFrame({"c": [1, 1], "g": ["a", "b"]})
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_project_z():
    d = data_algebra.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project({"c": "c.max()"})

    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.pd.DataFrame({"c": [1]})
    assert data_algebra.test_util.equivalent_frames(expect, res)


def test_project_zz():
    d = data_algebra.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    with pytest.raises(ValueError):
        describe_table(d, "d").project()


def test_project():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = data_algebra.pd.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(
        od(ymax="y.max()", ymin="y.min()"), group_by=["c", "g"]
    )

    sql = ops.to_sql(db_model)

    data_algebra.test_util.check_op_round_trip(ops)

    res = ops.eval(data_map=od(d=d))

    expect = data_algebra.pd.DataFrame(
        {"c": [1, 1], "g": ["a", "b"], "ymax": [3, 4], "ymin": [1, 2]}
    )

    assert data_algebra.test_util.equivalent_frames(expect, res)
