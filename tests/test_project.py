
import data_algebra.util
import pandas
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml



def test_R_yaml():
    d = pandas.DataFrame({
        'x': [1, 1, 2, 2],
        'y': [1, 2, 3, 4],
    })

    ops1 = describe_table(d). \
        project({'sum_y': 'y.sum()'})
    res1 = ops1.transform(d)
    expect1 = pandas.DataFrame({
        'sum_y': [10],
    })
    assert data_algebra.util.equivalent_frames(res1, expect1)

    ops2 = describe_table(d). \
        project({'sum_y': 'y.sum()'},
                group_by=['x'])
    res2 = ops2.transform(d)
    expect2 = pandas.DataFrame({
        'x': [1, 2],
        'sum_y': [3, 7],
    })
    assert data_algebra.util.equivalent_frames(res2, expect2)


def test_project0():
    d = pandas.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(
        group_by=["c", "g"]
    )

    res = ops.transform(d)

    expect = pandas.DataFrame(
        {"c": [1, 1], "g": ["a", "b"]}
    )

    assert data_algebra.util.equivalent_frames(expect, res)


def test_project():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = pandas.DataFrame(
        {"c": [1, 1, 1, 1], "g": ["a", "b", "a", "b"], "y": [1, 2, 3, 4]}
    )

    ops = describe_table(d, "d").project(
        od(ymax="y.max()", ymin="y.min()"), group_by=["c", "g"]
    )

    sql = ops.to_sql(db_model)

    data_algebra.yaml.check_op_round_trip(ops)

    res = ops.eval_pandas(data_map=od(d=d), eval_env=locals())

    expect = pandas.DataFrame(
        {"c": [1, 1], "g": ["a", "b"], "ymax": [3, 4], "ymin": [1, 2]}
    )

    assert data_algebra.util.equivalent_frames(expect, res)
