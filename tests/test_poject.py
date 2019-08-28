
import pandas
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml

def test_project():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = pandas.DataFrame({'c': [1, 1, 1, 1], 'g': ['a', 'b', 'a', 'b'], 'y': [1, 2, 3, 4]})

    ops = describe_pandas_table(d, "d") .\
        project(od(ymax='y.max()', ymin='y.min()'), group_by=['c', 'g'])

    sql = ops.to_sql(db_model)

    data_algebra.yaml.check_op_round_trip(ops)

    res = ops.eval_pandas(od(d=d))

    expect = pandas.DataFrame({'c':[1, 1], 'g': ['a', 'b'], 'ymax': [3, 4], 'ymin': [1, 2]})

    assert data_algebra.util.equivalent_frames(expect, res)
