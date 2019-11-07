import pandas

import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml


def test_drop_columns():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = pandas.DataFrame({"x": [1], "y": [2]})

    ops = describe_table(d, "d").drop_columns(["x"])

    sql = ops.to_sql(db_model)

    res = ops.eval_pandas(data_map=od(d=d), eval_env=locals())

    data_algebra.test_util.check_op_round_trip(ops)

    expect = pandas.DataFrame({"y": [2]})

    assert data_algebra.test_util.equivalent_frames(expect, res)
