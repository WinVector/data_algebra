import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL


def test_drop_columns():
    db_model = data_algebra.PostgreSQL.PostgreSQLModel()

    d = data_algebra.default_data_model.pd.DataFrame({"x": [1], "y": [2]})

    ops = describe_table(d, "d").drop_columns(["x"])

    sql = ops.to_sql(db_model)

    res = ops.eval(data_map={"d": d})

    expect = data_algebra.default_data_model.pd.DataFrame({"y": [2]})

    assert data_algebra.test_util.equivalent_frames(expect, res)
