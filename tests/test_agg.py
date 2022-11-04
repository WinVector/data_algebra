import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util
import data_algebra.SQLite


def test_agg():
    d = data_algebra.pandas_model.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4], "g": [1, 1, 2, 2],}
    )

    ops = describe_table(d, table_name="d").project({"x": "x.max()"}, group_by=["g"])

    expect = data_algebra.pandas_model.default_data_model.pd.DataFrame({"g": [1, 2], "x": [2, 4],})

    db_handle = data_algebra.SQLite.SQLiteModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
