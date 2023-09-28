import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite


def test_extend_order_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {("x_" + str(i)): range(i * 20, i * 20 + 5) for i in range(10)}
    )

    ops = describe_table(d, table_name="d").extend({"z": 1})

    res_1 = ops.transform(d)
    cols_pandas = [c for c in res_1.columns if c != "z"]

    expect = ["x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9"]
    assert cols_pandas == expect

    sqllite_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as sqllite_conn:
        sqllite_model.prepare_connection(sqllite_conn)
        sqllite_model.insert_table(sqllite_conn, d, "d")
        sqllite_sql = ops.to_sql(sqllite_model)
        res_sqlite = sqllite_model.read_query(sqllite_conn, sqllite_sql)

    cols_sql = [c for c in res_sqlite.columns if c != "z"]
    expect = ["x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9"]
    assert cols_sql == expect
