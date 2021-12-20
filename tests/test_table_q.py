import data_algebra
import data_algebra.SQLite
from data_algebra.data_ops import *


def test_table_q_1():
    ops = TableDescription(table_name="d", column_names=["x"])
    db_model = data_algebra.SQLite.SQLiteModel()
    sql_str = db_model.to_sql(ops)
    assert isinstance(sql_str, str)
    assert "SELECT" in sql_str
