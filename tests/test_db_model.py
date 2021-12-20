import sqlite3

import data_algebra
import data_algebra.db_model
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.SQLite


def test_db_model_base():
    db_model = data_algebra.db_model.DBModel()

    db_model.__repr__()
    str(db_model)

    conn = sqlite3.connect(":memory:")

    # just for testing, should use a data_algebra.SQLite.SQLiteModel in practice
    db_handle = data_algebra.db_model.DBHandle(db_model=db_model, conn=conn)
    str(db_handle)
    db_handle.__repr__()

    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [3, 4, None], "col3": [4, 5, 7]}
    )

    assert db_model.quote_identifier("a") == '"a"'
    assert db_model.quote_table_name("a") == '"a"'
    assert db_model.quote_string("a") == "'a'"

    conn.close()


def test_db_model_table_def_values_sql():
    db_handle = data_algebra.SQLite.example_handle()
    d2 = data_algebra.default_data_model.pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [3, 4, None], "col3": [4, 5, 7]}
    )
    td = descr(d2=d2)
    sql_list = td.example_values_to_sql_str_list(db_handle.db_model)
    sql = "\n".join(sql_list)
    db_handle.read_query(sql)
    back = db_handle.read_query(sql)
    db_handle.close()
    assert data_algebra.test_util.equivalent_frames(d2, back)
