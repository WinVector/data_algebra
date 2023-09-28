import sqlite3

import data_algebra
import data_algebra.db_model
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
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

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [3, 4, None], "col3": [4, 5, 7]}
    )

    assert db_model.quote_identifier("a") == '"a"'
    assert db_model.quote_table_name("a") == '"a"'
    assert db_model.quote_string("a") == "'a'"

    conn.close()
