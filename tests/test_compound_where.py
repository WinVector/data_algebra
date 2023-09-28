import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util

import data_algebra.SQLite


def test_compount_where_and():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": ["c", None, "d", None],
            "x": [1, 2, None, None],
            "y": [3, None, 4, None],
        }
    )

    ops = describe_table(d, table_name="d").select_rows(
        'a == "a" and b == "c" and x > 0 and y < 4'
    )

    db_handle = data_algebra.SQLite.SQLiteModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": ["a"], "b": ["c"], "x": [1.0], "y": [3.0],}
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect,
    )
