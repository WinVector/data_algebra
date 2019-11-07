import sqlite3
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.SQLite


# https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html


def test_window_fns():

    d = pandas.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
        }
    )

    table_desciption = describe_table(d)
    ops = table_desciption.extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    ).extend(
        {
            "ngroup": "_ngroup()",
            "size": "_size()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "count_v": "v.count()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )

    res1 = ops.transform(d)

    expect1 = pandas.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "row_number": [1, 1, 2, 1, 2, 3],
            "ngroup": [0, 1, 1, 2, 2, 2],
            "size": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10, 45, 45, 80, 80, 80],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "count_v": [1, 2, 2, 3, 3, 3],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res1, expect1)

    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()
    db_model.prepare_connection(conn)

    ops_db = table_desciption.extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    ).extend(
        {
            # 'ngroup': '_ngroup()',
            "size": "_size()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "count_v": "v.count()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )

    db_model.insert_table(conn, d, table_desciption.table_name)
    sql1 = ops_db.to_sql(db_model)
    res1_db = db_model.read_query(conn, sql1)

    expect2 = expect1[
        [
            "g",
            "x",
            "v",
            "row_number",
            "shift_v",
            "size",
            "max_v",
            "min_v",
            "sum_v",
            "mean_v",
            "count_v",
            "size_v",
        ]
    ]

    assert data_algebra.test_util.equivalent_frames(res1_db, expect2)

    # clean up
    conn.close()
