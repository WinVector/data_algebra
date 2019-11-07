import sqlite3
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.SQLite


# https://github.com/WinVector/data_algebra/blob/master/Examples/WindowFunctions/WindowFunctions.ipynb
def test_window2():
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
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "ngroup": [0, 1, 1, 2, 2, 2],
            "size": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10, 45, 45, 80, 80, 80],
            "count_v": [1, 2, 2, 3, 3, 3],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect1, res1)

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
    sql1 = ops_db.to_sql(db_model, pretty=True)

    res1_db = db_model.read_query(conn, sql1)

    expect1_db = pandas.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10, 40, 50, 70, 80, 90],
            "row_number": [1, 1, 2, 1, 2, 3],
            "shift_v": [None, None, 40.0, None, 70.0, 80.0],
            "size": [1, 2, 2, 3, 3, 3],
            "max_v": [10, 50, 50, 90, 90, 90],
            "min_v": [10, 40, 40, 70, 70, 70],
            "sum_v": [10, 90, 90, 240, 240, 240],
            "mean_v": [10.0, 45.0, 45.0, 80.0, 80.0, 80.0],
            "count_v": [1, 2, 2, 3, 3, 3],
            "size_v": [1, 2, 2, 3, 3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect1_db, res1_db)

    id_ops_a = table_desciption.project(group_by=["g"]).extend(
        {"ngroup": "_row_number()",}, order_by=["g"]
    )

    id_ops_b = table_desciption.natural_join(id_ops_a, by=["g"], jointype="LEFT")

    sql2 = id_ops_b.to_sql(db_model)

    cur = conn.cursor()
    cur.execute("CREATE TABLE remote_result AS " + sql2)

    res2_db = db_model.read_table(conn, "remote_result")

    expect2_db = pandas.DataFrame(
        {
            "g": [1, 2, 2, 3, 3, 3],
            "v": [10, 40, 50, 70, 80, 90],
            "x": [1, 4, 5, 7, 8, 9],
            "ngroup": [1, 2, 2, 3, 3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect2_db, res2_db)

    # clean up
    conn.close()

    res2_pandas = id_ops_b.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect2_db, res2_pandas)
