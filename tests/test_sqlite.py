import sqlite3
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.SQLite
import data_algebra.util


def test_sqlite():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # From:
    #   https://docs.python.org/3.5/library/sqlite3.html

    # noinspection SqlNoDataSourceInspection
    cur.execute(
        """CREATE TABLE stocks
                 (date text, trans text, symbol text, qty real, price real)"""
    )

    # Insert a row of data
    # noinspection SqlNoDataSourceInspection
    cur.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    conn.commit()
    # work a simple example

    ops = TableDescription(
        "stocks", ["date", "trans", "symbol", "qty", "price"]
    ).extend({"cost": "qty * price"})

    pp = ops.to_python(pretty=True)

    db_model = data_algebra.SQLite.SQLiteModel()

    sql = ops.to_sql(db_model, pretty=True)

    res = db_model.read_query(conn, sql)

    # clean up
    conn.close()

    expect = pandas.DataFrame(
        {
            "symbol": ["RHAT"],
            "date": ["2006-01-05"],
            "price": 35.14,
            "qty": 100.0,
            "trans": "BUY",
            "cost": 3514.0,
        }
    )

    assert data_algebra.test_util.equivalent_frames(res, expect)

    d2 = pandas.DataFrame(
        {
            "symbol": ["RHAT"],
            "date": ["2006-01-05"],
            "price": 35.14,
            "qty": 100.0,
            "trans": "BUY",
        }
    )

    res2 = ops.eval_pandas(data_map={"stocks": d2}, eval_env=locals())

    assert data_algebra.test_util.equivalent_frames(res2, expect)
