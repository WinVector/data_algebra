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

    expect = data_algebra.pd.DataFrame(
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

    d2 = data_algebra.pd.DataFrame(
        {
            "symbol": ["RHAT"],
            "date": ["2006-01-05"],
            "price": 35.14,
            "qty": 100.0,
            "trans": "BUY",
        }
    )

    res2 = ops.eval_pandas(data_map={"stocks": d2})

    assert data_algebra.test_util.equivalent_frames(res2, expect)


def test_sqllite_g2():
    ops = TableDescription(
        table_name='d',
        column_names=['col1', 'col2', 'col3']). \
        extend({
        'sum23': 'col2 + col3'
    }). \
        extend({
        'x': 1.0
    }). \
        extend({
        'x': 2.0
    }). \
        extend({
        'x': 3.0
    }). \
        extend({
        'x': 4.0
    }). \
        extend({
        'x': 5.0
    }). \
        project({'x': 'x.max()'},
                group_by=['sum23']). \
        extend({'ratio': 'x / sum23',
                'sum': 'x + sum23',
                'diff': 'x - sum23'}). \
        select_columns(['ratio', 'sum23', 'diff']). \
        select_rows('sum23 > 8'). \
        drop_columns(['sum23']). \
        rename_columns({'rat': 'ratio'}). \
        rename_columns({'rat': 'diff', 'diff': 'rat'}). \
        order_rows(['rat'])

    d = pandas.DataFrame({
        'col1': [1, 2, 2],
        'col2': [3, 4, 3],
        'col3': [4, 5, 7]
    })

    res_pandas = ops.transform(d)

    sql_model = data_algebra.SQLite.SQLiteModel()

    q = ops.to_sql(db_model=sql_model)

    conn = sqlite3.connect(':memory:')
    sql_model.prepare_connection(conn)
    sql_model.insert_table(conn, d, table_name='d')

    #conn.execute('CREATE TABLE res AS ' + q)
    #res_sql = sql_model.read_table(conn, 'res')
    res_sql = sql_model.read_query(conn, q)

    conn.close()

    assert data_algebra.test_util.equivalent_frames(res_pandas, res_sql, check_row_order=True)
