
import numpy

import pytest
import sqlite3

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util
import data_algebra.SQLite


q1 = """
SELECT "x",
       "z",
       "q",
       "q" + 3 AS "h"
FROM
  (SELECT "x",
          "z",
          "z" + 2 AS "q"
   FROM
     (SELECT "x",
             "x" + 1 AS "z"
      FROM "d") "extend_1") "extend_2"
"""


q2 = """
WITH
    "extend_1" AS 
    (SELECT "x",
          "x" + 1 AS "z"
     FROM "d"),
    "extend_2" AS (SELECT "x",
          "z",
          "z" + 2 AS "q"
     FROM "extend_1"
    )
    SELECT "x",
           "z",
           "q",
           "q" + 3 AS "h"
    FROM "extend_2"
"""


def test_with_query_example_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3]
    })
    ops = describe_table(d, table_name='d') .\
        extend({'z': 'x + 1'}) .\
        extend({'q': 'z + 2'}) .\
        extend({'h': 'q + 3'})

    res_pandas = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3],
        'z': [2, 3, 4],
        'q': [4, 5, 6],
        'h': [7, 8, 9]
    })

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    db_model = data_algebra.SQLite.SQLiteModel()
    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_handle = db_model.db_handle(conn)
        db_handle.insert_table(d, table_name='d')
        sql_regular = db_handle.to_sql(ops, pretty=True, use_with=False)
        res_regular = db_handle.read_query(sql_regular)
        sql_with = db_handle.to_sql(ops, pretty=True, use_with=True)
        res_with = db_handle.read_query(sql_with)

    assert data_algebra.test_util.equivalent_frames(res_regular, expect)
    assert data_algebra.test_util.equivalent_frames(res_with, expect)
