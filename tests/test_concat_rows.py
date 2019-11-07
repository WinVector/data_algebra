import pandas
import numpy
import sqlite3

import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.yaml
import data_algebra.test_util


def test_concat_rows():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.SQLite.SQLiteModel()

    d1 = pandas.DataFrame({"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]})

    d2 = pandas.DataFrame({"x": [0, 4, None], "y": [2, 7, None]})

    ops4 = describe_table(d1, "d1").concat_rows(b=describe_table(d2, "d2"))

    assert data_algebra.test_util.formats_to_self(ops4)
    data_algebra.test_util.check_op_round_trip(ops4)

    res_pandas = ops4.eval_pandas(data_map={"d1": d1, "d2": d2}, eval_env=locals())

    expect = pandas.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, None, 0.0, 4.0, None],
            "y": [1.0, 2.0, None, 3.0, 2.0, 7.0, None],
            "source_name": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    sql = ops4.to_sql(db_model, pretty=True)

    conn = sqlite3.connect(":memory:")
    db_model.insert_table(conn, d1, "d1")
    db_model.insert_table(conn, d2, "d2")
    res_sqlite = db_model.read_query(conn, sql)
    # neaten up
    conn.close()

    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
