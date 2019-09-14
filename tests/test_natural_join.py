import pandas
import numpy
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.yaml
import sqlite3


def test_natural_join_columns():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    db_model = data_algebra.SQLite.SQLiteModel()

    d = pandas.DataFrame({"x": [-1, 0, 1, numpy.nan], "y": [1, 2, numpy.nan, 3]})

    d2 = pandas.DataFrame({"qq": [10, 20, 30], "y": [1.0, 2.0, 3.0], "x": [4, 5, 7]})

    ops4 = describe_table(d, "d").natural_join(
        b=describe_table(d2, "d2"), by=["y"], jointype="LEFT"
    )

    data_algebra.yaml.check_op_round_trip(ops4)

    res_pandas = ops4.eval_pandas(data_map={"d": d, "d2": d2}, eval_env=locals())

    expect = pandas.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 7.0],
            "y": [1.0, 2.0, numpy.nan, 3.0],
            "qq": [10.0, 20.0, numpy.nan, 30.0],
        }
    )

    assert data_algebra.util.equivalent_frames(expect, res_pandas)

    sql = ops4.to_sql(db_model, pretty=True)

    conn = sqlite3.connect(":memory:")
    db_model.insert_table(conn, d, "d")
    db_model.insert_table(conn, d2, "d2")
    res_sqlite = db_model.read_query(conn, sql)
    # neaten up
    conn.close()

    assert data_algebra.util.equivalent_frames(expect, res_sqlite)
