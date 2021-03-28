import math
import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.env
import data_algebra.PostgreSQL
import data_algebra.SQLite
import data_algebra.util

# https://github.com/WinVector/data_algebra/blob/master/Examples/LogisticExample/ScoringExample.ipynb


def test_scoring_example():
    # set some things in our environment
    data_algebra.env.push_onto_namespace_stack(locals())

    d_local = data_algebra.default_data_model.pd.DataFrame(
        {
            "subjectID": [1, 1, 2, 2],
            "surveyCategory": [
                "withdrawal behavior",
                "positive re-framing",
                "withdrawal behavior",
                "positive re-framing",
            ],
            "assessmentTotal": [5, 2, 3, 4],
            "irrelevantCol1": ["irrel1"] * 4,
            "irrelevantCol2": ["irrel2"] * 4,
        }
    )

    scale = 0.237

    with data_algebra.env.Env(locals()) as env:
        ops = (
            TableDescription("d", ["subjectID", "surveyCategory", "assessmentTotal"])
            .extend({"probability": "(assessmentTotal * scale).exp()"})
            .extend({"total": "probability.sum()"}, partition_by="subjectID")
            .extend({"probability": "probability/total"})
            .extend({"sort_key": "-probability"})
            .extend(
                {"row_number": "_row_number()"},
                partition_by=["subjectID"],
                order_by=["sort_key"],
            )
            .select_rows("row_number == 1")
            .select_columns(["subjectID", "surveyCategory", "probability"])
            .rename_columns({"diagnosis": "surveyCategory"})
        )

    py_source = ops.to_python(strict=True, pretty=False)
    py_sourcep = ops.to_python(strict=True, pretty=True)

    # Pandas calculate
    res = ops.eval(data_map={"d": d_local})
    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "subjectID": [1, 2],
            "diagnosis": ["withdrawal behavior", "positive re-framing"],
            "probability": [0.670622, 0.558974],
        }
    )
    assert data_algebra.test_util.equivalent_frames(expect, res, float_tol=1e-3)

    # test database aspects

    db_model_p = data_algebra.PostgreSQL.PostgreSQLModel()
    db_model_s = data_algebra.SQLite.SQLiteModel()

    sql_p = ops.to_sql(db_model_p, pretty=True)
    sql_s = ops.to_sql(db_model_p, pretty=True)

    conn = sqlite3.connect(":memory:")
    db_model_s.prepare_connection(conn)

    db_model_s.insert_table(conn, d_local, "d")
    back = db_model_s.read_table(conn, "d")

    res_sql = db_model_s.read_query(conn, sql_s)

    assert data_algebra.test_util.equivalent_frames(expect, res_sql, float_tol=1e-3)

    # test instrumentation

    tables = ops.get_tables()
    cols_used = ops.columns_used()

    # be neat
    conn.close()
