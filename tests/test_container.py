import data_algebra

from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
from data_algebra.op_container import OpC, one
import data_algebra.test_util

import data_algebra.MySQL


def test_container_1():
    d = data_algebra.default_data_model.pd.DataFrame(
        {
            "subjectID": [1, 1, 2, 2],
            "surveyCategory": [
                "withdrawal behavior",
                "positive re-framing",
                "withdrawal behavior",
                "positive re-framing",
            ],
            "assessmentTotal": [5.0, 2.0, 3.0, 4.0],
            "irrelevantCol1": ["irrel1"] * 4,
            "irrelevantCol2": ["irrel2"] * 4,
        }
    )

    scale = 0.237

    op_container = OpC()
    _ = op_container.column_namespace
    ops2 = (
        op_container.describe_table(d, "d")
            .extend({"probability": (_.assessmentTotal * scale).exp()})
            .extend({"total": _.probability.sum()}, partition_by="subjectID")
            .extend({"probability": _.probability / _.total})
            .extend({'ncat': one.sum()},
                    partition_by=["subjectID"],
                    )
            .extend(
                {"row_number": one.cumsum()},
                partition_by=["subjectID"],
                order_by=["probability"],
                reverse=["probability"],
            )
            .select_rows(_.row_number == 1)
            .select_columns(["subjectID", "surveyCategory", "probability", "ncat"])
            .rename_columns({"diagnosis": "surveyCategory"})
            .ops()
    )

    db_handle = data_algebra.MySQL.MySQLModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops2, pretty=True)
    assert isinstance(sql, str)
    # print(sql)

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "subjectID": [1, 2],
            "diagnosis": ["withdrawal behavior", "positive re-framing"],
            "probability": [0.670622, 0.558974],
            'ncat': [2, 2],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops2, data=d, expect=expect, float_tol=1e-4
    )