
import data_algebra
import data_algebra.test_util
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.PostgreSQL
import data_algebra.SQLite
import data_algebra.util

# https://github.com/WinVector/data_algebra/blob/master/Examples/LogisticExample/ScoringExample.ipynb


def test_scoring_example():
    d_local = data_algebra.data_model.default_data_model().pd.DataFrame(
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

    ops = (
        TableDescription(
            table_name="d",
            column_names=["subjectID", "surveyCategory", "assessmentTotal"],
        )
        .extend({"probability": f"(assessmentTotal * {scale}).exp()"})
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

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "subjectID": [1, 2],
            "diagnosis": ["withdrawal behavior", "positive re-framing"],
            "probability": [0.670622, 0.558974],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data=d_local, expect=expect, float_tol=1e-3,
    )

    # test instrumentation

    tables = ops.get_tables()
    cols_used = ops.columns_used()
