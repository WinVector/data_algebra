from data_algebra.data_ops import *
from data_algebra.expr import frame, row_number
import data_algebra.test_util


def test_free_expr():
    scale = 0.237

    ops1 = (
        TableDescription("d", ["subjectID", "surveyCategory", "assessmentTotal"])
        .extend({"probability": (frame().assessmentTotal * scale).exp()})
        .extend({"total": frame().probability.sum()}, partition_by=["subjectID"])
        .extend({"probability": frame().probability / frame().total})
        .extend({"sort_key": -frame().probability})
        .extend(
            {"row_number": row_number()},
            partition_by=["subjectID"],
            order_by=["sort_key"],
        )
        .select_rows(frame().row_number == 1.0)
        .select_columns(["subjectID", "surveyCategory", "probability"])
        .rename_columns({"diagnosis": "surveyCategory"})
    )

    ops2 = (
        TableDescription("d", ["subjectID", "surveyCategory", "assessmentTotal"])
        .extend({"probability": f"(assessmentTotal * {scale}).exp()"})
        .extend({"total": "probability.sum()"}, partition_by=["subjectID"])
        .extend({"probability": "probability/total"})
        .extend({"sort_key": "-probability"})
        .extend(
            {"row_number": "_row_number()"},
            partition_by=["subjectID"],
            order_by=["sort_key"],
        )
        .select_rows("row_number == 1.0")
        .select_columns(["subjectID", "surveyCategory", "probability"])
        .rename_columns({"diagnosis": "surveyCategory"})
    )

    assert str(ops1) == str(ops2)

    f = frame

    ops3 = (
        TableDescription("d", ["subjectID", "surveyCategory", "assessmentTotal"])
        .extend({"probability": (f().assessmentTotal * scale).exp()})
        .extend({"total": f().probability.sum()}, partition_by=["subjectID"])
        .extend({"probability": f().probability / f().total})
        .extend({"sort_key": -f().probability})
        .extend(
            {"row_number": row_number()},
            partition_by=["subjectID"],
            order_by=["sort_key"],
        )
        .select_rows(f().row_number == 1.0)
        .select_columns(["subjectID", "surveyCategory", "probability"])
        .rename_columns({"diagnosis": "surveyCategory"})
    )

    assert str(ops1) == str(ops3)

    f = frame()

    ops4 = (
        TableDescription("d", ["subjectID", "surveyCategory", "assessmentTotal"])
        .extend({"probability": (f.assessmentTotal * scale).exp()})
        .extend({"total": f.probability.sum()}, partition_by=["subjectID"])
        .extend({"probability": f.probability / f.total})
        .extend({"sort_key": -f.probability})
        .extend(
            {"row_number": row_number()},
            partition_by=["subjectID"],
            order_by=["sort_key"],
        )
        .select_rows(f.row_number == 1.0)
        .select_columns(["subjectID", "surveyCategory", "probability"])
        .rename_columns({"diagnosis": "surveyCategory"})
    )

    assert str(ops1) == str(ops4)
