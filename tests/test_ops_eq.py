import data_algebra
import data_algebra.test_util
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra


def test_ops_eq():
    scale = 0.237

    ops_1 = (
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

    ops_2 = (
        TableDescription(
            table_name="d",
            column_names=["subjectID", "surveyCategory", "assessmentTotal"],
        )
        .extend({"probability": "(assessmentTotal * 0.237).exp()"})
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

    assert ops_1 == ops_2

    ops_3 = (
        TableDescription(
            table_name="d",
            column_names=["subjectID", "surveyCategory", "assessmentTotal"],
        )
        .extend({"probability": "(assessmentTotal * 0.5).exp()"})
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

    assert ops_1 != ops_3

    ops_4 = (
        TableDescription(
            table_name="d",
            column_names=["subjectID", "surveyCategory", "assessmentTotal"],
        )
        .extend({"probability": "(assessmentTotal * 0.237).exp()"})
        .extend({"total": "probability.sum()"}, partition_by="subjectID")
        .extend({"probability": "probability/total"})
        .extend({"sort_key": "-probability"})
        .extend(
            {"row_number": "_row_number()"},
            partition_by=["subjectID"],
            order_by=["sort_key"],
        )
        .select_rows("row_number == 2")
        .select_columns(["subjectID", "surveyCategory", "probability"])
        .rename_columns({"diagnosis": "surveyCategory"})
    )

    assert ops_1 != ops_4
