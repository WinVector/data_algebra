import data_algebra
import data_algebra.db_model
from data_algebra.data_ops import *
import data_algebra.BigQuery


def test_set_quoting_1():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, -2, 3, -4]})
    targets = [-2, -5]

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name="d").extend({"select": f"x.is_in({targets})"})

    sql_format_options = data_algebra.db_model.SQL_Format_Options(
        use_with=True,
        annotate=False,
        sql_indent=' ',
        initial_commas=False)
    sql = bq_handle.to_sql(ops, sql_format_options=sql_format_options)
    assert "'" not in sql
    assert '"' not in sql


def test_set_quoting_2():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, -2, 3, -4]})

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name="d").extend({"select": f"x.is_in({-5, 1+2})"})

    sql_format_options = data_algebra.db_model.SQL_Format_Options(
        use_with=True,
        annotate=False,
        sql_indent=' ',
        initial_commas=False)
    sql = bq_handle.to_sql(ops, sql_format_options=sql_format_options)
    assert "'" not in sql
    assert '"' not in sql
    assert "3" in sql
