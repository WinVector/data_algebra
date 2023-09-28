import data_algebra
import data_algebra.db_model
import data_algebra.sql_format_options
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.BigQuery

import pytest


def test_set_quoting_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, -2, 3, -4]})
    targets = [-2, -5]

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name="d").extend({"select": f"x.is_in({targets})"})

    sql_format_options = data_algebra.sql_format_options.SQLFormatOptions(
        use_with=True, annotate=False, sql_indent=" ", initial_commas=False
    )
    sql = bq_handle.to_sql(ops, sql_format_options=sql_format_options)
    assert "'" not in sql
    assert '"' not in sql


def test_set_quoting_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, -2, 3, -4]})

    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(None)

    ops = describe_table(d, table_name="d").extend({"select": "x.is_in({-5, 3})"})

    sql_format_options = data_algebra.sql_format_options.SQLFormatOptions(
        use_with=True, annotate=False, sql_indent=" ", initial_commas=False
    )
    sql = bq_handle.to_sql(ops, sql_format_options=sql_format_options)
    assert "'" not in sql
    assert '"' not in sql
    assert "3" in sql


def test_set_quoting_inhom():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, -2, 3, -4]})
    with pytest.raises(AssertionError):
        describe_table(d, table_name="d").extend({"select": "x.is_in({-5, 'a'})"})


def test_set_quoting_flex():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a", None]})
    ops = describe_table(d, table_name="d").extend({"select": "x.is_in({'a'})"})
    ops.transform(d)


def test_set_quoting_mismatch():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a", "b"]})
    ops = describe_table(d, table_name="d").extend({"select": "x.is_in({-5, -2})"})
    with pytest.raises(TypeError):
        ops.transform(d)


def test_set_quoting_exclude_none():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": ["a", "b"]})
    with pytest.raises(Exception):
        describe_table(d, table_name="d").extend({"select": "x.is_in({-5, None})"})
