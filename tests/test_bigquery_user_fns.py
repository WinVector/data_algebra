import datetime

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util

import data_algebra.BigQuery


def test_bigquery_user_fns_integrated():
    pd = data_algebra.data_model.default_data_model().pd
    datetime_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y-%m-%d"

    d = pd.DataFrame(
        {
            "as_int64_col": ["50", "3"],
            "as_str_col": [1.2, 2],
            "trimstr_col": ["abdefg", "0123456"],
            "coalesce_0_col": [1, None],
            "coalesce_col_0": ["a", None],
            "coalesce_col_1": ["b", "c"],
            "parse_datetime_col": ["2000-01-01 12:13:21", "2020-04-05 14:03:00"],
            "parse_date_col": ["2000-03-01", "2020-04-05"],
            "input_datetime_col_0": pd.to_datetime(
                pd.Series(["2010-01-01 12:13:21", "2030-04-05 14:03:00"]),
                format=datetime_format,
            ),
            "input_datetime_col_1": pd.to_datetime(
                pd.Series(["2010-01-01 12:11:21", "2030-04-06 14:03:00"]),
                format=datetime_format,
            ),
            "input_date_col_0": pd.to_datetime(
                pd.Series(["2000-01-02", "2035-04-05"]), format=date_format
            ).dt.date,
            "input_date_col_1": pd.to_datetime(
                pd.Series(["2000-01-02", "2035-05-05"]), format=date_format
            ).dt.date,
        }
    )

    ops = describe_table(d, table_name="d").extend(
        {
            "as_int64_res": "as_int64_col.as_int64()",
            "as_str_res": "as_str_col.as_str()",
            "trimstr_res": "trimstr_col.trimstr(0, 2)",
            "coalesce_0_res": "coalesce_0_col %?% 0",
            "coalesce_res": "coalesce_col_0 %?% coalesce_col_1",
            "datetime_to_date_res": "input_datetime_col_0.datetime_to_date()",
            "parse_datetime_res": 'parse_datetime_col.parse_datetime("%Y-%m-%d %H:%M:%S")',
            "parse_date_res": 'parse_date_col.parse_date("%Y-%m-%d")',
            "format_datetime_res": 'input_datetime_col_0.format_datetime("%Y-%m-%d %H:%M:%S")',
            "format_date_res": 'input_date_col_0.format_date("%Y-%m-%d")',
            "dayofweek_res": "input_date_col_0.dayofweek()",
            "dayofyear_res": "input_date_col_0.dayofyear()",
            "dayofmonth_res": "input_date_col_0.dayofmonth()",
            "weekofyear_res": "input_date_col_0.weekofyear()",
            "month_res": "input_date_col_0.month()",
            "quarter_res": "input_date_col_0.quarter()",
            "year_res": "input_date_col_0.year()",
            "timestamp_diff_res": "input_datetime_col_0.timestamp_diff(input_datetime_col_1)",
            "date_diff_res": "input_date_col_0.date_diff(input_date_col_1)",
            "base_Sunday_res": "input_date_col_0.base_Sunday()",
        }
    )

    # print(data_algebra.util.pandas_to_example_str(ops.transform(d)))

    res_pandas = ops.transform(d)

    expect = pd.DataFrame(
        {
            "as_int64_col": ["50", "3"],
            "as_str_col": [1.2, 2.0],
            "trimstr_col": ["abdefg", "0123456"],
            "coalesce_0_col": [1.0, None],
            "coalesce_col_0": ["a", None],
            "coalesce_col_1": ["b", "c"],
            "parse_datetime_col": ["2000-01-01 12:13:21", "2020-04-05 14:03:00"],
            "parse_date_col": ["2000-03-01", "2020-04-05"],
            "input_datetime_col_0": [
                pd.Timestamp("2010-01-01 12:13:21"),
                pd.Timestamp("2030-04-05 14:03:00"),
            ],
            "input_datetime_col_1": [
                pd.Timestamp("2010-01-01 12:11:21"),
                pd.Timestamp("2030-04-06 14:03:00"),
            ],
            "input_date_col_0": [datetime.date(2000, 1, 2), datetime.date(2035, 4, 5)],
            "input_date_col_1": [datetime.date(2000, 1, 2), datetime.date(2035, 5, 5)],
            "as_int64_res": [50, 3],
            "as_str_res": ["1.2", "2.0"],
            "trimstr_res": ["ab", "01"],
            "coalesce_0_res": [1.0, 0.0],
            "coalesce_res": ["a", "c"],
            "datetime_to_date_res": [
                datetime.date(2010, 1, 1),
                datetime.date(2030, 4, 5),
            ],
            "parse_datetime_res": [
                pd.Timestamp("2000-01-01 12:13:21"),
                pd.Timestamp("2020-04-05 14:03:00"),
            ],
            "parse_date_res": [datetime.date(2000, 3, 1), datetime.date(2020, 4, 5)],
            "format_datetime_res": ["2010-01-01 12:13:21", "2030-04-05 14:03:00"],
            "format_date_res": ["2000-01-02", "2035-04-05"],
            "dayofweek_res": [1, 5],
            "dayofyear_res": [2, 95],
            "dayofmonth_res": [2, 5],
            "weekofyear_res": [1, 13],
            "month_res": [1, 4],
            "quarter_res": [1, 2],
            "year_res": [2000, 2035],
            "timestamp_diff_res": [120.0, -86400.0],
            "date_diff_res": [0, -30],
            "base_Sunday_res": [datetime.date(2000, 1, 2), datetime.date(2035, 4, 1)],
        }
    )

    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    test_handle = data_algebra.BigQuery.BigQueryModel().db_handle(conn=None)
    test_sql = test_handle.to_sql(ops)
    assert isinstance(test_sql, str)

    if data_algebra.test_util.test_BigQuery:
        with data_algebra.BigQuery.example_handle() as db_handle:
            if db_handle is not None:
                d_remote = db_handle.insert_table(
                    d, table_name="d", allow_overwrite=True
                )
                res_db = db_handle.read_query(ops)
                db_handle.drop_table("d")
                # times come back with UTC timezone and some other differences
                # match_cols = (res_db == expect).all(axis=0)
                # list(match_cols.index[match_cols == False])
                # convert a few things as we check
                assert res_db.shape == expect.shape
                assert set(expect.columns) == set(res_db.columns)
                res_db_check = res_db.loc[:, list(expect.columns)].copy()
                expect_types = [str(type(v)) for v in expect.iloc[0, :]]
                res_db_check["timestamp_diff_res"] = (
                    1.0 * res_db_check["timestamp_diff_res"]
                )  # convert to float
                db_types = [str(type(v)) for v in res_db_check.iloc[0, :]]
                assert all([a == b for a, b in zip(expect_types, db_types)])
                assert isinstance(res_db["as_str_res"][0], str)
                assert all(
                    res_db["as_str_res"].astype(float)
                    == expect["as_str_res"].astype(float)
                )
                # check all but some input columns and things we have checked
                expect_check = expect.copy()
                for c in ["input_datetime_col_0", "input_datetime_col_1", "as_str_res"]:
                    del expect_check[c]
                    del res_db_check[c]
                assert data_algebra.test_util.equivalent_frames(
                    expect_check, res_db_check
                )
