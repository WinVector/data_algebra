from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util


def test_user_sql():
    # based on:
    # https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/User_SQL.ipynb
    d1 = data_algebra.default_data_model.pd.DataFrame(
        {"g": ["a", "a", "b", "b"], "v1": [1, 2, 3, 4], "v2": [5, 6, 7, 8],}
    )

    ops = SQLNode(
        sql="""
              SELECT
                *,
                v1 * v2 AS v3
              FROM
                d1
            """,
        column_names=["g", "v1", "v2", "v3"],
        view_name="derived_results",
    ).extend({"v4": "v3 + v1"})

    expect = d1.copy()
    expect["v3"] = expect["v1"] * expect["v2"]
    expect["v4"] = expect["v3"] + expect["v1"]
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d1, table_name="d1")

    res_sqllite = sqlite_handle.read_query(ops)
    res_pandas = ops.eval({"derived_results": sqlite_handle})
    sqlite_handle.close()

    assert data_algebra.test_util.equivalent_frames(res_sqllite, expect)
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

