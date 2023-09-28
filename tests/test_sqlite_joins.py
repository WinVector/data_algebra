import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.BigQuery
import data_algebra.SQLite
import data_algebra.MySQL
from data_algebra.sql_format_options import SQLFormatOptions


def test_sqlite_joins_left_to_right():
    # No none keys, as they treat missingness different
    # in Pandas and databases
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b"],
            "v1": [1, None, 3, 4, None],
            "v2": [None, 1, 2, 7, 8],
        }
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"g": ["c", "b", "b"], "v1": [None, 1, None], "v2": [1, None, 2],}
    )
    ops_1 = descr(d1=d1).natural_join(b=descr(d2=d2), by=["g"], jointype="left")
    expect_1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b", "b", "b", "b"],
            "v1": [1.0, None, 3.0, 3.0, 4.0, 4.0, 1.0, None],
            "v2": [None, 1.0, 2.0, 2.0, 7.0, 7.0, 8.0, 8.0],
        }
    )
    res_1 = ops_1.eval({"d1": d1, "d2": d2})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    data_algebra.test_util.check_transform(
        ops=ops_1,
        data={"d1": d1, "d2": d2},
        expect=expect_1,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )
    ops_2 = descr(d1=d1).natural_join(b=descr(d2=d2), by=["g"], jointype="right")
    expect_2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["c", "b", "b", "b", "b", "b", "b"],
            "v1": [None, 3.0, 4.0, 1.0, 3.0, 4.0, None],
            "v2": [1.0, 2.0, 7.0, 8.0, 2.0, 7.0, 8.0],
        }
    )
    res_2 = ops_2.eval({"d1": d1, "d2": d2})
    assert data_algebra.test_util.equivalent_frames(res_2, expect_2)
    data_algebra.test_util.check_transform(
        ops=ops_2,
        data={"d1": d1, "d2": d2},
        expect=expect_2,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )
    # check test is strong enough
    assert not data_algebra.test_util.equivalent_frames(expect_1, expect_2)
    # naive reversal (interferes with coalesce)
    ops_n = descr(d2=d2).natural_join(b=descr(d1=d1), by=["g"], jointype="left")
    res_n = ops_n.eval({"d1": d1, "d2": d2})
    assert not data_algebra.test_util.equivalent_frames(res_n, expect_1)
    assert not data_algebra.test_util.equivalent_frames(res_n, expect_2)


def test_sqlite_joins_simulate_full_join():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b"],
            "v1": [1, None, 3, 4, None],
            "v2": [None, 1, None, 7, 8],
        }
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"g": ["c", "b", "b"], "v1": [None, 1, None], "v2": [1, None, 2],}
    )
    join_columns = ["g"]
    ops = descr(d1=d1).natural_join(b=descr(d2=d2), by=join_columns, jointype="full")
    res_pandas = ops.eval({"d1": d1, "d2": d2})
    data_algebra.test_util.check_transform(
        ops=ops,
        data={"d1": d1, "d2": d2},
        expect=res_pandas,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )
    ops_simulate = (
        # get shared key set
        descr(d1=d1)
        .project({}, group_by=join_columns)
        .concat_rows(b=descr(d2=d2).project({}, group_by=join_columns), id_column=None,)
        .project({}, group_by=join_columns)
        # simulate full join with left joins
        .natural_join(b=descr(d1=d1), by=join_columns, jointype="left")
        .natural_join(b=descr(d2=d2), by=join_columns, jointype="left")
    )
    db_model = data_algebra.BigQuery.BigQueryModel()
    sql = db_model.to_sql(
        ops_simulate,
        sql_format_options=data_algebra.sql_format_options.SQLFormatOptions(
            use_with=True,
            annotate=False,
            use_cte_elim=True,
        )
    )
    assert isinstance(sql, str)
    data_algebra.test_util.check_transform(
        ops=ops_simulate,
        data={"d1": d1, "d2": d2},
        expect=res_pandas,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )


def test_sqlite_joins_staged():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "b"],
            "v1": [1, None, 3, 4, None],
            "v2": [None, 1, None, 7, 8],
        }
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"g": ["c", "b", "b"], "v1": [None, 1, None], "v2": [1, None, 2],}
    )
    join_columns = ["g"]
    ops = (
        descr(d1=d1)
        .extend({"v1": "v1 * v1"})
        .natural_join(
            b=(descr(d2=d2).extend({"v2": "v2 + v2"})), by=join_columns, jointype="full"
        )
    )
    res_pandas = ops.eval({"d1": d1, "d2": d2})
    data_algebra.test_util.check_transform(
        ops=ops,
        data={"d1": d1, "d2": d2},
        expect=res_pandas,
        models_to_skip={
            data_algebra.MySQL.MySQLModel(),
        },
    )
