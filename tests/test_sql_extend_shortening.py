
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.test_util

import data_algebra.MySQL


def test_sql_extend_shortening_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "g": ["a", "b", "b", "a", "a"], "o": [5, 4, 3, 2, 1],}
    )

    ops = (
        describe_table(d, table_name="d")
        .extend({"sx": "x.sum()"})
        .extend({"og1": "(1).cumsum()"}, partition_by=["g"], order_by=["x"])
        .extend(
            {"og2": "(1).cumsum()"}, partition_by=["g"], order_by=["x"], reverse=["x"]
        )
    )

    # show op-chain is non shortened, as we don't do that in for the Pandas path
    assert isinstance(ops, ExtendNode)
    assert isinstance(ops.sources[0], ExtendNode)
    assert isinstance(ops.sources[0].sources[0], ExtendNode)

    # show the SQL is shortened
    db_handle = data_algebra.MySQL.MySQLModel().db_handle(conn=None)
    sql = db_handle.to_sql(ops)
    assert isinstance(sql, str)
    assert sql.lower().count("select") == 1

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "g": ["a", "b", "b", "a", "a"],
            "o": [5, 4, 3, 2, 1],
            "sx": [15, 15, 15, 15, 15],
            "og1": [1, 1, 2, 2, 3],
            "og2": [3, 2, 1, 2, 1],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, float_tol=1e-4
    )

    # check again after passing through string form, this is check partition=1 notation isn't hurting things

    ops_2 = eval(str(ops))

    assert isinstance(ops_2, ExtendNode)
    assert isinstance(ops_2.sources[0], ExtendNode)
    assert isinstance(ops_2.sources[0].sources[0], ExtendNode)

    sql_2 = db_handle.to_sql(ops_2)
    assert isinstance(sql_2, str)
    assert sql_2.lower().count("select") == 1

    data_algebra.test_util.check_transform(
        ops=ops_2, data=d, expect=expect, float_tol=1e-4
    )


def test_ops_extend_shortening_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4, 5], "g": ["a", "b", "b", "a", "a"], "o": [5, 4, 3, 2, 1],}
    )

    ops = (
        describe_table(d, table_name="d")
        .extend({"sx": "x.sum()"})
        .extend({"og1": "(1).sum()"})
        .extend({"og2": "(1).sum()"})
    )

    # show op-chain is shortened
    assert isinstance(ops, ExtendNode)
    assert isinstance(ops.sources[0], TableDescription)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "g": ["a", "b", "b", "a", "a"],
            "o": [5, 4, 3, 2, 1],
            "sx": [15, 15, 15, 15, 15],
            "og1": [5, 5, 5, 5, 5],
            "og2": [5, 5, 5, 5, 5],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect, float_tol=1e-4
    )
