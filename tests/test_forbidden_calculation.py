import sqlite3

import pytest

import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util


def test_forbidden_calculation():
    td = TableDescription(table_name="d", column_names=["a", "b", "c"])

    # test using undefined column
    with pytest.raises(ValueError):
        td.rename_columns({"a": "x"})

    # test colliding with know column
    with pytest.raises(ValueError):
        td.rename_columns({"a": "b"})

    # test swaps don't show up in forbidden
    ops1 = td.rename_columns({"a": "b", "b": "a"})

    # test new column creation triggers forbidden annotation
    ops2 = td.rename_columns({"e": "a"})

    # test merge
    ops3 = td.rename_columns({"e": "a"}).rename_columns({"f": "b"})

    # test composition
    ops4 = td.rename_columns({"e": "a"}).rename_columns({"a": "b"})


def test_calc_interface():
    td = TableDescription(table_name="d", column_names=["a"])
    ops = td.rename_columns({"b": "a"})

    d_good = data_algebra.data_model.default_data_model().pd.DataFrame({"a": [1]})
    d_extra = data_algebra.data_model.default_data_model().pd.DataFrame({"a": [1], "b": [2]})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"b": [1]})

    # check that table-defs narrow data
    res0 = td.transform(d_extra)
    assert data_algebra.test_util.equivalent_frames(res0, d_good)
    data_algebra.test_util.check_transform(ops=td, data=d_extra, expect=d_good)

    res1 = ops.transform(d_good)
    assert data_algebra.test_util.equivalent_frames(res1, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d_good, expect=expect)

    # act on interface should be strict
    with pytest.raises(AssertionError):
        ops.act_on(d_extra)

    # allowed through normal path
    res2 = ops.transform(d_extra)
    assert data_algebra.test_util.equivalent_frames(res2, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d_extra, expect=expect)

    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d_extra, table_name="d")

    sql = ops.to_sql(db_model)
    # Note: extra columns during execution is not an error.
    res_db_bad = db_model.read_query(conn, sql)
    data_model = {"d": [c for c in db_model.read_table(conn, "d", limit=1).columns]}
    conn.close()

    assert data_algebra.test_util.equivalent_frames(res_db_bad, expect)

    with pytest.raises(ValueError):
        ops.check_constraints(data_model, strict=True)

    ops.check_constraints(data_model, strict=False)

    # the examples we are interested in

    ops_first = td.extend({"x": 1})
    res_first = ops_first.transform(d_extra)
    expect_first = data_algebra.data_model.default_data_model().pd.DataFrame({"a": [1], "x": [1],})
    assert data_algebra.test_util.equivalent_frames(res_first, expect_first)
