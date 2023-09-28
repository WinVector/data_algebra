import sqlite3

import numpy

import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util
import data_algebra.db_model
import data_algebra.MySQL
import data_algebra.SQLite


# simple direct tests of basic expressions


def test_ops():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    d_orig = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    td = describe_table(d, table_name="d")

    sql_model = data_algebra.SQLite.SQLiteModel()
    conn = sqlite3.connect(":memory:")
    sql_model.prepare_connection(conn)
    db_handle = data_algebra.db_model.DBHandle(db_model=sql_model, conn=conn)
    tbl_map = {"d": db_handle.insert_table(d, table_name="d")}

    def check_ops(ops, expect):
        data_algebra.test_util.check_transform(
            ops=ops,
            data=d,
            expect=expect,
            try_on_DBs=False,
        )
        res_pandas = ops.transform(d)
        assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
        if True or test_sql:
            query = ops.to_sql(sql_model)
            res_db = sql_model.read_query(conn, query)
            assert data_algebra.test_util.equivalent_frames(res_db, res_pandas)
            if expect is not None:
                assert data_algebra.test_util.equivalent_frames(res_db, expect)
        assert data_algebra.test_util.equivalent_frames(d, d_orig)

    ops = td.extend({"x": "x == 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [False, True, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x != 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [True, False, True, True], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x < 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [True, False, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)
    ops = td.extend({"x": "x <= 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [True, True, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x > 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [False, False, True, True],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x >= 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [False, True, True, True], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 == x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [False, True, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 != x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [True, False, True, True], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 > x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [True, False, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)
    ops = td.extend({"x": "2 >= x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [True, True, False, False],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 < x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [False, False, True, True],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 <= x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [False, True, True, True], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "-x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [-1, -2, -3, -4], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x + 1"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [2, 3, 4, 5], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "1 + x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [2, 3, 4, 5], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x - 1"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [0, 1, 2, 3], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "1 - x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [0, -1, -2, -3], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x * 2"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [2, 4, 6, 8], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2 * x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [2, 4, 6, 8], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x / 2.0"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1 / 2, 2 / 2, 3 / 2, 4 / 2],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2.0 / x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [2 / 1, 2 / 2, 2 / 3, 2 / 4],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x // 2.0"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1 // 2, 2 // 2, 3 // 2, 4 // 2],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2.0 // x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [2 // 1, 2 // 2, 2 // 3, 2 // 4],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x % 2.0"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1 % 2, 2 % 2, 3 % 2, 4 % 2],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2.0 % x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [2 % 1, 2 % 2, 2 % 3, 2 % 4],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "x ** 2.0"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1 ** 2, 2 ** 2, 3 ** 2, 4 ** 2],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "2.0 ** x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4],
            "x2": [0.1, 0.2, 0.3, 0.4],
            "g": [1, 1, 2, 2],
        }
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "-x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [-1, -2, -3, -4], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "-(1 + x)"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [-2, -3, -4, -5], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    ops = td.extend({"x": "+x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [+1, +2, +3, +4], "x2": [0.1, 0.2, 0.3, 0.4], "g": [1, 1, 2, 2],}
    )
    check_ops(ops, expect)

    fns_to_check = {
        "abs": numpy.abs,
        "arccos": numpy.arccos,
        # 'arccosh': numpy.arccosh,  # generates warning on values we are feeding
        "arcsin": numpy.arcsin,
        "arcsinh": numpy.arcsinh,
        "arctan": numpy.arctan,
        "arctanh": numpy.arctanh,
        "ceil": numpy.ceil,
        "cos": numpy.cos,
        "cosh": numpy.cosh,
        "exp": numpy.exp,
        # "expm1": numpy.expm1,
        "floor": numpy.floor,
        "log": numpy.log,
        "log10": numpy.log10,
        # "log1p": numpy.log1p,
        "round": numpy.round,
        "sin": numpy.sin,
        "sinh": numpy.sinh,
        "sqrt": numpy.sqrt,
        "tanh": numpy.tanh,
    }
    for k, f in fns_to_check.items():
        # print(k)
        ops = td.extend({"x3": "x2." + k + "()"})
        expect = data_algebra.data_model.default_data_model().pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "x2": [0.1, 0.2, 0.3, 0.4],
                "g": [1, 1, 2, 2],
                "x3": [f(xi) for xi in [0.1, 0.2, 0.3, 0.4]],
            }
        )
        check_ops(ops, expect)

    # clean up
    conn.close()
