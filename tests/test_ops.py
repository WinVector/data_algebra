
import sqlite3

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util
import data_algebra.db_model
import data_algebra.SQLite


# simple direct tests of basic expressions

def test_ops():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
        'g': [1, 1, 2, 2],
    })
    d_orig = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
        'g': [1, 1, 2, 2],
    })
    td = describe_table(d, table_name='d')

    sql_model = data_algebra.SQLite.SQLiteModel()
    conn = sqlite3.connect(':memory:')
    sql_model.prepare_connection(conn)
    db_handle = data_algebra.db_model.DBHandle(sql_model, conn)
    tbl_map = {'d': db_handle.insert_table(d, table_name='d')}

    def check_ops(ops, expect=None, *, test_sql=True):
        res_pandas = ops.transform(d)
        if expect is not None:
            assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
        if test_sql:
            query = ops.to_sql(sql_model)
            res_db = sql_model.read_query(conn, query)
            assert data_algebra.test_util.equivalent_frames(res_db, res_pandas)
            if expect is not None:
                assert data_algebra.test_util.equivalent_frames(res_db, expect)
        assert data_algebra.test_util.equivalent_frames(d, d_orig)

    ops = td. \
        extend({'x': 'x == 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, True, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x != 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, False, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x < 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, False, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)
    ops = td. \
        extend({'x': 'x <= 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, True, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x > 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, False, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x >= 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, True, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 == x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, True, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 != x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, False, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 > x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, False, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)
    ops = td. \
        extend({'x': '2 >= x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [True, True, False, False],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 < x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, False, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 <= x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [False, True, True, True],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '-x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [-1, -2, -3, -4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x + 1'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2, 3, 4, 5],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '1 + x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2, 3, 4, 5],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x - 1'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [0, 1, 2, 3],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '1 - x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [0, -1, -2, -3],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x * 2'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2, 4, 6, 8],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2 * x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2, 4, 6, 8],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x / 2.0'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1/2, 2/2, 3/2, 4/2],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2.0 / x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2/1, 2/2, 2/3, 2/4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x // 2.0'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1//2, 2//2, 3//2, 4//2],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect, test_sql=False)

    ops = td. \
        extend({'x': '2.0 // x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2//1, 2//2, 2//3, 2//4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect, test_sql=False)

    ops = td. \
        extend({'x': 'x % 2.0'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1%2, 2%2, 3%2, 4%2],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '2.0 % x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2%1, 2%2, 2%3, 2%4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': 'x ** 2.0'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1**2, 2**2, 3**2, 4**2],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect, test_sql=False)  # TODO SQL translation

    ops = td. \
        extend({'x': '2.0 ** x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [2**1, 2**2, 2**3, 2**4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect, test_sql=False)  # TODO SQL translation

    ops = td. \
        extend({'x': '-x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [-1, -2, -3, -4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)

    ops = td. \
        extend({'x': '+x'})
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [+1, +2, +3, +4],
        'g': [1, 1, 2, 2],
    })
    check_ops(ops, expect)



    # clean up
    conn.close()
