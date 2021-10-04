
import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *
# import data_algebra.SQLite


def test_sqlite_joins():
    # No none keys, as they treat missingness different
    # in Pandas and databases
    d1 = data_algebra.default_data_model.pd.DataFrame({
        'g': ['a', 'a', 'b', 'b', 'b'],
        'v1': [1, None, 3, 4, None],
        'v2': [None, 1, 2, 7, 8],
    })

    d2 = data_algebra.default_data_model.pd.DataFrame({
        'g': ['c', 'b', 'b'],
        'v1': [None, 1, None],
        'v2': [1, None, 2],
    })

    # sqlite_handle = data_algebra.SQLite.example_handle()

    ops_1 = descr(d1=d1).natural_join(b=descr(d2=d2), by=['g'], jointype='left')
    expect_1 = data_algebra.default_data_model.pd.DataFrame({
        'g': ['a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'],
        'v1': [1.0, None, 3.0, 3.0, 4.0, 4.0, 1.0, None],
        'v2': [None, 1.0, 2.0, 2.0, 7.0, 7.0, 8.0, 8.0],
        })
    res_1 = ops_1.eval({'d1': d1, 'd2': d2})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    # print(sqlite_handle.to_sql(ops_1))
    data_algebra.test_util.check_transform(
        ops=ops_1,
        data={'d1': d1, 'd2': d2},
        expect=expect_1)

    ops_2 = descr(d1=d1).natural_join(b=descr(d2=d2), by=['g'], jointype='right')
    expect_2 = data_algebra.default_data_model.pd.DataFrame({
        'g': ['c', 'b', 'b', 'b', 'b', 'b', 'b'],
        'v1': [None, 3.0, 4.0, 1.0, 3.0, 4.0, None],
        'v2': [1.0, 2.0, 7.0, 8.0, 2.0, 7.0, 8.0],
        })
    res_2 = ops_2.eval({'d1': d1, 'd2': d2})
    assert data_algebra.test_util.equivalent_frames(res_2, expect_2)
    # print(sqlite_handle.to_sql(ops_2))
    data_algebra.test_util.check_transform(
        ops=ops_2,
        data={'d1': d1, 'd2': d2},
        expect=expect_2)

    # check test is strong enough
    assert not data_algebra.test_util.equivalent_frames(expect_1, expect_2)

    # naive reversal (interferes with coalesce)
    ops_n = descr(d2=d2).natural_join(b=descr(d1=d1), by=['g'], jointype='left')
    res_n = ops_n.eval({'d1': d1, 'd2': d2})
    assert not data_algebra.test_util.equivalent_frames(res_n, expect_1)
    assert not data_algebra.test_util.equivalent_frames(res_n, expect_2)

    # sqlite_handle.close()