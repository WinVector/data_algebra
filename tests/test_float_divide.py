

import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.util
import data_algebra.SQLite
import data_algebra.MySQL
import data_algebra.SparkSQL


def test_float_divide_needed():
    # need %/% as databases often use integer arithmetic
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'a': [1, 1, 1],
        'b': [1, 2, 0]
    })
    ops = (
        descr(d=d)
            .extend({
                'r1': 'a/b',
            })
    )
    res_pandas = ops.transform(d)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_sqlite = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert not data_algebra.test_util.equivalent_frames(res_sqlite, res_pandas)


def test_float_divide_works():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'a': [1, 1, 1, 0],
        'b': [1, 2, 0, 0],
    })
    ops = (
        descr(d=d)
            .extend({
                'r': 'a %/% b'
                })
            .extend({  # sqlite returns nan for 1/0, not +inf
                'r': f'r.is_bad().where(None, r)'
                })
    )
    res_pandas = ops.transform(d)
    expect = pd.DataFrame({
        'a': [1, 1, 1, 0],
        'b': [1, 2, 0, 0],
        'r': [1.0, 0.5, None, None],
        })
    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_sqlite = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(expect, res_sqlite)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            data_algebra.MySQL.MySQLModel(),  # sqlalchemy won't insert inf
            data_algebra.SparkSQL.SparkSQLModel(),  # probably not inserting values
            },
        )
