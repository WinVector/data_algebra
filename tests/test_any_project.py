
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.SQLite
import data_algebra.MySQL
import data_algebra.SparkSQL
import data_algebra.PostgreSQL


def test_any_project():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [.1, .1, .3, .4],
        'g': ['a', 'a', 'b', 'ccc'],
    })
    ops = (
        descr(d=d)
            .project(
                {'new_column': 'x.any_value()'},
                group_by=['g'])
            .order_rows(['g'])
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'g': ['a', 'b', 'ccc'],
        'new_column': [0.1, 0.3, 0.4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    # TODO: turn this next step on for more dbs
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            data_algebra.SQLite.SQLiteModel(),
            data_algebra.MySQL.MySQLModel(),
            data_algebra.SparkSQL.SparkSQLModel(),
            data_algebra.PostgreSQL.PostgreSQLModel(),
        })
