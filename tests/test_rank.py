

import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.SQLite
import data_algebra.MySQL
import data_algebra.SparkSQL
import data_algebra.PostgreSQL
import data_algebra.BigQuery


def test_rank_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [1, 2, 1],
    })
    ops = (
        descr(d=d)
            .extend({'x_rank': 'x.rank()'})
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'x': [1, 2, 1],
        'x_rank': [1.5, 3.0, 1.5],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            data_algebra.SQLite.SQLiteModel(),
            data_algebra.PostgreSQL.PostgreSQLModel(),
            data_algebra.BigQuery.BigQueryModel(),
            data_algebra.MySQL.MySQLModel(),
            data_algebra.SparkSQL.SparkSQLModel(),  # cumulative implementation (nice but not same as python)
        })
