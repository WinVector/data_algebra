
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.db_model
import data_algebra.SQLite


def test_dag_elim():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x + 1'})
            .natural_join(
                b=(
                    descr(d=d)
                        .extend({'y': 'x + 1'})
                ),
                by=['x'],
                jointype='left',
            )
    )
    db_model = data_algebra.SQLite.SQLiteModel()
    sql = db_model.to_sql(
        ops,
        sql_format_options=data_algebra.db_model.SQLFormatOptions(use_with=True, annotate=False)
    )
    # assert sql.count('"d"') == 1  # show table is referenced exactly once
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
