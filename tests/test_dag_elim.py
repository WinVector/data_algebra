
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
    n_d = sql.count('"d"')
    assert n_d > 0
    assert n_d <= 2
    # assert n_d == 1  # show table is referenced exactly once
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


# some variations that could trigger an earlier bug

def test_dag_elim_bee():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x'})
            .natural_join(
                b=(
                    descr(d=d)
                        .extend({'y': 'x'})
                ),
                by=['x'],
                jointype='left',
            )
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_bet():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x'})
            .natural_join(
                b=descr(d=d),
                by=['x'],
                jointype='left',
            )
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_bte():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .natural_join(
                b=(
                    descr(d=d)
                        .extend({'y': 'x'})
                ),
                by=['x'],
                jointype='left',
            )
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_btt():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .natural_join(
                b=descr(d=d),
                by=['x'],
                jointype='left',
            )
    )
    db_model = data_algebra.SQLite.SQLiteModel()
    sql = db_model.to_sql(
        ops,
        sql_format_options=data_algebra.db_model.SQLFormatOptions(use_with=True, annotate=False)
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_bttb():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .natural_join(
                b=descr(d2=d),
                by=['x'],
                jointype='left',
            )
    )
    db_model = data_algebra.SQLite.SQLiteModel()
    sql = db_model.to_sql(
        ops,
        sql_format_options=data_algebra.db_model.SQLFormatOptions(use_with=False, annotate=False)
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3],
    })
    data_algebra.test_util.check_transform(ops=ops, data={'d': d, 'd2': d}, expect=expect)


def test_dag_elim_uee():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x'})
            .concat_rows(
                b=(
                    descr(d=d)
                        .extend({'y': 'x'})
                ),
            )
    )
    db_model = data_algebra.SQLite.SQLiteModel()
    sql = db_model.to_sql(
        ops,
        sql_format_options=data_algebra.db_model.SQLFormatOptions(use_with=True, annotate=False)
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3, 1, 2, 3],
        'y': [1, 2, 3, 1, 2, 3],
        'source_name': ['a', 'a', 'a', 'b', 'b', 'b'],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_uet():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x'})
            .concat_rows(
                b=descr(d=d),
            )
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3, 1, 2, 3],
        'y': [1, 2, 3, 1, 2, 3],
        'source_name': ['a', 'a', 'a', 'b', 'b', 'b'],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_ute():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .concat_rows(
                b=(
                    descr(d=d)
                        .extend({'y': 'x'})
                ),
            )
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3, 1, 2, 3],
        'y': [1, 2, 3, 1, 2, 3],
        'source_name': ['a', 'a', 'a', 'b', 'b', 'b'],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_dag_elim_utt():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3],
    })
    ops = (
        descr(d=d)
            .concat_rows(
                b=descr(d=d),
            )
    )
    db_model = data_algebra.SQLite.SQLiteModel()
    sql = db_model.to_sql(
        ops,
        sql_format_options=data_algebra.db_model.SQLFormatOptions(use_with=True, annotate=False)
    )
    expect = pd.DataFrame({
        'x': [1, 2, 3, 1, 2, 3],
        'y': [1, 2, 3, 1, 2, 3],
        'source_name': ['a', 'a', 'a', 'b', 'b', 'b'],
    })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)

