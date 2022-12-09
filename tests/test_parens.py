
import re
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.util
from data_algebra.sql_format_options import SQLFormatOptions
import data_algebra.SQLite


def test_parens_select_rows():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [1, 2, 3, 4]
    })
    ops = (
        descr(d=d)
            .select_rows('x > 1 and x < 4')
    )
    sql = data_algebra.SQLite.SQLiteModel().to_sql(ops)
    smushed_sql = re.sub(r'\s+', '', sql)
    assert '("x">1)AND("x"<4)' in smushed_sql
    expect = pd.DataFrame({
        'x': [2, 3],
        })
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)


def test_parens_extend():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [1, 2, 3, 4]
    })
    ops = (
        descr(d=d)
            .extend({'y': 'x > 1 and x < 4'})
    )
    sql = data_algebra.SQLite.SQLiteModel().to_sql(
        ops,
        sql_format_options=SQLFormatOptions(annotate=False))
    smushed_sql = re.sub(r'\s+', '', sql)
    assert '("x">1)AND("x"<4)' in smushed_sql
    expect = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [False, True, True, False],
        })
    data_algebra.test_util.check_transform(ops, data={"d": d}, expect=expect)
