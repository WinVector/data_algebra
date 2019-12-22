
import sqlite3

import pandas

import pytest

from data_algebra.data_ops import *
import data_algebra.test_util
from data_algebra.arrow import fmt_as_arrow


def test_forbidden_calculation():
    td = TableDescription(table_name='d', column_names=['a', 'b', 'c'])

    # test using undefined column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'x'})

    # test colliding with know column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'b'})

    # test swaps don't show up in forbidden
    ops1 = td.rename_columns({'a': 'b', 'b': 'a'})
    f1 = ops1.forbidden_columns()
    assert set(f1['d']) == set()

    # test new column creation triggers forbidden annotation
    ops2 = td.rename_columns({'e': 'a'})
    f2 = ops2.forbidden_columns()
    assert set(['e']) == f2['d']

    # test merge
    ops3 = td.rename_columns({'e': 'a'}).rename_columns({'f': 'b'})
    f3 = ops3.forbidden_columns()
    assert set(['e', 'f']) == f3['d']

    # test composition
    ops4 = td.rename_columns({'e': 'a'}).rename_columns({'a': 'b'})
    f4 = ops4.forbidden_columns()
    assert set(['e']) == f4['d']


def test_calc_interface():
    td = TableDescription(table_name='d', column_names=['a'])
    ops = td.rename_columns({'b': 'a'})

    d_good = pandas.DataFrame({'a': [1]})
    d_extra = pandas.DataFrame({'a': [1], 'b': [2]})
    expect = pandas.DataFrame({'b': [1]})

    # check that table-defs narrow data
    res0 = td.transform(d_extra)
    assert data_algebra.test_util.equivalent_frames(res0, d_good)
    data_algebra.test_util.check_transform(ops=td, data=d_extra, expect=d_good)

    # check that def composition can widen arrows
    td_extra = describe_table(d_extra, 'd_extra')
    assert td.apply_to(td_extra) == td_extra
    assert td_extra.apply_to(td) == td
    assert (td >> td_extra) == td
    assert (td_extra >> td) == td_extra

    res1 = ops.transform(d_good)
    assert data_algebra.test_util.equivalent_frames(res1, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d_good, expect=expect)

    with pytest.raises(ValueError):
        ops.act_on(d_extra)

    ops.transform(d_extra)

    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()
    db_model.prepare_connection(conn)

    db_model.insert_table(conn, d_extra, table_name='d')

    sql = ops.to_sql(db_model, pretty=True)
    # Note: extra columns during execution is not an error.
    res_db_bad = db_model.read_query(conn, sql)
    data_model = {'d': [c for c in db_model.read_table(conn, 'd', limit=1).columns]}
    conn.close()

    assert data_algebra.test_util.equivalent_frames(res_db_bad, expect)

    with pytest.raises(ValueError):
        ops.check_constraints(data_model, strict=True)

    ops.check_constraints(data_model, strict=False)

    # the examples we are interested in

    ops_first = td.extend({'x': 1})
    res_first = ops_first.transform(d_extra)
    expect_first = data_algebra.pd.DataFrame({
        'a': [1],
        'x': [1],
        })
    assert data_algebra.test_util.equivalent_frames(res_first, expect_first)

    ops_widened = ops_first.apply_to(td_extra)
    assert ops_widened == (td_extra >> ops_first)
    res_widened = ops_widened.transform(d_extra)
    expect_widened = data_algebra.pd.DataFrame({
        'a': [1],
        'b': [2],
        'x': [1],
        })
    assert data_algebra.test_util.equivalent_frames(res_widened, expect_widened)

    ops_strict = td.rename_columns({'b': 'a'})

    with pytest.raises(ValueError):
        ops_strict.apply_to(td_extra)

    new_ops = ops_strict.apply_to(td.drop_columns('b'))
    fmt_as_arrow(ops_strict)
    fmt_as_arrow(new_ops)

    assert data_algebra.test_util.equivalent_frames(new_ops.transform(d_good), expect)
    assert data_algebra.test_util.equivalent_frames(new_ops.transform(d_extra), expect)
    data_algebra.test_util.check_transform(ops=new_ops, data=d_good, expect=expect)
    data_algebra.test_util.check_transform(ops=new_ops, data=d_extra, expect=expect)