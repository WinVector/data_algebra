
import numpy

import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.SQLite


def test_if_else_return_type():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [True, False, None],
    })
    ops = (
        descr(d=d)
            .extend({
                'w': 'x.where(1.0, 2.0)',
                'i': 'x.if_else(1.0, 2.0)',
            })
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'x': [True, False, None],
        'w': [1.0, 2.0, 2.0],
        'i': [1.0, 2.0, numpy.nan],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    assert str(res['w'].dtype) == 'float64'
    assert str(res['i'].dtype) == 'float64'
    numpy.isnan(res.loc[:, ['w', 'i']])  # when column types are wrong this threw in pyvteat test_KDD2009.py
    sqlite_handle = data_algebra.SQLite.example_handle()
    sqlite_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_sqlite = sqlite_handle.read_query(ops)
    sqlite_handle.close()
    assert data_algebra.test_util.equivalent_frames(res_sqlite, expect)
    assert str(res_sqlite['w'].dtype) == 'float64'
    assert str(res_sqlite['i'].dtype) == 'float64'
    numpy.isnan(res_sqlite.loc[:, ['w', 'i']])  # when column types are wrong this threw in pyvteat test_KDD2009.py
