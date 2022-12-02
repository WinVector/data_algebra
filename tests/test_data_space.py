
"""test data model isolation"""

import data_algebra
import data_algebra.test_util
import data_algebra.pandas_space


def test_pandas_data_space_1():
    dm = data_algebra.pandas_model.default_data_model
    pd = dm.pd
    ps = data_algebra.pandas_space.PandasSpace(dm)
    t0 = dm.data_frame({'x': [1, 2, 3]})
    ps.insert(key="t0", value=t0)
    ops = (
        data_algebra.descr(t0=t0)
            .extend({'y': 'x + 1'})
    )
    res_descr = ps.execute(ops)
    res = ps.retrieve(res_descr.table_name)
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
