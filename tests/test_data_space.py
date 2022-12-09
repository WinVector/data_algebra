
"""test data model isolation"""

import data_algebra
import data_algebra.test_util
import data_algebra.data_model_space
import data_algebra.db_space


def test_pandas_data_space_1():
    dm = data_algebra.data_model.default_data_model()
    pd = dm.pd
    t0 = dm.data_frame({'x': [1, 2, 3]})
    with data_algebra.data_model_space.DataModelSpace(dm) as ds:
        ds.insert(key="t0", value=t0)
        ops = (
            data_algebra.descr(t0=t0)
                .extend({'y': 'x + 1'})
        )
        res_descr = ds.execute(ops)
        res = ds.retrieve(res_descr.table_name)
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_db_data_space_1():
    pd = data_algebra.data_model.default_data_model().pd
    t0 = pd.DataFrame({'x': [1, 2, 3]})
    with data_algebra.db_space.DBSpace() as ds:
        ds.insert(key="t0", value=t0)
        ops = (
            data_algebra.descr(t0=t0)
                .extend({'y': 'x + 1'})
        )
        res_descr = ds.execute(ops)
        res = ds.retrieve(res_descr.table_name)
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
