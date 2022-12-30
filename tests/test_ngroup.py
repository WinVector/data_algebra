
import numpy
import data_algebra
import data_algebra.test_util

have_polars = False
try:
    import polars as pl  # conditional import
    have_polars = True
except ModuleNotFoundError:
    pass


"""
# not implemented yet TODO
def test_ngroup_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "g": ["a", "a", "b", "ccc"],
    })
    ops = (
        data_algebra.descr(d=d)
            .extend({"new_column": "_ngroup()"}, partition_by=["g"])
    )
    res_pandas = ops.transform(d)
    expect =  pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "g": ["a", "a", "b", "ccc"],
        "new_column": [0, 0, 1, 2],
    })
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    if have_polars:
        res_polars = ops.transform(pl.DataFrame(d))
        assert data_algebra.test_util.equivalent_frames(res_polars, pl.DataFrame(expect))
    data_algebra.test_util.check_transform(ops, data=d, expect=expect)
"""


"""
# not sure we want to fix this yet TODO
def test_ngroup_cumcount_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "g": ["a", "a", "b", "ccc"],
        "z": [1.6, numpy.nan, -2.1, None],
    })
    ops = (
        data_algebra.descr(d=d)
            .extend({"new_column": "z.cumcount()"}, partition_by=["g"], order_by=["row_id"])
    )
    res_pandas = ops.transform(d)
    expect =  pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "g": ["a", "a", "b", "ccc"],
        "z": [1.6, numpy.nan, -2.1, None],
        "new_column": [0, 1, 0, 0],
    })
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    if have_polars:
        res_polars = ops.transform(pl.DataFrame(d))
        assert data_algebra.test_util.equivalent_frames(res_polars, pl.DataFrame(expect))
    data_algebra.test_util.check_transform(ops, data=d, expect=expect)
"""
