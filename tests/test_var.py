
import numpy
import data_algebra
import data_algebra.test_util

have_polars = False
try:
    import polars as pl  # conditional import
    have_polars = True
except ModuleNotFoundError:
    pass


def test_var_extend_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "x": [2, 1, 2, 3, 4],
        "g": ["a", "b", "a", "c", "b"],
    })
    ops = (
        data_algebra.descr(d=d)
            .extend({
                "mean": "x.mean()",
                "std": "x.std()",
                "var": "x.var()",
                }, 
                partition_by=["g"])
    )
    res_pandas = ops.transform(d)
    expect =  pd.DataFrame({
        "x": [2, 1, 2, 3, 4],
        "g": ["a", "b", "a", "c", "b"],
        "mean": [2.0, 2.5, 2.0, 3.0, 2.5],
        "var": [0.0, 4.5, 0.0, numpy.nan, 4.5],  # sample variance
    })
    expect["std"] = numpy.sqrt(expect["var"])
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    if have_polars:
        res_polars = ops.transform(pl.DataFrame(d))
        expect_polars = (
            pl.DataFrame({
                "x": [2, 1, 2, 3, 4],
                "g": ["a", "b", "a", "c", "b"],
                "mean": [2.0, 2.5, 2.0, 3.0, 2.5],
                "var": [0.0, 4.5, 0.0, numpy.nan, 4.5],  # sample variance
                })
                .with_columns([pl.col("var").sqrt().alias("std")])
        )
        assert data_algebra.test_util.equivalent_frames(res_polars, expect_polars)
    data_algebra.test_util.check_transform(ops, data=d, expect=expect,
        try_on_Polars=False, # sample variance, except for size 1 groups, TODO: fix
        try_on_DBs=False,  # some dbs don't let std be used as a window fn
    )


def test_var_project_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "x": [2, 1, 2, 3, 4],
        "g": ["a", "b", "a", "c", "b"],
    })
    ops = (
        data_algebra.descr(d=d)
            .project({
                "mean": "x.mean()",
                "std": "x.std()",
                "var": "x.var()",
                }, 
                group_by=["g"])
    )
    res_pandas = ops.transform(d)
    expect =  pd.DataFrame({
        "g": ["a", "b", "c"],
        "mean": [2.0, 2.5, 3.0],
        "var": [0.0, 4.5, numpy.nan],  # sample variance
    })
    expect["std"] = numpy.sqrt(expect["var"])
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    if have_polars:
        res_polars = ops.transform(pl.DataFrame(d))
        expect_polars = (
            pl.DataFrame({
                "g": ["a", "b", "c"],
                "mean": [2.0, 2.5, 3.0],
                "var": [0.0, 4.5, numpy.nan],  # sample variance
                })
            .with_columns([pl.col("var").sqrt().alias("std")])
        )
        assert data_algebra.test_util.equivalent_frames(res_polars, expect_polars)
    data_algebra.test_util.check_transform(ops, data=d, expect=expect,
        try_on_Polars=False, # sample variance, except for size 1 groups, TODO: fix
    )