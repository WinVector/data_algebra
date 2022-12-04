
import data_algebra

have_polars = False
try:
    import polars as pl
    import data_algebra.polars_model
    have_polars = True
except ModuleNotFoundError:
    pass


def test_polars_1():
    if have_polars:
        d = pl.DataFrame({"x": range(100),  "y": range(100)})
        ops = (
            data_algebra.descr(d=d)
                .select_columns(["x"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": range(100)})
        assert res.frame_equal(expect)
