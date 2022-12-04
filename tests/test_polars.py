
import numpy as np
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


def test_polars_2():
    if have_polars:
        # from Examples/DupRows.ipynb

        def generate_example(*, n_columns: int = 5, n_rows: int = 10):
            assert isinstance(n_columns, int)
            assert isinstance(n_rows, int)
            return pl.DataFrame({
                f"col_{i:03d}": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True) for i in range(n_columns)
            })
        
        rng = np.random.default_rng(2022)
        d = generate_example(n_columns=2, n_rows=100)
        ops = (
            data_algebra.descr(d=d)
                .extend({"count": "(1).sum()"}, partition_by=d.columns)
                .select_rows("count > 1")
                .drop_columns(["count"])
        )
        # res = ops.transform(d)
        # TODO: take test further
