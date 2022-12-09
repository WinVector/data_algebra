
import numpy as np
import data_algebra
import data_algebra.data_model
import data_algebra.test_util

have_polars = False
try:
    import polars as pl
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
        assert data_algebra.test_util.equivalent_frames(res.to_pandas(), expect.to_pandas())


def test_polars_1b():
    if have_polars:
        d = pl.DataFrame({"x": range(100),  "y": range(100)})
        ops = (
            data_algebra.descr(d=d)
                .drop_columns(["y"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": range(100)})
        assert data_algebra.test_util.equivalent_frames(res.to_pandas(), expect.to_pandas())


def test_polars_1c():
    if have_polars:
        d = pl.DataFrame({"x": range(100),  "y": range(100)})
        ops = (
            data_algebra.descr(d=d)
                .select_rows("x < 50")
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": range(50),  "y": range(50)})
        assert data_algebra.test_util.equivalent_frames(res.to_pandas(), expect.to_pandas())
    

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
        d = generate_example(n_columns=3, n_rows=20)
        ops = (
            data_algebra.descr(d=d)
                .extend({"count": 1})
                .extend({"count": "count.sum()"}, partition_by=d.columns)
                .select_rows("count > 1")
                .drop_columns(["count"])
        )
        res_polars = ops.transform(d)
        res_pandas = ops.transform(d.to_pandas())
        assert isinstance(res_polars, pl.DataFrame)
        assert isinstance(res_pandas, data_algebra.data_model.default_data_model().pd.DataFrame)
        assert res_polars.shape == res_pandas.shape
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), res_pandas)


def test_polars_2c():
    if have_polars:
        # from Examples/DupRows.ipynb

        def generate_example(*, n_columns: int = 5, n_rows: int = 10):
            assert isinstance(n_columns, int)
            assert isinstance(n_rows, int)
            return pl.DataFrame({
                f"col_{i:03d}": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True) for i in range(n_columns)
            })
        
        rng = np.random.default_rng(2022)
        d = generate_example(n_columns=3, n_rows=20)
        ops = (
            data_algebra.descr(d=d)
                .extend({"count": "(1).sum()"}, partition_by=d.columns)
                .select_rows("count > 1")
        )
        res_polars = ops.transform(d)
        res_pandas = ops.transform(d.to_pandas())
        assert isinstance(res_polars, pl.DataFrame)
        assert isinstance(res_pandas, data_algebra.data_model.default_data_model().pd.DataFrame)
        assert res_polars.shape == res_pandas.shape
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), res_pandas)


def test_polars_2e():
    if have_polars:
        # from Examples/DupRows.ipynb

        def generate_example(*, n_columns: int = 5, n_rows: int = 10):
            assert isinstance(n_columns, int)
            assert isinstance(n_rows, int)
            return pl.DataFrame({
                f"col_{i:03d}": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True) for i in range(n_columns)
            })
        
        rng = np.random.default_rng(2022)
        d = generate_example(n_columns=3, n_rows=20)
        ops = (
            data_algebra.descr(d=d)
                .extend({"count": 1})
                .extend({"count": "count.sum()"}, partition_by=[])
        )
        res_polars = ops.transform(d)
        res_pandas = ops.transform(d.to_pandas())
        assert isinstance(res_polars, pl.DataFrame)
        assert isinstance(res_pandas, data_algebra.data_model.default_data_model().pd.DataFrame)
        assert res_polars.shape == res_pandas.shape
        assert np.max(res_pandas["count"]) == np.min(res_pandas["count"])
        assert np.max(res_pandas["count"]) == res_pandas.shape[0]
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), res_pandas)


def test_polars_group_by():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2, 3],  "y": ['a', 'a', 'b']})
        ops = (
            data_algebra.descr(d=d)
                .project({"x": "x.mean()"}, group_by=["y"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"y": ["a", "b"], "x": [1.5, 3.0]})
        assert data_algebra.test_util.equivalent_frames(res.to_pandas(), expect.to_pandas())
