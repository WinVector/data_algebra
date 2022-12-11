
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


def test_polars_logistic():
    if have_polars:
        # from Examples/Polars/PolarsLogisticExample.ipynb
        d_polars = pl.DataFrame({
            'subjectID':[1, 1, 2, 2],
            'surveyCategory': [ "withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
            'assessmentTotal': [5., 2., 3., 4.],
            'irrelevantCol1': ['irrel1']*4,
            'irrelevantCol2': ['irrel2']*4,
        })
        scale = 0.237
        ops = (
            data_algebra.descr(d=d_polars)
                .extend({'probability': f'(assessmentTotal * {scale}).exp()'})
                .extend({'total': 'probability.sum()'},
                        partition_by='subjectID')
                .extend({'probability': 'probability / total'})
                .extend({'row_number': '(1).cumsum()'},
                        partition_by=['subjectID'],
                        order_by=['probability'], 
                        reverse=['probability'])
                .select_rows('row_number == 1')
                .select_columns(['subjectID', 'surveyCategory', 'probability'])
                .map_columns({'surveyCategory': 'diagnosis'})
                .order_rows(["subjectID"])
            )
        res_polars = ops.transform(d_polars)
        expect = pl.DataFrame(
            {
                "subjectID": [1, 2],
                "diagnosis": ["withdrawal behavior", "positive re-framing"],
                "probability": [0.670622, 0.558974],
            }
        )
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas(), float_tol=1.0e-3)


def test_polars_project():
    if have_polars:
        d_polars = pl.DataFrame({
            'subjectID':[1, 1, 2, 2],
            'assessmentTotal': [5., 2., 3., -4.],
        })
        ops = (
            data_algebra.descr(d=d_polars)
                .project({"tot": "assessmentTotal.sum()"}, group_by=["subjectID"])
        )
        res_polars = ops.transform(d_polars)
        expect = pl.DataFrame(
            {
                "subjectID": [1, 2],
                "tot": [7., -1.],
            }
        )
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_join():
    if have_polars:
        d_a = pl.DataFrame({
            'a': [1, 2],
            'q': [3, 5],
        })
        d_b = pl.DataFrame({
            'a': [1, 2],
            'z': ['a', 'b'],
        })
        ops = (
            data_algebra.descr(d_a=d_a)
                .natural_join(
                    data_algebra.descr(d_b=d_b),
                    on=["a"],
                    jointype='left'
                )
        )
        res_polars = ops.eval({"d_a": d_a, "d_b": d_b})
        expect = pl.DataFrame({
            'a': [1, 2],
            'q': [3, 5],
            'z': ['a', 'b'],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_concat():
    if have_polars:
        d_a = pl.DataFrame({
            'a': [1, 2],
            'q': [3, 5],
        })
        d_b = pl.DataFrame({
            'a': [1, 3],
            'q': [7, 8],
        })
        ops = (
            data_algebra.descr(d_a=d_a)
                .concat_rows(data_algebra.descr(d_b=d_b))
        )
        res_polars = ops.eval({"d_a": d_a, "d_b": d_b})
        expect = pl.DataFrame({
            'a': [1, 2, 1, 3],
            'q': [3, 5, 7, 8],
            'source_name': ["a", "a", "b", "b"],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_project_max():
    if have_polars:
        d = pl.DataFrame({
            "g": ["a", "a", "b"],
            "v": ["x", "y", "x"],
        })
        ops = (
            data_algebra.descr(d=d)
                            .project(
                                {
                                    "min_v": "v.min()",
                                    "max_v": "v.max()",
                                },
                                group_by=["g"]
                                )
        )
        res_polars = ops.transform(d)
        expect = pl.DataFrame({
            "g": ["a", "b"],
            "min_v": ["x", "x"],
            "max_v": ["y", "x"],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())
