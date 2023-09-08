
import datetime
import numpy as np
import data_algebra
import data_algebra.data_model
import data_algebra.test_util
import data_algebra.cdata
import pytest

have_polars = False
try:
    import polars as pl  # conditional import
    have_polars = True
except ModuleNotFoundError:
    pass


def test_polars_const_bool():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2]})
        ops = (
            data_algebra.descr(d=d)
                .extend({"y": "False", "z": "True"})
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": [1, 2], "y": [False, False], "z": [True, True]})
        assert data_algebra.test_util.equivalent_frames(res, expect)


def test_polars_extend_cumsum():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3]})
        ops = (
            data_algebra.descr(d=d)
                .extend({"n": "(1).cumsum()"}, partition_by=["x"], order_by=["y"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3], "n": [1, 1, 2]})
        assert data_algebra.test_util.equivalent_frames(res, expect)


def test_polars_extend_count():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3]})
        ops = (
            data_algebra.descr(d=d)
                .extend({"n": "_count()"}, partition_by=["x"], order_by=["y"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3], "n": [1, 1, 2]})
        assert data_algebra.test_util.equivalent_frames(res, expect)


def test_polars_project_count():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3]})
        ops = (
            data_algebra.descr(d=d)
                .project({"n": "_count()"}, group_by=["x"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": [1, 2], "n": [2, 1]})
        assert data_algebra.test_util.equivalent_frames(res, expect)


def test_polars_project_sum():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2, 1], "y": [1, 2, 3]})
        ops = (
            data_algebra.descr(d=d)
                .project({"n": "(1).sum()"}, group_by=["x"])
        )
        res = ops.transform(d)
        expect = pl.DataFrame({"x": [1, 2], "n": [2, 1]})
        assert data_algebra.test_util.equivalent_frames(res, expect)


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


def test_polars_project_max_int():
    if have_polars:
        d = pl.DataFrame({
            "g": ["a", "a", "b"],
            "v": [1, 2, 3],
        })
        # d.group_by(["g"]).agg([pl.col("v").min().alias("v_min"), pl.col("v").max().alias("v_max")])
        # returns correct answer
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
            "min_v": [1, 3],
            "max_v": [2, 3],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_project_max_str():
    if have_polars:
        d = pl.DataFrame({
            "g": ["a", "a", "b"],
            "v": ["x", "y", "x"],
        })
        # d.group_by(["g"]).agg([pl.col("v").min().alias("v_min"), pl.col("v").max().alias("v_max")])
        # returns nulls
        # known Polars bug:
        # https://stackoverflow.com/q/74763636/6901725
        # https://github.com/pola-rs/polars/issues/5735
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


def test_polars_project_max_date():
    if have_polars:
        d = pl.DataFrame({
            "g": ["a", "a", "b"],
            "v": ['2020-01-01', "2023-01-01", "2020-01-01"],
        })
        # From: https://stackoverflow.com/a/71759536/6901725
        # works:
        # d.with_columns([pl.col("v").cast(str).str.strptime(pl.Date, fmt="%Y-%m-%d", strict=False).cast(pl.Date).alias("v2")])
        ops = (
            data_algebra.descr(d=d)
                .extend({"v": "v.parse_date('%Y-%m-%d')"})
                .project(
                    {
                        "min_v": "v.min()",
                        "max_v": "v.max()",
                    },
                    group_by=["g"]
                    )
                .order_rows(["g"])
        )
        res_polars = ops.transform(d)
        expect = pl.DataFrame({
            "g": ["a", "b"],
            "min_v": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)],
            "max_v": [datetime.date(2023, 1, 1), datetime.date(2020, 1, 1)],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_project_max_date_2():
    if have_polars:
        d = pl.DataFrame({
            "g": ["a", "a", "b"],
            "v": ['2020-01-01', "2023-01-01", "2020-01-01"],
        })
        # From: https://stackoverflow.com/a/71759536/6901725
        # works:
        # d.with_columns([pl.col("v").cast(str).str.strptime(pl.Date, fmt="%Y-%m-%d", strict=False).cast(pl.Date).alias("v2")])
        ops = (
            data_algebra.descr(d=d)
                .extend({"v": "v.parse_date('%Y-%m-%d')"})
                .project(
                    {
                        "min_v": "v.min()",
                        "max_v": "v.max()",
                    },
                    group_by=["g"]
                    )
                .order_rows(["g"])
        )
        res_polars = ops.transform(d)
        expect = pl.DataFrame({
            "g": ["a", "b"],
            "min_v": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)],
            "max_v": [datetime.date(2023, 1, 1), datetime.date(2020, 1, 1)],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas())


def test_polars_group_min_max_example():
    # from Examples/TimedGroupedCalc.ipynb
    if have_polars:
        pd = data_algebra.data_model.default_data_model().pd
        rng = np.random.default_rng(2022)

        def mk_example(*, n_rows: int, n_groups: int):
            assert n_rows > 0
            assert n_groups > 0
            groups = [f"group_{i:04d}" for i in range(n_groups)]
            d = pd.DataFrame({
                "group": rng.choice(groups, size=n_rows, replace=True),
                "value": rng.normal(size=n_rows)
            })
            return d
        
        d_Pandas = mk_example(n_rows=10, n_groups=2)
        d_Polars = pl.DataFrame(d_Pandas)
        res_pandas = (
            d_Pandas
                .groupby(["group"])
                .agg({"value": ["min", "max"]})
            )
        res_polars = (
            d_Polars
                .group_by(["group"])
                .agg([
                    pl.col("value").min().alias("min_value"),
                    pl.col("value").max().alias("max_value"),
                ])
            )
        expect = pl.DataFrame({
            "group": ["group_0000", "group_0001"],
            "min_value": [-2.931249, -1.440234],
            "max_value": [1.667716, 0.078888],
        })
        assert data_algebra.test_util.equivalent_frames(res_polars.to_pandas(), expect.to_pandas(), float_tol=1.0e-3)


def test_polars_cdata_example():
    if have_polars:
        pd = data_algebra.data_model.default_data_model().pd
        c1 = pd.DataFrame({
            "k1": [1, 2, 3],
            "v1": ["a", "c", "e"],
            "v2": ["b", "d", "f"],
        })
        c2 = pd.DataFrame({
            "k2": [4, 5],
            "w1": ["a", "b"],
            "w2": ["c", "d"],
            "w3": ["e", "f"],
        })
        record_map_pandas = data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    c1,
                    control_table_keys=["k1"],
                    record_keys=["id"],
                ),
                blocks_out=data_algebra.cdata.RecordSpecification(
                    c2,
                    control_table_keys=["k2"],
                    record_keys=["id"],
                ),
        )
        d = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 2],
            "k1": [1, 2, 3, 1, 2, 3],
            "v1": ["a", "c", "e", "g", "i", "k"],
            "v2": ["b", "d", "f", "h", "j", "l"],
        })
        expect =  pd.DataFrame({
            "id": [1, 1, 2, 2],
            "k2": [4, 5, 4, 5],
            "w1": ["a", "b", "g", "h"],
            "w2": ["c", "d", "i", "j"],
            "w3": ["e", "f", "k", "l"],
        })
        conv_pandas_rm = record_map_pandas.transform(d)
        assert data_algebra.test_util.equivalent_frames(conv_pandas_rm, expect)
        ops = (
            data_algebra.descr(d=d)
                .convert_records(record_map=record_map_pandas)
        )
        conv_pandas_ops = ops.transform(d)
        assert data_algebra.test_util.equivalent_frames(conv_pandas_ops, expect)
        conv_polars_ops = ops.transform(pl.DataFrame(d))
        assert isinstance(conv_polars_ops, pl.DataFrame)
        assert data_algebra.test_util.equivalent_frames(conv_polars_ops.to_pandas(), expect)
        # again with pure Polars structures
        record_map_polars = data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    pl.DataFrame(c1),
                    control_table_keys=["k1"],
                    record_keys=["id"],
                ),
                blocks_out=data_algebra.cdata.RecordSpecification(
                    pl.DataFrame(c2),
                    control_table_keys=["k2"],
                    record_keys=["id"],
                ),
        )
        conv_pure_polars_rm = record_map_polars.transform(pl.DataFrame(d))
        assert isinstance(conv_pure_polars_rm, pl.DataFrame)
        assert data_algebra.test_util.equivalent_frames(conv_pure_polars_rm.to_pandas(), expect)
        ops_polars = (
            data_algebra.descr(d=pl.DataFrame(d))
                .convert_records(record_map=record_map_polars)
        )
        conv_pure_polars_ops = ops_polars.transform(pl.DataFrame(d))
        assert isinstance(conv_pure_polars_ops, pl.DataFrame)
        assert data_algebra.test_util.equivalent_frames(conv_pure_polars_ops.to_pandas(), expect)


def test_polars_cdata_example_exbb():
    if have_polars:
        c1 = pl.DataFrame({
            "k1": [1, 2, 3],
            "v1": ["a", "c", "e"],
            "v2": ["b", "d", "f"],
        })
        c2 = pl.DataFrame({
            "k2": [4, 5],
            "w1": ["a", "b"],
            "w2": ["c", "d"],
            "w3": ["e", "f"],
        })
        rm = data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    c1,
                    control_table_keys=["k1"],
                    record_keys=["id"],
                ),
                blocks_out=data_algebra.cdata.RecordSpecification(
                    c2,
                    control_table_keys=["k2"],
                    record_keys=["id"],
                ),
        )
        rm_str = str(rm)
        assert isinstance(rm_str, str)
        rm_repr = rm.__repr__()
        assert isinstance(rm_repr, str)
        inp1 = rm.example_input()
        assert isinstance(inp1, pl.DataFrame)
        expect_inp1 = pl.DataFrame({
            "id": ["id record key", "id record key", "id record key"],
            "k1": [1, 2, 3],
            "v1": ["a value", "c value", "e value"],
            "v2": ["b value", "d value", "f value"],
        })
        assert data_algebra.test_util.equivalent_frames(inp1.to_pandas(), expect_inp1.to_pandas())
        out1 = rm.transform(inp1)
        assert isinstance(out1, pl.DataFrame)
        expect_out1 = pl.DataFrame({
            "id": ["id record key", "id record key"],
            "k2": [4, 5],
            "w1": ["a value", "b value"],
            "w2": ["c value", "d value"],
            "w3": ["e value", "f value"],
        })
        assert data_algebra.test_util.equivalent_frames(out1.to_pandas(), expect_out1.to_pandas())
        back_1 = rm.inverse().transform(out1)
        assert data_algebra.test_util.equivalent_frames(back_1.to_pandas(), expect_inp1.to_pandas())


def test_polars_cdata_example_exbr():
    if have_polars:
        c1 = pl.DataFrame({
            "k1": [1, 2, 3],
            "v1": ["a", "c", "e"],
            "v2": ["b", "d", "f"],
        })
        rm = data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    c1,
                    control_table_keys=["k1"],
                    record_keys=["id"],
                ),
        )
        rm_str = str(rm)
        assert isinstance(rm_str, str)
        rm_repr = rm.__repr__()
        assert isinstance(rm_repr, str)
        inp1 = rm.example_input()
        assert isinstance(inp1, pl.DataFrame)
        expect_inp1 = pl.DataFrame({
            "id": ["id record key", "id record key", "id record key"],
            "k1": [1, 2, 3],
            "v1": ["a value", "c value", "e value"],
            "v2": ["b value", "d value", "f value"],
        })
        assert data_algebra.test_util.equivalent_frames(inp1.to_pandas(), expect_inp1.to_pandas())
        out1 = rm.transform(inp1)
        assert isinstance(out1, pl.DataFrame)
        expect_out1 = pl.DataFrame({
            "id": ["id record key"],
            "a": ["a value"],
            "b": ["b value"],
            "c": ["c value"],
            "d": ["d value"],
            "e": ["e value"],
            "f": ["f value"],
        })
        assert data_algebra.test_util.equivalent_frames(out1.to_pandas(), expect_out1.to_pandas())
        back_1 = rm.inverse().transform(out1)
        assert data_algebra.test_util.equivalent_frames(back_1.to_pandas(), expect_inp1.to_pandas())


def test_polars_cdata_example_exrb():
    if have_polars:
        c2 = pl.DataFrame({
            "k2": [4, 5],
            "w1": ["a", "b"],
            "w2": ["c", "d"],
            "w3": ["e", "f"],
        })
        rm = data_algebra.cdata.RecordMap(
                blocks_out=data_algebra.cdata.RecordSpecification(
                    c2,
                    control_table_keys=["k2"],
                    record_keys=["id"],
                ),
        )
        rm_str = str(rm)
        assert isinstance(rm_str, str)
        rm_repr = rm.__repr__()
        assert isinstance(rm_repr, str)
        inp1 = rm.example_input()
        assert isinstance(inp1, pl.DataFrame)
        expect_inp1 = pl.DataFrame({
            "id": ["id record key"],
            "a": ["a value"],
            "b": ["b value"],
            "c": ["c value"],
            "d": ["d value"],
            "e": ["e value"],
            "f": ["f value"],
        })
        assert data_algebra.test_util.equivalent_frames(inp1.to_pandas(), expect_inp1.to_pandas())
        out1 = rm.transform(inp1)
        assert isinstance(out1, pl.DataFrame)
        expect_out1 = pl.DataFrame({
            "id": ["id record key", "id record key"],
            "k2": [4, 5],
            "w1": ["a value", "b value"],
            "w2": ["c value", "d value"],
            "w3": ["e value", "f value"],
        })
        assert data_algebra.test_util.equivalent_frames(out1.to_pandas(), expect_out1.to_pandas())
        back_1 = rm.inverse().transform(out1)
        assert data_algebra.test_util.equivalent_frames(back_1.to_pandas(), expect_inp1.to_pandas())


def test_polars_table_is_keyed_by_columns():
    if have_polars:
        d = pl.DataFrame(
            {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]}
        )
        local_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
        assert local_model.table_is_keyed_by_columns(d, column_names=["a", "b"])
        assert not local_model.table_is_keyed_by_columns(d, column_names=["a"])


def test_is_inf_polars():
    if have_polars:
        d = pl.DataFrame({
            'a': [1.0, np.inf, np.nan, None, 0.0, -1.0, -np.inf],
        })
        ops = (
            data_algebra.descr(d=d)
                .extend({
                    'is_inf': 'a.is_inf().where(1, 0)',
                    'is_nan': 'a.is_nan().where(1, 0)',
                    'is_bad': 'a.is_bad().where(1, 0)',
                    'is_null': 'a.is_null().where(1, 0)',
                    })
        )
        res_polars = ops.transform(d)
        expect = pl.DataFrame({
            'a': [1.0, np.inf, np.nan, None, 0.0, -1.0, -np.inf],
            'is_inf': [0, 1, 0, 0, 0, 0, 1],
            'is_nan': [0, 0, 1, 0, 0, 0, 0],
            'is_bad': [0, 1, 1, 1, 0, 0, 1],
            'is_null': [0, 0, 0, 1, 0, 0, 0],   # Pandas can't tell the difference, Polars can
            })
        assert data_algebra.test_util.equivalent_frames(expect, res_polars)


def test_polars_arrow_table_narrows():
    if have_polars:
        d = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        ops = data_algebra.TableDescription(column_names=["x"])
        res = ops.transform(d)
        assert isinstance(res, pl.DataFrame)
        expect = pl.DataFrame({"x": [1, 2]})
        assert data_algebra.test_util.equivalent_frames(res, expect)
        with pytest.raises(AssertionError):
            d >> ops
        res2 = d[["x"]] >> ops
        assert isinstance(res2, pl.DataFrame)
        assert data_algebra.test_util.equivalent_frames(res2, expect)


def test_polars_minimum_1_issue():
    if have_polars:
        d = pl.DataFrame({
            "x": [1, 2, 3, 4, 5, 6], 
            "g": [1, 1, 1, 2, 2, 2],
            "x_g_max": [2, 2, 2, 5, 5, 5],
        })
        ops = (
            data_algebra.descr(d=d)
                .extend({"xl": "x.minimum(x_g_max)"})
        )
        expect =pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "g": [1, 1, 1, 2, 2, 2],
                "x_g_max": [2, 2, 2, 5, 5, 5],
                "xl": [1, 2, 2, 4, 5, 5],
            }
        )
        # was raising "ValueError: could not convert value 'Unknown' as a Literal"
        res = ops.transform(d)
        assert isinstance(res, pl.DataFrame)
        assert np.max(np.abs(np.array(res['xl'] - expect['xl']))) < 1e-8
