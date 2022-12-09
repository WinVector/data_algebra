
import os
import pickle
import gzip

import data_algebra
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util

have_polars = False
try:
    import polars as pl
    have_polars = True
except ModuleNotFoundError:
    pass


def test_expression_expectations_polars_1():
    known_not_to_work = [  # TODO: fix all of these
        '_count', '_ngroup', '_row_number', '_size', 'arctan2', 'around',
        'as_int64', 'as_str', 'base_Sunday', 'bfill', 'coalesce', 'concat',
        'count', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum',
        'date_diff', 'datetime_to_date', 'dayofmonth', 'dayofweek',
        'dayofyear', 'expm1', 'ffill', 'first', 'fmax', 'fmin',
        'format_date', 'format_datetime', 'if_else', 'is_bad', 'is_in',
        'is_inf', 'last', 'log1p', 'mapv', 'maximum', 'minimum', 'month',
        'nunique', 'parse_date', 'parse_datetime', 'quarter', 'rank',
        'remainder', 'round', 'shift', 'size', 'std', 'timestamp_diff',
        'trimstr', 'var', 'weekofyear', 'where', 'year']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb 
    expectation_path = os.path.join(dir_path, "expr_expectations.pkl.gz")
    with gzip.open(expectation_path, 'rb') as in_f:
        expectation_map = pickle.load(in_f)
    d = expectation_map['d']
    d_polars = None
    if have_polars:
        d_polars = pl.DataFrame(d)
    expectations = expectation_map['expectations']
    for op, op_class, exp, ops, expect in expectations:
        if data_algebra.data_model.default_data_model().is_appropriate_data_instance(expect):
            # test Pandas
            res = ops.transform(d)
            assert data_algebra.data_model.default_data_model().is_appropriate_data_instance(res)
            assert data_algebra.test_util.equivalent_frames(res, expect)
            if op not in known_not_to_work:
                # test Polars
                if have_polars:
                    res_polars = ops.transform(d_polars)
                    assert isinstance(res_polars, pl.DataFrame)
                    res_polars_pandas = res_polars.to_pandas()
                    assert data_algebra.data_model.default_data_model().is_appropriate_data_instance(res_polars_pandas)
                    assert data_algebra.test_util.equivalent_frames(res_polars_pandas, expect)

