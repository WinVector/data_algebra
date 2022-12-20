
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
    known_not_to_work = set([
        'arctan2', 
        'base_Sunday',
        'bfill',
        'concat',
        'count',
        'date_diff',
        'datetime_to_date',
        'dayofmonth', 
        'dayofweek',
        'dayofyear',
        'expm1',
        'ffill',
        'format_date',
        'format_datetime',
        'is_in',
        'log1p',
        'month',
        'mapv',
        'nunique',
        'parse_date',
        'quarter',
        'shift',
        'size',   # TODO: fix this, failing to load constant def bug
        'std',
        'timestamp_diff',
        'trimstr',
        'var',
        'weekofyear',
        'year',
        '_ngroup',
        ])
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
            if not data_algebra.data_model.default_data_model().is_appropriate_data_instance(res):
                raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
            if not  data_algebra.test_util.equivalent_frames(res, expect):
                raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
            if op not in known_not_to_work:
                # test Polars
                if have_polars:
                    try:
                        res_polars = ops.transform(d_polars)
                    except Exception as ex:
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
                    if not isinstance(res_polars, pl.DataFrame):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
                    res_polars_pandas = res_polars.to_pandas()
                    if not data_algebra.data_model.default_data_model().is_appropriate_data_instance(res_polars_pandas):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
                    if not data_algebra.test_util.equivalent_frames(res_polars_pandas, expect):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")

