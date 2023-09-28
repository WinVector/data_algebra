
import os
import pickle
import gzip

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite
import data_algebra.test_util

have_polars = False
try:
    import polars as pl  # conditional import
    have_polars = True
except ModuleNotFoundError:
    pass


def test_expression_expectations_polars_1():
    if have_polars:
        known_not_to_work = set([
            'arctan2',
            'expm1',
            'log1p',
            'count',  # Pandas null treatment in the wrong
            'cumcount',  # pending implementation
            'base_Sunday',
            'date_diff',
            'datetime_to_date',
            'dayofmonth',
            'dayofweek',
            'dayofyear',
            'format_date',
            'format_datetime',
            'month',
            'parse_date',
            'quarter',
            'weekofyear',
            'year',
            'timestamp_diff',
            'trimstr',
            'std',
            'var',
            '_ngroup',  # pending group numbering implementation
            ])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb 
        expectation_path = os.path.join(dir_path, "expr_expectations.pkl.gz")
        with gzip.open(expectation_path, 'rb') as in_f:
            expectation_map = pickle.load(in_f)
        d = expectation_map['d']
        d_polars = pl.DataFrame(d)
        expectations = expectation_map['expectations']
        for op, op_class, exp, ops, expect in expectations:
            if data_algebra.data_model.default_data_model().is_appropriate_data_instance(expect):
                if op not in known_not_to_work:
                    # test Polars
                    res_polars = ops.transform(d_polars)
                    if not isinstance(res_polars, pl.DataFrame):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
                    res_polars_pandas = res_polars.to_pandas()
                    if not data_algebra.data_model.default_data_model().is_appropriate_data_instance(res_polars_pandas):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
                    if not data_algebra.test_util.equivalent_frames(res_polars_pandas, expect):
                        raise ValueError(f"failure on {op} {op_class} {exp} {ops}")
