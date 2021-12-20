
import os
import datetime
import pandas as pd
import pickle
import gzip

import data_algebra.test_util
from data_algebra.data_ops import *



def test_expression_expectations_1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb
    datetime_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y-%m-%d"

    d = pd.DataFrame({
        'row_id': [0, 1, 2, 3],
        'a': [False, False, True, True],
        'b': [False, True, False, True],
        'x': [.1, .2, .3, .4],
        'y': [2.4, 1.33, 1.2, 1.1],
        'z': [1.6, None, -2.1, 0],
        'g': ['a', 'a', 'b', 'ccc'],
        "str_datetime_col": ["2000-01-01 12:13:21", "2020-04-05 14:03:00", "2000-01-01 12:13:21",
                             "2020-04-05 14:03:00"],
        "str_date_col": ["2000-03-01", "2020-04-05", "2000-03-01", "2020-04-05"],
        "datetime_col_0": pd.to_datetime(
            pd.Series(["2010-01-01 12:13:21", "2030-04-05 14:03:00", "2010-01-01 12:13:21", "2030-04-05 14:03:00"]),
            format=datetime_format,
        ),
        "datetime_col_1": pd.to_datetime(
            pd.Series(["2010-01-01 12:11:21", "2030-04-06 14:03:00", "2010-01-01 12:11:21", "2030-04-06 14:03:00"]),
            format=date_format,
        ),
        "date_col_0": pd.to_datetime(
            pd.Series(["2000-01-02", "2035-04-05", "2000-01-02", "2035-04-05"]),
            format=date_format
        ).dt.date,
        "date_col_1": pd.to_datetime(
            pd.Series(["2000-01-02", "2035-05-05", "2000-01-02", "2035-05-05"]),
            format=date_format
        ).dt.date,
    })

    def f(expression):
        return ex(
            data(d=d)
                .extend({'new_column': expression})
                .select_columns(['row_id', 'new_column'])
                .order_rows(['row_id'])
        )

    def fg(expression):
        return ex(
            data(d=d)
                .extend(
                {'new_column': expression},
                partition_by=['g'],
                order_by=['row_id'])
                .select_columns(['g', 'row_id', 'new_column'])
                .order_rows(['g', 'row_id'])
        )

    expectation_path = os.path.join(
        dir_path, "expr_expectations.pkl.gz"
    )
    with gzip.open(expectation_path, 'rb') as in_f:
        expectation_map = pickle.load(in_f)
    f_expectations = expectation_map['f_expectations']
    w_expectations = expectation_map['w_expectations']

    for exp, expect in f_expectations.items():
        res = f(exp)
        assert data_algebra.test_util.equivalent_frames(res, expect)

    for exp, expect in w_expectations.items():
        res = fg(exp)
        assert data_algebra.test_util.equivalent_frames(res, expect)

    # TODO: test on db
