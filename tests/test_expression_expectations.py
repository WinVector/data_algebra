
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
    d = expectation_map['d']
    f_expectations = expectation_map['f_expectations']
    w_expectations = expectation_map['w_expectations']

    for exp, expect in f_expectations.items():
        res = f(exp)
        assert data_algebra.test_util.equivalent_frames(res, expect)

    for exp, expect in w_expectations.items():
        res = fg(exp)
        assert data_algebra.test_util.equivalent_frames(res, expect)

    # TODO: test on db
