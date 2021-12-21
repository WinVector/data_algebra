
import os
import datetime
import pandas as pd
import pickle
import gzip

from data_algebra.data_ops import *
import data_algebra.BigQuery
import data_algebra.SQLite
import data_algebra.test_util


def test_expression_expectations_1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb
    datetime_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y-%m-%d"

    def f(expression):
        return (
            descr(d=d)
                .extend({'new_column': expression})
                .select_columns(['row_id', 'new_column'])
                .order_rows(['row_id'])
        )

    def fg(expression):
        return (
            descr(d=d)
                .extend(
                    {'new_column': expression},
                    partition_by=['g'])
                .select_columns(['g', 'row_id', 'new_column'])
                .order_rows(['g', 'row_id'])
        )

    def fw(expression):
        return (
            descr(d=d)
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
    g_expectations = expectation_map['g_expectations']
    w_expectations = expectation_map['w_expectations']

    ops_list = f_expectations + g_expectations + w_expectations
    for op, op_class, exp, ops, expect in ops_list:
        res = ops.transform(d)
        assert data_algebra.test_util.equivalent_frames(res, expect)
