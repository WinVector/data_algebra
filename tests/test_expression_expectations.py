
import os
import pickle
import gzip

import data_algebra
from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util
from data_algebra.op_catalog import methods_table
import data_algebra.SparkSQL
import data_algebra.BigQuery


def test_expression_expectations_1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb

    do_super_expensive_rechecks = True
    too_expensive_to_include_in_rechecks = [
        str(data_algebra.SparkSQL.SparkSQLModel()),
        str(data_algebra.BigQuery.BigQueryModel()),
    ]

    expectation_path = os.path.join(dir_path, "expr_expectations.pkl.gz")
    with gzip.open(expectation_path, 'rb') as in_f:
        expectation_map = pickle.load(in_f)
    d = expectation_map['d']
    e_expectations = expectation_map['e_expectations']
    g_expectations = expectation_map['g_expectations']
    w_expectations = expectation_map['w_expectations']
    u_results = expectation_map['u_results']

    ops_list = e_expectations + g_expectations + w_expectations
    for op, op_class, exp, ops, expect in ops_list:
        if data_algebra.default_data_model.is_appropriate_data_instance(expect):
            res = ops.transform(d)
            assert data_algebra.test_util.equivalent_frames(res, expect)
        if do_super_expensive_rechecks:
            matching = methods_table.loc[methods_table['expression'] == exp, :]
            if matching.shape[0] == 1:
                to_test = [c for c in matching.columns if 'Model' in c]
                to_skip = set([c for c in to_test if matching[c].values[0] != 'y'])
                for c in too_expensive_to_include_in_rechecks:
                    to_skip.add(c)
                if len(to_skip) < len(to_test):
                    data_algebra.test_util.check_transform(
                        ops=ops,
                        data=d,
                        expect=expect,
                        valid_for_empty=False,
                        empty_produces_empty=False,
                        models_to_skip=to_skip,
                    )
    for op, op_class, exp, ops, expect in u_results:
        # re-run, but don't check value
        res = ops.transform(d)
        assert data_algebra.default_data_model.is_appropriate_data_instance(res)
