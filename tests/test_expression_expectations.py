
"""direct tests from op catalog"""

import os
import pickle
import gzip

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
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
    expectations = expectation_map['expectations']
    u_results = expectation_map['u_results']
    for op, op_class, exp, ops, expect in expectations:
        if data_algebra.data_model.default_data_model().is_appropriate_data_instance(expect):
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
                    # goes into cache
                    data_algebra.test_util.check_transform(
                        ops=ops,
                        data=d,
                        expect=expect,
                        valid_for_empty=False,
                        empty_produces_empty=False,
                        models_to_skip=to_skip,
                        try_on_Polars=False,  # TODO: complete coverage, and turn this on
                    )
    for op, op_class, exp, ops, expect in u_results:
        # re-run, but don't check value
        res = ops.transform(d)
        assert data_algebra.data_model.default_data_model().is_appropriate_data_instance(res)


def test_expression_expectations_direct():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb
    # includes not cached tests
    expectation_path = os.path.join(dir_path, "expr_expectations.pkl.gz")
    with gzip.open(expectation_path, 'rb') as in_f:
        expectation_map = pickle.load(in_f)
    d = expectation_map['d']
    expectations = expectation_map['expectations']
    u_results = expectation_map['u_results']
    db_handles = data_algebra.test_util.get_test_dbs()
    for h in db_handles:
        h.insert_table(d, table_name='d', allow_overwrite=True)
    for op, op_class, exp, ops, expect in expectations:
        if data_algebra.data_model.default_data_model().is_appropriate_data_instance(expect):
            res = ops.transform(d)
            assert data_algebra.test_util.equivalent_frames(res, expect)
            matching = methods_table.loc[methods_table['expression'] == exp, :]
            if matching.shape[0] == 1:
                to_test = {c for c in matching.columns if 'Model' in c}
                for h in db_handles:
                    model_name = str(h.db_model)
                    if (model_name in to_test) and (matching[model_name].values[0] == 'y'):
                        res_db = h.read_query(ops)
                        assert data_algebra.test_util.equivalent_frames(res_db, expect)
    for op, op_class, exp, ops, expect in u_results:
        if data_algebra.data_model.default_data_model().is_appropriate_data_instance(expect):
            res = ops.transform(d)
            assert data_algebra.data_model.default_data_model().is_appropriate_data_instance(res)
            matching = methods_table.loc[methods_table['expression'] == exp, :]
            if matching.shape[0] == 1:
                to_test = {c for c in matching.columns if 'Model' in c}
                for h in db_handles:
                    model_name = str(h.db_model)
                    if (model_name in to_test) and (matching[model_name].values[0] == 'y'):
                        res_db = h.read_query(ops)
                        assert data_algebra.data_model.default_data_model().is_appropriate_data_instance(res_db)
    for h in db_handles:
        h.drop_table('d')
        h.close()
