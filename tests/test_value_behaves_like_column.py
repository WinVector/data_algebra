
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.util


def test_value_behaves_like_column_extend():
    d = data_algebra.default_data_model.pd.DataFrame({
            'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            'one': 1,
        })
    ops = (
        descr(d=d)
            .extend(
                {
                    'sum_1': '(1).sum()',
                    'sum_one': 'one.sum()',
                },
                partition_by=['ID']
            ))
    expect = data_algebra.default_data_model.pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'one': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'sum_1': [2, 2, 1, 1, 4, 4, 4, 4, 2, 2, 1],
        'sum_one': [2, 2, 1, 1, 4, 4, 4, 4, 2, 2, 1],
        })
    data_algebra.test_util.check_transform(
        ops,
        data=d,
        expect=expect)


def test_value_behaves_like_column_project():
    d = data_algebra.default_data_model.pd.DataFrame({
            'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            'one': 1,
        })
    ops = (
        descr(d=d)
            .project(
                {
                    'sum_1': '(1).sum()',
                    'sum_one': 'one.sum()',
                },
                group_by=['ID']
            ))
    expect = data_algebra.default_data_model.pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'sum_1': [2, 1, 1, 4, 2, 1],
        'sum_one': [2, 1, 1, 4, 2, 1],
        })
    data_algebra.test_util.check_transform(
        ops,
        data=d,
        expect=expect)
