import pytest

import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util


def test_value_behaves_like_column_extend():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
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
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'one': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'sum_1': [2, 2, 1, 1, 4, 4, 4, 4, 2, 2, 1],
        'sum_one': [2, 2, 1, 1, 4, 4, 4, 4, 2, 2, 1],
        })
    data_algebra.test_util.check_transform(
        ops,
        data=d,
        expect=expect)


def test_value_behaves_like_column_extend2a():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
            'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            'one': 1,
        })
    with pytest.raises(ValueError):
        (
        descr(d=d)
            .extend(
                {
                    'sum_1': '(1).cumsum()',
                    'sum_one': 'one.cumsum()',
                },
                partition_by=['ID'],
                order_by=['ID'],  # SQLite failed in this case, it treated SUM() as total even with window specifeid
            ))


def test_value_behaves_like_column_extend2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
            'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            'one': 1,
        })
    d['order'] = range(d.shape[0])
    ops = (
        descr(d=d)
            .extend(
                {
                    'sum_1': '(1).cumsum()',
                    'sum_one': 'one.cumsum()',
                },
                partition_by=['ID'],
                order_by=['order'],
            ))
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'one': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'order': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'sum_1': [1, 2, 1, 1, 1, 2, 3, 4, 1, 2, 1],
        'sum_one': [1, 2, 1, 1, 1, 2, 3, 4, 1, 2, 1],
        })
    data_algebra.test_util.check_transform(
        ops,
        data=d,
        expect=expect)


def test_value_behaves_like_column_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
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
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'sum_1': [2, 1, 1, 4, 2, 1],
        'sum_one': [2, 1, 1, 4, 2, 1],
        })
    data_algebra.test_util.check_transform(
        ops,
        data=d,
        expect=expect)
