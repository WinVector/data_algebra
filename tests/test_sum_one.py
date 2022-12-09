
import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, ex


def test_sum_one():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'group': ['a', 'a', 'b', 'b', 'b'],
        'one': [1, 1, 1, 1, 1],
    })
    ops = (
        descr(d=d)
            .project(
            {
                'sum_one': 'one.sum()',
                'sum_1': '(1).sum()',
            },
            group_by=['group']
        )
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'group': ['a', 'b'],
        'sum_one': [2, 3],
        'sum_1': [2, 3],
        })
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)

