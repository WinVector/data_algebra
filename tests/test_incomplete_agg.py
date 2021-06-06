
import data_algebra
from data_algebra.data_ops import *

import data_algebra.test_util

import pytest


def test_incomplete_agg_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        'genus': (1, 1, 1, 2),
        'group': ('a', 'a', 'b', 'b'),
        'x': (1, 2, 3, 4),
        'y': (10, 20, 30, 40),
        })
    ops_1 = describe_table(d, table_name='d') .\
        project({
            'x': 'x.mean()',
            'y': 'y.mean()',
            },
            group_by=['genus', 'group'])
    res_1 = ops_1.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'genus': (1, 1, 2),
        'group': ('a', 'b', 'b'),
        'x': (1.5, 3, 4),
        'y': (15, 30, 40),
        })
    assert data_algebra.test_util.equivalent_frames(expect, res_1)

    with pytest.raises(ValueError):
        ops_bad = describe_table(d, table_name='d') .\
            project({
                'x': 'x.mean()',
                'y': 'y',  # error: forgoat aggregator!
                },
                group_by=['genus', 'group'])
