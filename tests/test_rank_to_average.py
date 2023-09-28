
import numpy
import pytest

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SparkSQL
import data_algebra.SQLite
import data_algebra.PostgreSQL
import data_algebra.BigQuery
import data_algebra.test_util
import data_algebra.solutions


def test_rank_to_average():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'v': [1, 1, 2, 2, 3, 3],
        'g': ['a', 'a', 'a', 'b', 'b', 'b'],
    })
    ops = data_algebra.solutions.rank_to_average(
        descr(d=d),
        order_by=['v'],
        partition_by=['g'],
        rank_column_name='r',
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'v': [1, 1, 2, 2, 3, 3],
        'g': ['a', 'a', 'a', 'b', 'b', 'b'],
        'r': [1.5, 1.5, 3.0, 1.0, 2.5, 2.5],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
    )
