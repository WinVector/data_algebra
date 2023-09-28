
import numpy
import pytest

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SparkSQL
import data_algebra.test_util
import data_algebra.solutions


def test_locf():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'v': [1., numpy.nan, 3., numpy.nan, 2., numpy.nan],
        'g': ['a', 'a', 'a', 'b', 'b', 'b'],
        'o': [1, 2, 3, 4, 5, 6],
    })
    ops = data_algebra.solutions.last_observed_carried_forward(
        descr(d=d),
        order_by=['o'],
        partition_by=['g'],
        value_column_name='v',
        selection_predicate='is_bad()',
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'v': [1.0, 1.0, 3.0, None, 2.0, 2.0],
        'g': ['a', 'a', 'a', 'b', 'b', 'b'],
        'o': [1, 2, 3, 4, 5, 6],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            data_algebra.SparkSQL.SparkSQLModel(),
        },
    )
