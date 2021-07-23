
import numpy

from data_algebra.data_ops import *
import data_algebra.test_util

import pytest


def test_types_table_types():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [None, 1.0],
        'y': ['a', 'b'],
    })
    descr = describe_table(d, table_name='d')
    assert descr.column_types['x'] == numpy.float64
    assert descr.column_types['y'] == type('a')


def test_types_concat_good():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [None, 1.0],
        'y': ['a', 'b'],
    })
    ops = (
        describe_table(d, table_name='d')
            .concat_rows(b=describe_table(d, table_name='d'))
    )
    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [None, 1.0, None, 1.0],
        'y': ['a', 'b', 'a', 'b'],
        'source_name': ['a', 'a', 'b', 'b'],
        })

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_types_concat_bad():
    d1 = data_algebra.default_data_model.pd.DataFrame({
        'x': [None, 1.0],
        'y': ['a', 'b'],
    })
    d2 = data_algebra.default_data_model.pd.DataFrame({
        'x': [None, 1.0],
        'y': [0, 1],
    })
    # TODO: get it to raise here
    ops = (
        describe_table(d1, table_name='d1')
            .concat_rows(b=describe_table(d2, table_name='d2'))
    )

    with pytest.raises(AssertionError):
        ops.eval({'d1': d1, 'd2': d2})
