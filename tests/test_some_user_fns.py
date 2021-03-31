
import pandas

from data_algebra.data_ops import *

import data_algebra.BigQuery
import data_algebra.test_util


def test_TRIMSTR():
    trim_0_5 = data_algebra.BigQuery.TRIMSTR(start=0, stop=5)

    d = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
    })
    ops = describe_table(d, table_name='d') .\
        extend({
         'nx': trim_0_5('x')
        })
    res = ops.transform(d)

    expect = pandas.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
        'nx': ['01234', 'abcde'],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
