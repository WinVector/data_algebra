
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *

def test_extend_transform_1():
    d = pandas.DataFrame({
        'x': [1, 2]
    })
    ops = describe_table(d, table_name='d'). \
        extend({'y': 1})
    expect = pandas.DataFrame({
        'x': [1, 2],
        'y': [1, 1],
        })
    data_algebra.test_util.check_transform(ops, d, expect)
