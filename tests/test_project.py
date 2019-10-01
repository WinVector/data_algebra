import pandas

import data_algebra.util
from data_algebra.data_ops import *


def test_R_yaml():
    d = pandas.DataFrame({
        'x': [1, 1, 2, 2],
        'y': [1, 2, 3, 4],
    })

    ops1 = describe_table(d). \
        project({'sum_y': 'y.sum()'})
    res1 = ops1.transform(d)
    expect1 = pandas.DataFrame({
        'sum_y': [10],
    })
    assert data_algebra.util.equivalent_frames(res1, expect1)

    ops2 = describe_table(d). \
        project({'sum_y': 'y.sum()'},
                group_by=['x'])
    res2 = ops2.transform(d)
    expect2 = pandas.DataFrame({
        'x': [1, 2],
        'sum_y': [3, 7],
    })
    assert data_algebra.util.equivalent_frames(res2, expect2)
