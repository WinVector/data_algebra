
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *


def test_use_1():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    def add_to_column(pipeline, colname, delta):
        return pipeline.extend({colname: f'{colname} + {delta}'})

    ops = data(d=d).use(add_to_column, 'ID', 5)
    res = ops.ex()

    ops2 = data(d=d).extend({'ID': 'ID + 5'})
    res2 = ex(ops2)

    assert data_algebra.test_util.equivalent_frames(res, res2)
