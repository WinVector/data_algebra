
import numpy

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util


def test_scalar_columns():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4]}
    )

    ops = (
        describe_table(d, table_name="d")
            .extend({
                'z1': numpy.nan,
                'z2': None,
                'n1': 1,
                'c1': 'a',
                'b1': True,
                'f1': 2.1,
            })
    )
    res = ops.transform(d)

    expect = d.copy()
    expect['z1'] = numpy.nan
    expect['z2'] = None
    expect['n1'] = 1
    expect['c1'] = 'a'
    expect['b1'] = True
    expect['f1'] = 2.1

    assert data_algebra.test_util.equivalent_frames(res, expect)

    # data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
