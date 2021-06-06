
import numpy
import pandas

from data_algebra.data_ops import *
import data_algebra.test_util


def test_simple_expr_1():
    d_orig = pandas.DataFrame({
        'x': [1, 2, -3, 4]
    })
    d = d_orig.copy()

    ops = describe_table(d, table_name='d') .\
        extend({
            'z': 'x + 1',
            'sin_x': 'x.sin()',  # triggers numpy path
            'xm': '-x',
            'xs': '2 * (x-2).sign()',
        })

    res_pandas = ops.transform(d)

    expect = pandas.DataFrame({
        'x': [1, 2, -3, 4]
    })
    expect['z'] = expect['x'] + 1
    expect['sin_x'] = numpy.sin(expect['x'])
    expect['xm'] = -expect['x']
    expect['xs'] = 2 * numpy.sign(d['x'] - 2)

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    assert d.equals(d_orig)
