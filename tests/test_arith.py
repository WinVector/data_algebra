
import numpy
import pandas

from data_algebra.data_ops import *
import data_algebra.util
import data_algebra.test_util

def test_arith_1():
    d = pandas.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [6, 9, 3, 8],
    })

    ops = describe_table(d, table_name='d'). \
        extend({
            'a': 'x + y',
            'b': 'x - y',
            'c': 'x * y',
            'd': 'x / y',
            'e': 'x + y / 2',
            'f': 'x*x + y*y',
            'g': '(x*x + y*y).sqrt()',
            'h': 'x*x == y**2',
        })

    res = ops.transform(d)

    expect = d.copy()
    expect['a'] = expect.x + expect.y
    expect['b'] = expect.x - expect.y
    expect['c'] = expect.x * expect.y
    expect['d'] = expect.x / expect.y
    expect['e'] = expect.x + (expect.y / 2)
    expect['f'] = (expect.x * expect.x) + (expect.y * expect.y)
    expect['g'] = numpy.sqrt(((expect.x * expect.x) + (expect.y * expect.y)))
    expect['h'] = expect.x == expect.y

    assert data_algebra.test_util.equivalent_frames(expect, res)