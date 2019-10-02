
import pandas
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util

# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

def test_math():

    d = pandas.DataFrame({
        'g': [1, 2, 2, 3, 3, 3],
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10, 40, 50, 70, 80, 90],
    })

    ops = describe_table(d). \
        extend({
            'v_exp': 'v.exp()',
            'v_sin': 'v.sin()',
            'g_plus_x': 'g+x',
        })

    res1 = ops.transform(d)

    expect1 = pandas.DataFrame({
        'g': [1, 2, 2, 3, 3, 3],
        'x': [1, 4, 5, 7, 8, 9],
        'v': [10, 40, 50, 70, 80, 90],
        'v_exp': [22026.465794806718, 2.3538526683702e+17, 5.184705528587072e+21, 2.515438670919167e+30,
                  5.54062238439351e+34, 1.2204032943178408e+39],
        'v_sin': [-0.5440211108893699, 0.7451131604793488, -0.26237485370392877, 0.7738906815578891,
                  -0.9938886539233752, 0.8939966636005579],
        'g_plus_x': [2, 6, 7, 10, 11, 12],
    })


    assert data_algebra.util.equivalent_frames(res1, expect1)

    # TODO: try these through the DB (probably PostgreSQL)
