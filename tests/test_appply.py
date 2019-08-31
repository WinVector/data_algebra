
import pandas
import numpy
import data_algebra.util
from data_algebra.data_ops import *
from data_algebra.data_pipe import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml

def test_apply():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    data_algebra.env.push_onto_namespace_stack(locals())

    d = pandas.DataFrame({'x': [-1, 0, 1, numpy.nan], 'y': [1, 2, numpy.nan, 3]})

    expect_1 = pandas.DataFrame({'x': [0.0], 'y': [2.0], 'z':[0.0]})
    expect_2 = pandas.DataFrame({'x': [0.0], 'y': [2.0], 'z': [0.0], 'q': [2.0]})

    ops0 = TableDescription('t1', ['x', 'y']) .\
        extend({'z': 'x / y'})  .\
        select_rows('z >= 0')


    ops0.eval_pandas({'t1': d})

    res_0_1 = ops0.transform(d)

    assert data_algebra.util.equivalent_frames(expect_1, res_0_1)

    res_0_2 = d >> ops0

    assert data_algebra.util.equivalent_frames(expect_1, res_0_2)

    ops1 = (
        TableDescription('t1', ['x', 'y']) >>
            Extend({'z': 'x / y'}) >>
            SelectRows('z >= 0')
        )

    res_1_1 = ops1.eval_pandas({'t1': d})

    assert data_algebra.util.equivalent_frames(expect_1, res_1_1)

    res_1_2 = ops1.transform(d)

    assert data_algebra.util.equivalent_frames(expect_1, res_1_2)

    res_1_3 = d >> ops1

    assert data_algebra.util.equivalent_frames(expect_1, res_1_3)

    ops2 = Locum() .\
        extend({'z': 'x / y'})  .\
        select_rows('z >= 0')

    res_2_1 = ops2.transform(d)

    assert data_algebra.util.equivalent_frames(expect_1, res_2_1)

    res_2_2 = d >> ops2

    assert data_algebra.util.equivalent_frames(expect_1, res_2_2)

    ops3 = ( Locum('dtab') >>
            Extend({'z': 'x / y'})  >>
            SelectRows('z >= 0')
        )

    res_3_1 = ops3.transform(d)

    assert data_algebra.util.equivalent_frames(expect_1, res_3_1)

    res_3_2 = d >> ops3

    assert data_algebra.util.equivalent_frames(expect_1, res_3_2)

    res_4 = d >> wrap_pipeline(
        Extend({'z': 'x / y'})  >>
            SelectRows('z >= 0') >>
            Extend({'q': 'y - z'})
        )

    assert data_algebra.util.equivalent_frames(expect_2, res_4)

    res_5 = d >> ( Locum() .
            extend({'z': 'x / y'}) .
            select_rows('z >= 0') .
            extend({'q': 'y - z'})
        )

    assert data_algebra.util.equivalent_frames(expect_2, res_5)

    res_6 = d >> wrap_ops(
            Extend({'z': 'x / y'}),
            SelectRows('z >= 0'),
            Extend({'q': 'y - z'})
        )

    assert data_algebra.util.equivalent_frames(expect_2, res_6)

    res_7 = d >> ( Locum() .
            extend({'z': 'x / y'}) .
            select_rows('z >= 0') .
            extend({'q': 'y - z'})
        )

    assert data_algebra.util.equivalent_frames(expect_2, res_7)
