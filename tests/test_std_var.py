import pytest

import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.util
import numpy
import data_algebra.MySQL
import data_algebra.SQLite


# confirm implementation is sample standard deviation and sample variance

def test_std_var_project():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({'x': [1., 3.], 'g': ['a', 'a']})
    ops = (
        descr(d=d)
            .project(
                {
                    'v': 'x.var()',
                    's': 'x.std()',
                },
                group_by=['g'],
            )
    )
    expect = pd.DataFrame({'g': ['a'], 'v': [2.0], 's': [numpy.sqrt(2.0)]})
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        valid_for_empty=False,
    )


def test_std_var_extend_disallow_ordered_var():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({'x': [1., 3.], 'g': ['a', 'a'], 'z': [1, 2]})
    with pytest.raises(ValueError):
        ops = (
            descr(d=d)
                .extend(
                    {
                        'v': 'x.var()',
                    },
                    partition_by=['g'],
                    order_by=['z'],
                )
        )


def test_std_var_extend_disallow_ordered_std():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({'x': [1., 3.], 'g': ['a', 'a'], 'z': [1, 2]})
    with pytest.raises(ValueError):
        ops = (
            descr(d=d)
                .extend(
                    {
                        's': 'x.std()',
                    },
                    partition_by=['g'],
                    order_by=['z'],
                )
        )
