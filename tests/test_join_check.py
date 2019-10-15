
import pandas

import pytest

from data_algebra.data_ops import *
from data_algebra.arrow import *


def test_join_check():
    d1 = pandas.DataFrame({
        'key': ['a', 'b'],
        'x': [1, 2],
    })

    d2 = pandas.DataFrame({
        'key': ['b', 'c'],
        'y': [3, 4],
    })

    with pytest.raises(ValueError):
        describe_table(d1).\
            natural_join(
                b=describe_table(d2),
                by=['key'])
