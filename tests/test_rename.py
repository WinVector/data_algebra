
import pandas

import pytest

from data_algebra.data_ops import *
import data_algebra.SQLite
import data_algebra.test_util
import data_algebra.util


def test_rename_character():
    d = pandas.DataFrame({'x': [1], 'y': [2]})
    td = describe_table(d, table_name='d')

    swap = td.rename_columns({'y': 'x', 'x': 'y'})
    res_pandas = swap.transform(d)

    expect = data_algebra.pd.DataFrame({
        'y': [1],
        'x': [2],
    })

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(swap, data={'d': d}, expect=expect)

    # name collision is an error
    with pytest.raises(ValueError):
        td.rename_columns({'y': 'x'})
