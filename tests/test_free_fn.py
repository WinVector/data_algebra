
import numpy

import data_algebra.test_util
from data_algebra.data_ops import *

import data_algebra
import data_algebra.util
import data_algebra.test_util
import data_algebra.SQLite

import pytest


def test_free_fn():
    # show unknown fns are not allowed, unless registered
    d = data_algebra.default_data_model.pd.DataFrame(
        {"b": [-1, 2]}
    )

    # fn wasn't defined, so definition should raise error
    with pytest.raises(KeyError):
        ops = describe_table(d, table_name='d'). \
            extend({'r': 'FUNCTION_WE_DONT_KNOW_ABOUT(b)'})

    # define the fn and try again
    data_algebra.default_data_model.user_fun_map['FUNCTION_WE_DONT_KNOW_ABOUT'] = numpy.abs

    ops = describe_table(d, table_name='d'). \
        extend({'r': 'FUNCTION_WE_DONT_KNOW_ABOUT(b)'})

    # don't promote fn to method, which would error out
    assert data_algebra.test_util.formats_to_self(ops)

    res_pandas = ops.transform(d)
    del data_algebra.default_data_model.user_fun_map['FUNCTION_WE_DONT_KNOW_ABOUT']

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"b": [-1, 2], "r": [1, 2]}
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    # can make SQL (the whole point of allowing this path)
    handle = data_algebra.SQLite.SQLiteModel().db_handle(conn=None)
    sql = handle.to_sql(ops)
    assert isinstance(sql, str)
    assert 'FUNCTION_WE_DONT_KNOW_ABOUT("b")' in sql
