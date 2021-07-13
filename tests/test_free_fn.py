
import data_algebra.test_util
from data_algebra.data_ops import *

import data_algebra
import data_algebra.util
import data_algebra.test_util
import data_algebra.SQLite

import pytest


def test_free_fn():
    # show that free functions are allowed
    d = data_algebra.default_data_model.pd.DataFrame(
        {"b": [1, 2]}
    )

    ops = describe_table(d, table_name='d') .\
        extend({'r': 'FUNCTION_WE_DONT_KNOW_ABOUT(b)'})

    # don't promote fn to method, which would error out
    assert data_algebra.test_util.formats_to_self(ops)

    # fn wasn't defined, so transform should raise error
    # TODO: allow user to pass in new fns at transform/eval
    with pytest.raises(KeyError):
        ops.transform(d)

    # can make SQL (the whole point of allowing this path)
    handle = data_algebra.SQLite.SQLiteModel().db_handle(conn=None)
    sql = handle.to_sql(ops)
    assert isinstance(sql, str)
    assert 'FUNCTION_WE_DONT_KNOW_ABOUT(' in sql
