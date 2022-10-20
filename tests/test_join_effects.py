
import warnings
import pytest

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.util


def test_join_warn1():
    d_left = data_algebra.default_data_model.pd.DataFrame(
        {"k": ["a", "a", "b"], "x": [1, None, 3], "y": [1, None, None],}
    )

    d_right = data_algebra.default_data_model.pd.DataFrame(
        {"k": ["a", "b", "q"], "y": [10.0, 20.0, 30.0],}
    )

    ops = describe_table(d_left, table_name="d_left").natural_join(
        b=describe_table(d_right, table_name="d_right"), by="k", jointype="LEFT"
    )

    # https://stackoverflow.com/a/45671804/6901725
    # https://github.com/pytest-dev/pytest/issues/9404#issue-1076710891
    # https://stackoverflow.com/questions/45671803/how-to-use-pytest-to-assert-no-warning-is-raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ops.eval({"d_left": d_left, "d_right": d_right})

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"k": ["a", "a", "b"], "x": [1.0, None, 3.0], "y": [1.0, 10.0, 20.0],}
    )

    data_algebra.test_util.check_transform(
        ops=ops, data={"d_left": d_left, "d_right": d_right}, expect=expect
    )


def test_join_where_merge():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1], "y": [2], "z": ["q"]})
    ops = (
        descr(d=d)
            .natural_join(
                descr(d=d),
                by=[],
                jointype='outer',
                )
            .select_rows("x > y")
    )
    sql_str = ops.to_sql()
    assert isinstance(sql_str, str)
    # TODO: merge where clause from select_rows() into join step (and check).
