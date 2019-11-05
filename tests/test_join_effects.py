import pandas
import pytest

from data_algebra.data_ops import *
import data_algebra.util


def test_join_warn1():
    d_left = pandas.DataFrame(
        {"k": ["a", "a", "b"], "x": [1, None, 3], "y": [1, None, None],}
    )

    d_right = pandas.DataFrame({"k": ["a", "b", "q"], "y": [10, 20, 30],})

    ops = describe_table(d_left, table_name="d_left").natural_join(
        b=describe_table(d_right, table_name="d_right"), by="k", jointype="LEFT"
    )

    # https://stackoverflow.com/questions/45671803/how-to-use-pytest-to-assert-no-warning-is-raised
    with pytest.warns(None) as record:
        res = ops.eval({"d_left": d_left, "d_right": d_right})
        assert len(record) == 0

    expect = pandas.DataFrame(
        {"k": ["a", "a", "b"], "x": [1.0, None, 3.0], "y": [1.0, 10.0, 20.0],}
    )

    assert data_algebra.util.equivalent_frames(res, expect)


def test_join_wrap1():
    d_left = pandas.DataFrame(
        {"k": ["a", "a", "b"], "x": [1, None, 3], "y": [1, None, None],}
    )

    d_right = pandas.DataFrame({"k": ["a", "b", "q"], "y": [10, 20, 30],})

    ops = wrap(d_left, table_name="d_left").natural_join(
        b=wrap(d_right, table_name="d_right"), by="k", jointype="LEFT"
    )

    res = ops.ex()

    expect = pandas.DataFrame(
        {"k": ["a", "a", "b"], "x": [1.0, None, 3.0], "y": [1.0, 10.0, 20.0],}
    )

    assert data_algebra.util.equivalent_frames(res, expect)
