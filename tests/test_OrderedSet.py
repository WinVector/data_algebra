import pytest

import data_algebra.OrderedSet


def test_OrderedSet_1():
    expect = ["x_0", "x_1", "x_2", "x_3", "x_5", "x_6", "x_7", "x_8", "x_9"]
    s = data_algebra.OrderedSet.OrderedSet(expect)
    res = [v for v in s]
    assert expect == res

    expect2 = expect.copy()
    expect2.append("x_4")
    s.update(["x_3", "x_4"])
    res2 = [v for v in s]
    assert expect2 == res2

    s.add("x_1")
    res3 = [v for v in s]
    assert expect2 == res3


def test_OrderedSet_ordered_intersect():
    res = list(data_algebra.OrderedSet.ordered_intersect(["a", "b", "c"], ["c", "a"]))
    expect = ["a", "c"]
    assert res == expect


def test_OrderedSet_ordered_union():
    res = list(data_algebra.OrderedSet.ordered_union(["a", "b", "c"], ["d", "a"]))
    expect = ["a", "b", "c", "d"]
    assert res == expect


def test_OrderedSet_ordered_diff():
    res = list(data_algebra.OrderedSet.ordered_diff(["a", "b", "c"], ["c", "z"]))
    expect = ["a", "b"]
    assert res == expect


def test_OrderedSet_rejects_str():
    with pytest.raises(Exception):
        data_algebra.OrderedSet.OrderedSet('ab')


def test_OrderedSet_formats_to_self():
    a = data_algebra.OrderedSet.OrderedSet(['ab'])
    repr_str = a.__repr__()
    assert repr_str == "OrderedSet(['ab'])"
    str_str = str(a)
    assert str_str == "{'ab'}"
