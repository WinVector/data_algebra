
import data_algebra
import data_algebra.test_util
import data_algebra.data_ops
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util


def test_join_conditions_on_treatment():
    # check we are cannonicalizing joins into one notation
    expect = (['a', 'b'], ['a', 'b'])
    res1 = data_algebra.data_ops._convert_on_clause_to_parallel_lists(["a", "b"])
    assert res1 == expect
    res2 = data_algebra.data_ops._convert_on_clause_to_parallel_lists(("a", "b"))
    assert res2 == expect
    res3 = data_algebra.data_ops._convert_on_clause_to_parallel_lists({"a": "a", "b": "b"})
    assert res3 == expect
    res4 = data_algebra.data_ops._convert_on_clause_to_parallel_lists([("a", "a"), ("b", "b")])
    assert res4 == expect
    res5 = data_algebra.data_ops._convert_on_clause_to_parallel_lists((("a", "a"), ("b", "b")))
    assert res5 == expect


def test_join_conditions_on_back():
    expect1 = ['a', 'b']
    res1 = data_algebra.data_ops._convert_parallel_lists_to_on_clause(["a", "b"], ["a", "b"])
    assert res1 == expect1
    expect2 = [('a', 'b'), ('b', 'a')]
    res2 = data_algebra.data_ops._convert_parallel_lists_to_on_clause(["a", "b"], ["b", "a"])
    assert res2 == expect1
