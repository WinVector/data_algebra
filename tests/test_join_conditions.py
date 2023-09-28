
import data_algebra
import data_algebra.test_util
import data_algebra.data_ops
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.view_representations
import data_algebra.util


def test_join_conditions_on_treatment():
    # check we are cannonicalizing joins into one notation
    expect = (['a', 'b'], ['a', 'b'])
    res1 = data_algebra.view_representations._convert_on_clause_to_parallel_lists(["a", "b"])
    assert res1 == expect
    res2 = data_algebra.view_representations._convert_on_clause_to_parallel_lists(("a", "b"))
    assert res2 == expect
    res3 = data_algebra.view_representations._convert_on_clause_to_parallel_lists({"a": "a", "b": "b"})
    assert res3 == expect
    res4 = data_algebra.view_representations._convert_on_clause_to_parallel_lists([("a", "a"), ("b", "b")])
    assert res4 == expect
    res5 = data_algebra.view_representations._convert_on_clause_to_parallel_lists((("a", "a"), ("b", "b")))
    assert res5 == expect


def test_join_conditions_on_back():
    expect1 = ['a', 'b']
    res1 = data_algebra.view_representations._convert_parallel_lists_to_on_clause(["a", "b"], ["a", "b"])
    assert res1 == expect1
    expect2 = [('a', 'b'), ('b', 'a')]
    res2 = data_algebra.view_representations._convert_parallel_lists_to_on_clause(["a", "b"], ["b", "a"])
    assert res2 == expect2
    res3 = data_algebra.view_representations._convert_parallel_lists_to_on_clause(["a", "b"], ["b", "b"])
    expect3 =  [('a', 'b'), 'b']
    assert res3 == expect3


def test_join_conditions_on_join():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2, 3],
        "a": [4, 5, 6],
    })
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({
        "y": [1, 2, 3],
        "b": [7, 8, 9],
    })
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 2, 3],
        "a": [4, 5, 6],
        "b": [7, 8, 9],
    })
    ops = (
        descr(d1=d1)
            .natural_join(
                descr(d2=d2),
                on=[("x", "y")],
                jointype="left",
            )
    )
    res = ops.eval({"d1": d1, "d2": d2})
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops, data={"d1": d1, "d2": d2}, expect=expect,
    )
