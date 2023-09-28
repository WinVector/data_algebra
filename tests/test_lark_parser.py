import pytest

import data_algebra
import data_algebra.expr_rep
import data_algebra.parse_by_lark
from data_algebra.data_ops import data, descr, describe_table, ex


def test_lark_1():
    # raises an exception if no lark parser
    raw_tree = data_algebra.parse_by_lark.parser.parse("1 + 1/2")


def test_lark_1b():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "y": [2, 8, -1],}
    )
    data_def = {k: v for (k, v) in describe_table(d1).column_map().items()}
    tree = data_algebra.parse_by_lark.parse_by_lark("x == 1", data_def=data_def)


def test_lark_1c():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "y": [2, 8, -1],}
    )
    data_def = {k: v for (k, v) in describe_table(d1).column_map().items()}
    with pytest.raises(Exception):
        tree = data_algebra.parse_by_lark.parse_by_lark("x = 1", data_def=data_def)


def test_lark_1b():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "y": [2, 8, -1],}
    )
    data_def = {k: v for (k, v) in describe_table(d1).column_map().items()}
    tree = data_algebra.parse_by_lark.parse_by_lark("7", data_def=data_def)


def test_lark_2():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "y": [2, 8, -1],}
    )
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [11.6, None, 13], "b": [True, None, True], "c": ["x", "y", None],}
    )
    data_def = {k: v for (k, v) in describe_table(d1).column_map().items()}
    data_def.update({k: v for (k, v) in describe_table(d2).column_map().items()})
    expr = "x / (a+b)"
    # raw_tree = data_algebra.parse_by_lark.parser.parse(expr + "\n")
    # v = data_algebra.parse_by_lark._walk_lark_tree(raw_tree, data_def=data_def)
    tree = data_algebra.parse_by_lark.parse_by_lark(expr, data_def=data_def)
    assert isinstance(tree, data_algebra.expr_rep.PreTerm)
    assert str(tree) == "x / (a + b)"


def test_lark_quote():
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [11.6, None, 13], "b": [True, None, True], "c": ["x", "y", None],}
    )
    data_def = {k: v for (k, v) in describe_table(d2).column_map().items()}
    expr = 'c == "x"'
    # raw_tree = data_algebra.parse_by_lark.parser.parse(expr + "\n")
    # v = data_algebra.parse_by_lark._walk_lark_tree(raw_tree, data_def=data_def)
    tree = data_algebra.parse_by_lark.parse_by_lark(expr, data_def=data_def)
    assert isinstance(tree, data_algebra.expr_rep.PreTerm)
    assert str(tree) == "c == 'x'"
