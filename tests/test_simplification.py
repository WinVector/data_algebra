import data_algebra
import data_algebra.test_util
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode, SelectColumnsNode, SelectRowsNode
from data_algebra.data_ops import data, descr, describe_table, ex

import pytest


def test_simplification_1():
    ops = (
        TableDescription(table_name="d", column_names=["col1", "col2", "col3"])
        .extend({"sum23": "col2 + col3"})
        .extend({"x": 1})
        .extend({"x": 2})
        .extend({"x": 3})
        .extend({"x": 4})
        .extend({"x": 5})
        .select_columns(["x", "sum23", "col3"])
    )
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"col1": [1, 2], "col2": [3, 4], "col3": [4, 5]}
    )
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [5, 5], "sum23": [7, 9], "col3": [4, 5],}
    )
    assert data_algebra.test_util.equivalent_frames(res, expect)
    assert isinstance(ops, SelectColumnsNode)
    assert isinstance(ops.sources[0], ExtendNode)
    assert isinstance(ops.sources[0].sources[0], TableDescription)


def test_simplification_2():
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({"col1": [0, 1], "col2": [1, 0],})

    ops2 = (
        describe_table(d2, table_name="d2")
        .select_rows("col2 > 0")
        .select_rows("col1 / col2 > 0")
    )
    res = ops2.transform(d2)
    assert set(res.columns) == set(["col1", "col2"])
    assert res.shape[0] == 0
    assert isinstance(ops2, SelectRowsNode)
    assert isinstance(ops2.sources[0], SelectRowsNode)
    assert isinstance(ops2.sources[0].sources[0], TableDescription)
