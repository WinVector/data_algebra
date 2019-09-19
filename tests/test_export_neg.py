from data_algebra.data_ops import *


def test_export_neg():
    ops = TableDescription("d", ["probability"]).extend({"sort_key": "-probability"})
    objs_R = ops.collect_representation(dialect="R")
    expr = objs_R[1]["ops"]["sort_key"]
    assert "neg" not in expr
