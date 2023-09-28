import data_algebra
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex


def test_select_stacking():
    ops1 = (
        TableDescription(table_name="d", column_names=["a", "b", "c"])
        .select_columns(["a", "b"])
        .select_columns(["a", "b"])
    )
    ops1_str = format(ops1)
    assert ops1_str.count("select_columns") == 1

    ops2 = (
        TableDescription(table_name="d", column_names=["a", "b", "c"])
        .select_columns(["a", "b"])
        .select_columns(["a"])
    )
    ops2_str = format(ops2)
    assert ops2_str.count("select_columns") == 1
