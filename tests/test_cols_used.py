import data_algebra
import data_algebra.util
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex


def test_cols_used():
    table = TableDescription(table_name="d", column_names=["a", "b", "c", "d"])

    ops = table.select_columns(["a", "b"]).natural_join(
        b=table.select_columns(["a", "c"]), by=["a"], jointype="INNER"
    )

    used = ops.columns_used()
    d_used = used["d"]

    assert set(["a", "b", "c"]) == d_used

    ops2 = (
        TableDescription(table_name="d", column_names=["a", "b", "c", "d"])
        .select_columns(["a", "b"])
        .natural_join(
            b=TableDescription(
                table_name="d", column_names=["a", "b", "c", "d"]
            ).select_columns(["a", "c"]),
            by=["a"],
            jointype="INNER",
        )
    )

    used2 = ops2.columns_used()
    d_used2 = used2["d"]

    assert set(["a", "b", "c"]) == d_used2
