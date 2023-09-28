
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex


def test_narrow():
    ops = (
        TableDescription(
            table_name="stocks",
            column_names=["date", "trans", "symbol", "qty", "price"],
        )
        .extend({"cost": "qty * price"})
        .select_columns(["date", "cost"])
    )
    cused = ops.columns_used()
    assert cused["stocks"] == {"date", "price", "qty"}
