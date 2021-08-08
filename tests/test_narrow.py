from data_algebra.data_ops import *


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
