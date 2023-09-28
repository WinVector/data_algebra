import data_algebra
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode, OrderRowsNode, SelectColumnsNode
from data_algebra.data_ops import data, descr, describe_table, ex


def test_shorten():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x_s": ["s_03", "s_04", "s_02", "s_01", "s_03", "s_01"],
            "x_n": ["n_13", "n_48", "n_77", "n_29", "n_91", "n_93"],
            "y": [1.0312223, -1.3374379, -1.9347144, 1.2772708, -0.1238039, 0.3058670],
        }
    )
    table_desc = describe_table(d, table_name="d")

    ops1 = table_desc.order_rows(["x_s"])
    assert isinstance(ops1, OrderRowsNode)

    ops2 = ops1.extend({"y_mean": "y.mean()"})
    assert not isinstance(ops2.sources[0], OrderRowsNode)

    ops3 = ops1.select_columns(["x_s", "y"]).select_columns(["y"])
    assert not isinstance(ops2.sources[0], SelectColumnsNode)
