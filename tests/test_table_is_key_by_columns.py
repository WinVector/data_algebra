import data_algebra


def test_table_is_keyed_by_columns():
    local_model = data_algebra.data_model.default_data_model()
    pd = local_model.pd
    d = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]}
    )

    assert local_model.table_is_keyed_by_columns(d, column_names=["a", "b"])

    assert not local_model.table_is_keyed_by_columns(d, column_names=["a"])
