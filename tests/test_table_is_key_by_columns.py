import data_algebra
import data_algebra.util


def test_table_is_keyed_by_columns():
    d = data_algebra.default_data_model.pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]})

    assert data_algebra.util.table_is_keyed_by_columns(d, ["a", "b"])

    assert not data_algebra.util.table_is_keyed_by_columns(d, ["a"])
