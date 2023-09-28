import data_algebra

import data_algebra.test_util
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.test_util import formats_to_self
import data_algebra.util


def test_expr_parse():
    # check some differences in back to Python versus sending to Pandas
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4]}
    )

    ops0 = TableDescription(table_name="d", column_names=["a", "b", "c"]).extend(
        {"d": "a + 1"}
    )

    assert formats_to_self(ops0)

    res0 = ops0.transform(d)
    expect0 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [2, 1],}
    )
    assert data_algebra.test_util.equivalent_frames(res0, expect0)

    ops1 = TableDescription(table_name="d", column_names=["a", "b", "c"]).extend(
        {"d": "a.if_else(1, c)"}
    )

    assert formats_to_self(ops1)

    res1 = ops1.transform(d)
    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [1, 4],}
    )
    assert data_algebra.test_util.equivalent_frames(res1, expect1)
