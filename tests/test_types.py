from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import data_algebra.test_util


def test_types_concat_good():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [None, 1.0], "y": ["a", "b"],}
    )
    ops = describe_table(d, table_name="d").concat_rows(
        b=describe_table(d, table_name="d")
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [None, 1.0, None, 1.0],
            "y": ["a", "b", "a", "b"],
            "source_name": ["a", "a", "b", "b"],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
