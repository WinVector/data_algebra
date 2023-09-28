import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util


def test_join_multi_key():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", "zz", "zz"],
            "b": ["c", "zz", "d", "zz"],
            "y": ["3", "zz", "4", "zz"],
        }
    )

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", "zz", "zz"],
            "b": ["c", "zz", "d", "zz"],
            "x": ["1", "2", "zz", "zz"],
        }
    )

    ops = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=["a", "b"], jointype="left"
    )

    res_pandas = ops.eval({"d1": d1, "d2": d2})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": ["a", "b", "zz", "zz"],
            "b": ["c", "zz", "d", "zz"],
            "y": ["3", "zz", "4", "zz"],
            "x": ["1", "2", "zz", "zz"],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    data_algebra.test_util.check_transform(
        ops=ops, data={"d1": d1, "d2": d2}, expect=expect
    )
