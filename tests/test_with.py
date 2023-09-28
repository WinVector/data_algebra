
import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.test_util
import data_algebra.SQLite


def test_with_query_example_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3]})
    ops = (
        describe_table(d, table_name="d")
        .extend({"z": "x + 1"})
        .extend({"q": "z + 2"})
        .extend({"h": "q + 3"})
    )

    res_pandas = ops.transform(d)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3], "z": [2, 3, 4], "q": [4, 5, 6], "h": [7, 8, 9]}
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_with_query_example_2():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"k": [1, 2, 3], "x": [5, 10, 15],}
    )

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"k": [1, 2, 3], "y": [-3, 2, 1],}
    )

    ops = (
        describe_table(d1, table_name="d1")
        .extend({"z": "x + 1"})
        .natural_join(
            b=describe_table(d2, table_name="d2").extend({"q": "y - 1"}),
            by=["k"],
            jointype="left",
        )
        .extend({"m": "(x + y) / 2"})
    )

    res_pandas = ops.eval({"d1": d1, "d2": d2})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "k": [1, 2, 3],
            "x": [5, 10, 15],
            "z": [6, 11, 16],
            "y": [-3, 2, 1],
            "q": [-4, 1, 0],
            "m": [1.0, 6.0, 8.0],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)

    data_algebra.test_util.check_transform(
        ops=ops, data={"d1": d1, "d2": d2}, expect=expect
    )
