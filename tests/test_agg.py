import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util


def test_agg():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4], "g": [1, 1, 2, 2],}
    )

    ops = describe_table(d, table_name="d").project({"x": "x.max()"}, group_by=["g"])

    res = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame({"g": [1, 2], "x": [2, 4],})

    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_agg_fn():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4], "g": [1, 1, 2, 2],}
    )

    def our_fn(vals):
        return ", ".join(sorted([str(vi) for vi in set(vals)]))

    ops = describe_table(d, table_name="d").project(
        {"x": user_fn(our_fn, ["x"])}, group_by=["g"]
    )

    res = ops.transform(d)

    expect = data_algebra.default_data_model.pd.DataFrame(
        {"g": [1, 2], "x": ["1, 2", "3, 4"],}
    )

    assert data_algebra.test_util.equivalent_frames(res, expect)
