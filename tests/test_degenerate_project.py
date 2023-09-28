import pytest

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex


def test_degenerate_project():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": ["a", "a", "b", "b"]}
    )

    ops_good = describe_table(d).project({"x2": "x.max()"})
    res = ops_good.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x2": [4],})
    assert data_algebra.test_util.equivalent_frames(res, expect)

    ops2 = describe_table(d).extend({"x2": "x.max()"})
    assert ops2.windowed_situation
    res2 = ops2.transform(d)
    expect2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": ["a", "a", "b", "b"], "x2": [4, 4, 4, 4],}
    )
    assert data_algebra.test_util.equivalent_frames(res2, expect2)

    ops3 = describe_table(d).extend({"x2": "2*x"})
    assert not ops3.windowed_situation
    res3 = ops3.transform(d)
    expect3 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": ["a", "a", "b", "b"], "x2": [2, 4, 6, 8],}
    )
    assert data_algebra.test_util.equivalent_frames(res3, expect3)

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "x.max() + x.max()"})

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "x"})

    with pytest.raises(ValueError):
        describe_table(d).project({"x2": "1"})
