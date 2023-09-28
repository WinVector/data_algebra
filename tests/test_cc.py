import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.connected_components import connected_components
from data_algebra.test_util import formats_to_self


def test_cc_small():
    f = [1, 2, 4]
    g = [2, 3, 5]
    res = connected_components(f, g)
    expect = [1, 1, 4]
    assert res == expect

    f2 = [1, 3, 4]
    g2 = [2, 2, 5]
    res2 = connected_components(f2, g2)
    assert res == expect


def test_cc():
    f = [1, 4, 6, 2, 1]
    g = [2, 5, 7, 3, 7]
    res = connected_components(f, g)
    expect = [1, 4, 1, 1, 1]
    assert res == expect


def test_cc_ops():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"f": [1, 4, 6, 2, 1], "g": [2, 5, 7, 3, 7],}
    )

    ops = describe_table(d).extend({"c": "f.co_equalizer(g)"})
    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"f": [1, 4, 6, 2, 1], "g": [2, 5, 7, 3, 7], "c": [1, 4, 1, 1, 1],}
    )
    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_cc_ops_f():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"f": [1, 4, 6, 2, 1], "g": [2, 5, 7, 3, 7],}
    )

    ops = describe_table(d).extend({"c": "connected_components(f, g)"})
    assert formats_to_self(ops)

    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"f": [1, 4, 6, 2, 1], "g": [2, 5, 7, 3, 7], "c": [1, 4, 1, 1, 1],}
    )
    assert data_algebra.test_util.equivalent_frames(res, expect)
