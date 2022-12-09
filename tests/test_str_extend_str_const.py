
import data_algebra
import data_algebra.data_model
import data_algebra.test_util


def test_str_extend_str_const():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2]
        })
    ops = data_algebra.descr(d=d).extend({"n": '"xx"'})
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2],
        "n": ["xx", "xx"],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_str_extend_str_const_col():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2]
        })
    ops = data_algebra.descr(d=d).extend({"n": "x"})
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2],
        "n": [1, 2],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_str_extend_str_const_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2]
        })
    ops = data_algebra.descr(d=d).extend({"n": '"x"'})
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2],
        "n": ["x", "x"],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_str_extend_str_const_3():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2]
        })
    ops = data_algebra.descr(d=d).extend({"n": "'x'"})
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2],
        "n": ["x", "x"],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_str_extend_const_4():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2]
        })
    ops = data_algebra.descr(d=d).extend({"n": '"xx"', "n2": "x", "n3": '"x"'})
    res = ops.transform(d)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "x": [1, 2],
        "n": ["xx", "xx"],
        "n2": [1, 2],
        "n3": ["x", "x"],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
