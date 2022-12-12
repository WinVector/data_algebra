
import datetime
import data_algebra
import data_algebra.data_model
import data_algebra.test_util


def test_date_max_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "g": ["a", "a", "b"],
        "v": ['2020-01-01', "2023-01-01", "2020-01-01"],
    })
    ops = (
        data_algebra.descr(d=d)
            .extend({"v": "v.parse_date()"})
            .project(
                {
                    "min_v": "v.min()",
                    "max_v": "v.max()",
                },
                group_by=["g"]
                )
            .order_rows(["g"])
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        "g": ["a", "b"],
        "min_v": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)],
        "max_v": [datetime.date(2023, 1, 1), datetime.date(2020, 1, 1)],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    # TODO: add SQLite PARSEDATE
    # data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


def test_date_max_2():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        "g": ["a", "a", "b"],
        "v": ['2020-01-01', "2023-01-01", "2020-01-01"],
    })
    ops = (
        data_algebra.descr(d=d)
            .extend({"v": "v.parse_date('%Y-%m-%d')"})
            .project(
                {
                    "min_v": "v.min()",
                    "max_v": "v.max()",
                },
                group_by=["g"]
                )
            .order_rows(["g"])
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        "g": ["a", "b"],
        "min_v": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)],
        "max_v": [datetime.date(2023, 1, 1), datetime.date(2020, 1, 1)],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    # data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
