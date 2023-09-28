import pytest

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops_types import OperatorPlatform
from data_algebra.data_ops import describe_table
from data_algebra.arrow import DataOpArrow
import data_algebra.util
import data_algebra.arrow
from data_algebra.view_representations import TableDescription, ViewRepresentation


def test_arrow1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": ["a", "b", "b", "c", "c", "c"],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10.0, 40.0, 50.0, 70.0, 80.0, 90.0],
            "i": [True, True, False, False, False, False],
        }
    )

    table_description = TableDescription(
        table_name="d", column_names=["g", "x", "v", "i"]
    )

    id_ops_a = table_description.project(group_by=["g"]).extend(
        {"ngroup": "_row_number()",}, order_by=["g"]
    )

    id_ops_a.transform(d)

    id_ops_b = table_description.natural_join(id_ops_a, by=["g"], jointype="LEFT")

    id_ops_b.transform(d)

    # needs
    id_ops_b.columns_used()

    # produced
    id_ops_b.column_names

    a1 = data_algebra.arrow.DataOpArrow(id_ops_b)

    # check identity relns
    ri = a1.cod()
    li = a1.dom()
    li >> a1
    a1 >> ri
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> li >> a1)
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> a1 >> ri)
    a1.dom() >> a1
    a1.act_on(a1.dom())

    cols2_too_small = [c for c in (set(id_ops_b.column_names) - set(["i"]))]
    ordered_ops = TableDescription(
        table_name="d2", column_names=cols2_too_small
    ).extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    )
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
    # print(a2)

    with pytest.raises(ValueError):
        a1 >> a2

    cols2_too_large = list(id_ops_b.column_names) + ["q"]
    ordered_ops = TableDescription(
        table_name="d2", column_names=cols2_too_large
    ).extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    )
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
    # print(a2)

    with pytest.raises(ValueError):
        a1 >> a2

    ordered_ops = TableDescription(
        table_name="d2", column_names=id_ops_b.column_names
    ).extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    )
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops)

    unordered_ops = TableDescription(
        table_name="d3", column_names=ordered_ops.column_names
    ).extend(
        {
            "size": "_size()",
            "max_v": "v.max()",
            "min_v": "v.min()",
            "sum_v": "v.sum()",
            "mean_v": "v.mean()",
            "count_v": "v.count()",
            "size_v": "v.size()",
        },
        partition_by=["g"],
    )
    a3 = data_algebra.arrow.DataOpArrow(unordered_ops)
    # print(a3)

    f0 = (a3.act_on(a2.act_on(a1))).pipeline.__repr__()
    f1 = (a1 >> a2 >> a3).pipeline.__repr__()

    assert f1 == f0

    f2 = ((a1 >> a2) >> a3).pipeline.__repr__()

    assert f2 == f1

    f3 = (a1 >> (a2 >> a3)).pipeline.__repr__()

    assert f3 == f1

    a1 >> (a2 >> a3)

    r1 = (a1 >> a2 >> a3).transform(d)

    # Python default associates left to right so this is:
    # ((d >> a1) >> a2) >> a3
    r1b = d >> a1 >> a2 >> a3

    assert data_algebra.test_util.equivalent_frames(r1, r1b)

    # the preferred notation, work in operator space
    r2 = d >> (a1 >> a2 >> a3)

    assert data_algebra.test_util.equivalent_frames(r1, r2)

    # check identity relns
    ri = a1.cod()
    li = a1.dom()
    li >> a1
    a1 >> ri
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> li >> a1)
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> a1 >> ri)
    a1.dom() >> a1
    a1.act_on(a1.dom())


def test_arrow_cod_dom():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 4],})

    td = describe_table(d)

    a = td.extend({"z": "x.mean()"}, partition_by=["y"])

    a1 = DataOpArrow(a)

    a2 = DataOpArrow(a1.cod_as_table().extend({"ratio": "y / x"}))

    assert a1.cod_as_table() == a2.dom_as_table()
    assert a1.cod() == a2.dom()


def test_arrow_cod_dom_table():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 4],})
    td = describe_table(d)
    a = td.extend({"z": "x.mean()"}, partition_by=["y"])
    cod = a.cod()
    assert isinstance(cod, TableDescription)
    a_r = a >> cod
    assert a_r == a
    dom = a.dom()
    assert isinstance(dom, dict)
    assert len(dom) == 1
    dom_table = list(dom.values())[0]
    assert isinstance(dom_table, TableDescription)
    a_l = dom_table >> a
    assert a_l == a


def test_arrow_compose_2():
    b1 = DataOpArrow(
        TableDescription(column_names=["x", "y"], table_name=None).extend({"y": "x+1"})
    )
    b2 = DataOpArrow(
        TableDescription(column_names=["x", "y"], table_name=None).extend({"y": 9})
    )
    ops = b1 >> b2
    assert isinstance(ops.pipeline.sources[0], TableDescription)


def test_arrow_assoc_with_data():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2]})
    ops = DataOpArrow(data_algebra.descr(d=d).extend({"x": "x + 5"}))
    res_1 = d >> ops
    expect_1 = pd.DataFrame({"x": [6, 7]})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    ops_b = DataOpArrow(data_algebra.TableDescription(column_names=["x"]).extend({"x": "2 * x - 1"}))
    composite = ops >> ops_b
    assert isinstance(composite, DataOpArrow)
    expect_2 = pd.DataFrame({"x": [11, 13]})
    r_1 = d >> composite
    assert data_algebra.test_util.equivalent_frames(r_1, expect_2)
    r_2 = d >> (ops >> ops_b)
    assert data_algebra.test_util.equivalent_frames(r_2, expect_2)
    r_3 = (d >> ops) >> ops_b
    assert data_algebra.test_util.equivalent_frames(r_3, expect_2)
    r_4 = d >> ops >> ops_b
    assert data_algebra.test_util.equivalent_frames(r_4, expect_2)


def test_arrow_assoc_with_data_ops():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2]})
    ops = data_algebra.descr(d=d).extend({"x": "x + 5"})
    res_1 = d >> ops
    expect_1 = pd.DataFrame({"x": [6, 7]})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    ops_b = data_algebra.TableDescription(column_names=["x"]).extend({"x": "2 * x - 1"})
    composite = ops >> ops_b
    assert isinstance(composite, OperatorPlatform)
    expect_2 = pd.DataFrame({"x": [11, 13]})
    r_1 = d >> composite
    assert data_algebra.test_util.equivalent_frames(r_1, expect_2)
    r_2 = d >> (ops >> ops_b)
    assert data_algebra.test_util.equivalent_frames(r_2, expect_2)
    r_3 = (d >> ops) >> ops_b
    assert data_algebra.test_util.equivalent_frames(r_3, expect_2)
    r_4 = d >> ops >> ops_b
    assert data_algebra.test_util.equivalent_frames(r_4, expect_2)


def test_arrow_assoc_guard():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = DataOpArrow(data_algebra.descr(d=d).extend({"z": "x + y"}))
    res_1 = d >> ops
    expect_1 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [4, 6]})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    ops_b = DataOpArrow(data_algebra.TableDescription(column_names=["x"]).extend({"x": "2 * x - 1"}))
    with pytest.raises(ValueError):
        ops >> ops_b
    with pytest.raises(ValueError):
        ops_b >> ops
    with pytest.raises(AssertionError):
        d >> ops_b


def test_arrow_assoc_guard_table():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = data_algebra.descr(d=d).extend({"z": "x + y"})
    res_1 = d >> ops
    expect_1 = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [4, 6]})
    assert data_algebra.test_util.equivalent_frames(res_1, expect_1)
    ops_b = data_algebra.TableDescription(column_names=["x"]).extend({"x": "2 * x - 1"})
    with pytest.raises(AssertionError):
        ops >> ops_b
    with pytest.raises(AssertionError):
        ops_b >> ops
    with pytest.raises(AssertionError):
        d >> ops_b
    # the non-associativity we are defending against
    res_wide = ops_b.replace_leaves({"data_frame": ops}).transform(d)  # widens column def
    res_narrow = ops_b.transform(ops.transform(d))  # would want to match this
    assert len(res_wide.columns) != len(res_narrow.columns)
    # allowed
    ops_3 = TableDescription(column_names=['x'], table_name="zz").extend({"y": "x + 1"}) >> ops
    assert isinstance(ops_3, ViewRepresentation)
    res_3 = pd.DataFrame({"x": [1, 2]}) >> ops_3
    expect_3 = pd.DataFrame({"x": [1, 2], "y": [2, 3], "z": [3, 5]})
    assert data_algebra.test_util.equivalent_frames(res_3, expect_3)
    res_3_a = pd.DataFrame({"x": [1, 2]}) >> TableDescription(column_names=['x'], table_name="zz").extend({"y": "x + 1"}) >> ops
    assert data_algebra.test_util.equivalent_frames(res_3_a, expect_3)
    res_3_b = (pd.DataFrame({"x": [1, 2]}) >> TableDescription(column_names=['x'], table_name="zz").extend({"y": "x + 1"})) >> ops
    assert data_algebra.test_util.equivalent_frames(res_3_b, expect_3)
    res_3_c = pd.DataFrame({"x": [1, 2]}) >> (TableDescription(column_names=['x'], table_name="zz").extend({"y": "x + 1"}) >> ops)
    assert data_algebra.test_util.equivalent_frames(res_3_c, expect_3)


def test_arrow_table_narrows():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = data_algebra.TableDescription(column_names=["x"])
    res = ops.transform(d)
    expect = pd.DataFrame({"x": [1, 2]})
    assert data_algebra.test_util.equivalent_frames(res, expect)
    with pytest.raises(AssertionError):
        d >> ops
    res2 = d.loc[:, ["x"]] >> ops
    assert data_algebra.test_util.equivalent_frames(res2, expect)


def test_arrow_identity():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2]})
    ops = data_algebra.descr(d=d).extend({"y": "x + 1"})
    dl = data_algebra.TableDescription(table_name="d", column_names=["x"])
    ops_2 = dl >> ops
    assert ops_2 == ops
    dr = data_algebra.TableDescription(table_name="d2", column_names=["x", "y"])
    ops_3 = ops >> dr
    assert ops_3 == ops
    assert (dl >> dl) == dl
    assert (dr >> dr) == dr
    assert dl != dr
    res = d >> dl
    assert data_algebra.test_util.equivalent_frames(d, res)
