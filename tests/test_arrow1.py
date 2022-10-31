import pytest

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
from data_algebra.arrow import *
import data_algebra.util
import data_algebra.arrow


def test_arrow1():
    d = data_algebra.default_data_model.pd.DataFrame(
        {
            "g": ["a", "b", "b", "c", "c", "c"],
            "x": [1, 4, 5, 7, 8, 9],
            "v": [10.0, 40.0, 50.0, 70.0, 80.0, 90.0],
            "i": [True, True, False, False, False, False],
        }
    )

    d

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

    a1 = data_algebra.arrow.DataOpArrow(id_ops_b, strict=True)

    # check identity relns
    ri = a1.cod()
    li = a1.dom()
    li >> a1
    a1 >> ri
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> li >> a1)
    assert data_algebra.test_util.equivalent_frames(d >> a1, d >> a1 >> ri)
    a1.dom() >> a1
    a1.apply_to(a1.dom())

    # print(a1)

    a1.fit(d)

    # print(a1)

    # print(a1.__repr__())

    a1.transform(d)

    cols2_too_small = [c for c in (set(id_ops_b.column_names) - set(["i"]))]
    ordered_ops = TableDescription(
        table_name="d2", column_names=cols2_too_small
    ).extend(
        {"row_number": "_row_number()", "shift_v": "v.shift()",},
        order_by=["x"],
        partition_by=["g"],
    )
    a2 = data_algebra.arrow.DataOpArrow(ordered_ops, strict=True)
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

    

    a2.fit(a1.transform(d))

    

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

    

    a3.fit(a2.transform(a1.transform(d)))

    

    f0 = (a3.apply_to(a2.apply_to(a1))).pipeline.__repr__()
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
    a1.apply_to(a1.dom())


def test_arrow_cod_dom():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 4],})

    td = describe_table(d)

    a = td.extend({"z": "x.mean()"}, partition_by=["y"])

    a1 = DataOpArrow(a)

    a2 = DataOpArrow(a1.cod_as_table().extend({"ratio": "y / x"}))

    assert a1.cod_as_table() == a2.dom_as_table()
    assert a1.cod() == a2.dom()


def test_arrow_compose_2():
    b1 = DataOpArrow(
        TableDescription(column_names=["x", "y"], table_name=None).extend({"y": "x+1"})
    )
    b2 = DataOpArrow(
        TableDescription(column_names=["x", "y"], table_name=None).extend({"y": 9})
    )
    ops = b1 >> b2
    assert isinstance(ops.pipeline.sources[0], TableDescription)
