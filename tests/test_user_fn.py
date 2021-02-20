import numpy

import pytest

import data_algebra
from data_algebra.data_ops import *
import data_algebra.test_util
from data_algebra.expr_rep import *


def test_term_combine():
    t1 = ColumnReference(view=None, column_name="x")
    t1 + 1  # allowed
    t2 = FnCall(numpy.sin, [t1])
    with pytest.raises(TypeError):
        t2 + 1  # not allowed


def test_user_fn_1():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4], "g": [1, 1, 2, 2],}
    )

    def sumsq(x):
        return numpy.sum(x * x)

    ops_g = describe_table(d, table_name="d").project(
        {"x": user_fn(sumsq, "x")}, group_by=["g"]
    )
    res_g = ops_g.transform(d)
    expect_g = data_algebra.default_data_model.pd.DataFrame(
        {"g": [1, 2], "x": [5, 25],}
    )
    assert data_algebra.test_util.equivalent_frames(res_g, expect_g)


def test_extend_fn_exception():
    d = data_algebra.default_data_model.pd.DataFrame(
        {"x": [1, 2, 3, 4], "g": [1, 1, 2, 2],}
    )

    def normalize(x):
        x = x - numpy.mean(x)
        x = x / numpy.std(x)
        return x

    ops = describe_table(d, table_name="d").extend({"x": user_fn(normalize, "x")})
    res = ops.transform(d)
    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "x": [
                -1.3416407864998738,
                -0.4472135954999579,
                0.4472135954999579,
                1.3416407864998738,
            ],
            "g": [1, 1, 2, 2],
        }
    )
    assert data_algebra.test_util.equivalent_frames(res, expect)

    with pytest.raises(ValueError):
        describe_table(d, table_name="d").extend(
            {"x": user_fn(normalize, "x")}, partition_by=["g"]
        )


def test_user_fn_2():
    d_original = data_algebra.default_data_model.pd.DataFrame(
        {
            "const_col": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
            "noise_0": [
                "nl_2",
                "nl_0",
                "nl_2",
                "nl_0",
                "nl_4",
                "nl_0",
                "nl_0",
                "nl_1",
                "nl_1",
                "nl_0",
            ],
            "noise_1": [
                "nl_1",
                "nl_3",
                "nl_2",
                "nl_3",
                "nl_4",
                "nl_1",
                "nl_1",
                "nl_2",
                "nl_3",
                "nl_2",
            ],
            "noise_2": [
                "nl_2",
                "nl_0",
                "nl_4",
                "nl_3",
                "nl_4",
                "nl_4",
                "nl_3",
                "nl_0",
                "nl_3",
                "nl_0",
            ],
            "noise_3": [
                "nl_2",
                "nl_1",
                "nl_4",
                "nl_4",
                "nl_0",
                "nl_0",
                "nl_3",
                "nl_4",
                "nl_2",
                "nl_2",
            ],
            "noise_4": [
                "nl_3",
                "nl_0",
                "nl_4",
                "nl_1",
                "nl_0",
                "nl_3",
                "nl_2",
                "nl_0",
                "nl_0",
                "nl_4",
            ],
        }
    )
    # y_example = [-0.03693426, -1.90864297, -1.27181382, 0.73296212, 1.88960083,
    #              1.12960488, 0.99430907, -0.24343111, 0.73849456, 1.35531344]
    d_coded = data_algebra.default_data_model.pd.DataFrame(
        {
            "const_col": [
                0.5083722782350298,
                0.5966033407000854,
                0.5083722782350298,
                -0.16265063982529346,
                -0.16265063982529346,
                0.5083722782350298,
                -0.16265063982529346,
                0.5966033407000854,
                -0.16265063982529346,
                0.5966033407000854,
            ],
            "noise_0": [
                0.5083722782350298,
                0.9098928944797735,
                0.5083722782350298,
                0.14980545238880305,
                -0.16265063982529346,
                0.3036766134036335,
                0.14980545238880305,
                0.5966033407000854,
                -0.16265063982529346,
                0.9098928944797735,
            ],
            "noise_1": [
                0.5083722782350298,
                0.6983118645635149,
                0.5431479206566401,
                -0.16265063982529346,
                -0.16265063982529346,
                0.5083722782350298,
                0.35565962159606224,
                0.5966033407000854,
                -0.16265063982529346,
                0.5966033407000854,
            ],
            "noise_2": [
                0.5083722782350298,
                0.5966033407000854,
                0.5083722782350298,
                -0.16265063982529346,
                -0.0957250265673181,
                0.5083722782350298,
                -0.16265063982529346,
                0.5966033407000854,
                -0.16265063982529346,
                0.5966033407000854,
            ],
            "noise_3": [
                0.9020705141488081,
                0.5966033407000854,
                0.31566028372301,
                -0.5976098985669023,
                -0.16265063982529346,
                0.5083722782350298,
                -0.16265063982529346,
                -0.036514730121240235,
                0.4381627102724482,
                0.416892187791968,
            ],
            "noise_4": [
                0.5083722782350298,
                1.1210971923510848,
                0.5083722782350298,
                -0.16265063982529346,
                -0.8303896031810106,
                0.5083722782350298,
                -0.16265063982529346,
                1.1210971923510848,
                -0.8303896031810106,
                0.5966033407000854,
            ],
        }
    )

    # https://github.com/WinVector/data_algebra/blob/master/Examples/cdata/ranking_pivot_example.md
    class Container:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return self.value.__repr__()

        def __str__(self):
            return self.value.__repr__()

        def update(self, other):
            if not isinstance(other, Container):
                return self
            return Container(sorted([vi for vi in set(self.value).union(other.value)]))

    def sorted_concat(vals):
        return Container(sorted([vi for vi in set(vals)]))

    def combine_containers(lcv, rcv):
        return [lft.update(rgt) for lft, rgt in zip(lcv, rcv)]

    nrow = d_original.shape[0]
    ncol = d_original.shape[1]
    pairs = data_algebra.default_data_model.pd.DataFrame(
        {"idx": range(nrow), "complement": [Container([])] * nrow}
    )
    for j in range(ncol):
        dj = data_algebra.default_data_model.pd.DataFrame(
            {
                "orig": d_original.iloc[:, j],
                "coded": d_coded.iloc[:, j],
                "idx": range(nrow),
            }
        )
        ops_collect = (
            describe_table(dj, table_name="dj")
            .rename_columns({"coded_left": "coded", "idx_left": "idx"})
            .natural_join(
                b=describe_table(dj, table_name="dj"), jointype="full", by=["orig"]
            )
            .select_rows("(coded_left - coded).abs() > 1.0e-5")
            .project(
                {"complement": user_fn(sorted_concat, "idx")}, group_by=["idx_left"]
            )
            .rename_columns({"idx": "idx_left"})
        )
        pairsj = ops_collect.transform(dj)
        ops_join = (
            describe_table(pairs, table_name="pairs")
            .natural_join(
                b=describe_table(pairsj, table_name="pairsj").rename_columns(
                    {"c_right": "complement"}
                ),
                jointype="left",
                by=["idx"],
            )
            .extend(
                {"complement": user_fn(combine_containers, ["complement", "c_right"])}
            )
            .drop_columns("c_right")
        )
        pairs = ops_join.eval({"pairs": pairs, "pairsj": pairsj})

    expect_pairs = data_algebra.default_data_model.pd.DataFrame(
        {
            "idx": range(10),
            "complement": [
                Container([1, 3, 4, 6, 7, 8, 9]),
                Container([0, 2, 3, 4, 5, 6, 8]),
                Container([1, 3, 4, 6, 7, 8, 9]),
                Container([0, 1, 2, 5, 7, 9]),
                Container([0, 1, 2, 5, 7, 9]),
                Container([1, 3, 4, 6, 7, 8, 9]),
                Container([0, 1, 2, 5, 7, 9]),
                Container([0, 2, 3, 4, 5, 6, 8]),
                Container([0, 1, 2, 5, 7, 9]),
                Container([0, 2, 3, 4, 5, 6, 8]),
            ],
        }
    )

    assert pairs.shape == expect_pairs.shape
    assert all(
        [
            pairs["complement"][i].value == expect_pairs["complement"][i].value
            for i in range(expect_pairs.shape[0])
        ]
    )
