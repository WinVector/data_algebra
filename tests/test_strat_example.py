# https://github.com/WinVector/pyvtreat/blob/master/Examples/StratifiedCrossPlan/StratifiedCrossPlan.ipynb

import data_algebra

import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *


def test_strat_example():
    prepared_stratified = data_algebra.pd.DataFrame(
        {"y": [1, 0, 0, 1, 0, 0], "g": [0, 0, 0, 1, 1, 1], "x": [1, 2, 3, 4, 5, 6]}
    )

    ops = describe_table(prepared_stratified).project(
        {"sum": "y.sum()", "mean": "y.mean()", "size": "_size()",}, group_by=["g"]
    )

    res = ops.transform(prepared_stratified)

    expect = data_algebra.pd.DataFrame(
        {
            "g": [0, 1],
            "sum": [1, 1],
            "mean": [0.3333333333333333, 0.3333333333333333],
            "size": [3, 3],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res, expect)
