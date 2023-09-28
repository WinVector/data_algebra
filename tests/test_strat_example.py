# https://github.com/WinVector/pyvtreat/blob/master/Examples/StratifiedCrossPlan/StratifiedCrossPlan.ipynb

import data_algebra

import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex


def test_strat_example():
    prepared_stratified = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "y": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "g": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    ops = describe_table(prepared_stratified).project(
        {"sum": "y.sum()", "mean": "y.mean()", "size": "_size()",}, group_by=["g"]
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "g": [0, 1],
            "sum": [1, 1],
            "mean": [0.3333333333333333, 0.3333333333333333],
            "size": [3, 3],
        }
    )

    data_algebra.test_util.check_transform(
        ops=ops, data=prepared_stratified, expect=expect
    )


def test_strat_example_size():
    prepared_stratified = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"y": [1, 0, 0, 1, 0, 0], "g": [0, 0, 0, 1, 1, 1], "x": [1, 2, 3, 4, 5, 6]}
    )

    ops = describe_table(prepared_stratified).project(
        {"size": "_size()",}, group_by=["g"]
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"g": [0, 1], "size": [3, 3],}
    )

    data_algebra.test_util.check_transform(
        ops=ops, data=prepared_stratified, expect=expect
    )
