import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.env


# test representation paths
# TODO: more tests for more ops patterns


def test_example_data_ops_extend():
    ops = []
    q = 4
    x = 2
    var_name = "y"
    with data_algebra.env.Env(locals()) as env:
        ops = ops + [TableDescription("d", ["q", "y"]).extend({"z": "1/q + y"})]
