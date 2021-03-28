import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.env
import data_algebra.yaml


# test representation paths
# TODO: more tests for more ops patterns


def test_example_data_ops_extend():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    ops = []
    q = 4
    x = 2
    var_name = "y"
    with data_algebra.env.Env(locals()) as env:
        ops = ops + [TableDescription("d", ["q", "y"]).extend({"z": "1/q + y"})]

    for o in ops:
        data_algebra.test_util.check_op_round_trip(o)
