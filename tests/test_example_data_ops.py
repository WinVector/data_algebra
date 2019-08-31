from data_algebra.data_ops import *
from data_algebra.data_pipe import *
from data_algebra.util import od
import data_algebra.env
import data_algebra.yaml
import data_algebra.pipe


# test representation paths
# TODO: more tests for more ops patterns



def test_example_data_ops_extend():
    data_algebra.yaml.fix_ordered_dict_yaml_rep()
    ops = []
    q = 4
    x = 2
    var_name = 'y'
    with data_algebra.env.Env(locals()) as env:
        ops = ops + [
            TableDescription('d', ['x', 'y']).
                extend({'z': '_.x + _[var_name]/q + _get("x") + x'})
        ]

        ops = ops + [
            TableDescription('d', ['x', 'y']) >>
                Extend({'z': '1/q + x'}) >>
                SelectColumns(['y', 'z'])
        ]

        ops = ops + [
            TableDescription('d', ['x', 'y']).
                extend(od(z='1/q + _.x/_[var_name]', f=1, g='"2"', h=True))
        ]

        ops = ops + [data_algebra.pipe.build_pipeline(
            TableDescription('d', ['x', 'y']),
            Extend(od(z='1/_.y + 1/q', x='x+1'))
        )]

        ops = ops + [
            TableDescription('d', ['q', 'y']).
                extend({'z': '1/q + y'})
        ]

        ops = ops + [
            TableDescription('d', ['q', 'y']).
                extend({'z': 'q/_get("q") + y + _.q'})
        ]

    for o in ops:
        data_algebra.yaml.check_op_round_trip(o)
