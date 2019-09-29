
from data_algebra.data_ops import *


def test_select_stacking():
    ops1 = TableDescription(
            "d", ["a", "b", "c"]
        ).select_columns(
            ['a', 'b']
         ).select_columns(
            ['a', 'b']
         )
    ops1_str = format(ops1)
    assert ops1_str.count("select_columns")==1

    ops2 = TableDescription(
            "d", ["a", "b", "c"]
        ).select_columns(
            ['a', 'b']
         ).select_columns(
            ['a']
         )
    ops2_str = format(ops2)
    assert ops2_str.count("select_columns") == 1

