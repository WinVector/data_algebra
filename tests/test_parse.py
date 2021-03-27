import pytest

import data_algebra.util
from data_algebra.data_ops import *


def test_parse():
    q = 4

    # can see environments we are told about
    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    # can not see outter environment
    with pytest.raises(NameError):
        TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    TableDescription("d", ["x", "y"]).extend({"z": "x.is_null()", "q": "x.is_bad()"})


def test_parse_2():
    ops = TableDescription("d", ["x", "y", "s"]). \
        extend({
            "z": "x.sin()",
            "q": "x.remainder(y)"}). \
        extend({
            "row_number": "_row_number()"},
            partition_by=["y"],
            order_by=['s'])


