import pytest

import data_algebra
import data_algebra.util
from data_algebra.data_ops import *

import lark.exceptions


def test_parse():
    q = 4

    ops = TableDescription("d", ["x", "y"]).extend({"z": f"1/{q} + x"})

    # can not see outter environment
    with pytest.raises(NameError):
        TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    TableDescription("d", ["x", "y"]).extend({"z": "x.is_null()", "q": "x.is_bad()"})


def test_parse_2():
    ops = (
        TableDescription("d", ["x", "y", "s"])
        .extend({"z": "x.sin()", "q": "x.remainder(y)"})
        .extend({"row_number": "_row_number()"}, partition_by=["y"], order_by=["s"])
    )


def test_parse_4():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x | y"})


def test_parse_4b():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x | y | s"})


def test_parse_4c():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x + y + s"})


def test_parse_4d():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x * y * s"})


def test_parse_4e():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x * y / s"})


def test_parse_4f():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x %+% s"})


def test_parse_4f2():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x.concat(s)"})


def test_parse_4g():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x %+% y %+% s"})


def test_parse_4g2():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x.coalesce(s)"})


def test_parse_4h():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x %?% y %?% s"})


def test_parse_4i():
    ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x % y"})


def test_parse_5():
    with pytest.raises(lark.exceptions.UnexpectedToken):
        ops = TableDescription("d", ["x", "y", "s"]).extend({"z": "x || y"})


def test_parse_6():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend(
        {"z": "(u.sin() + w**2) / x + y / v"}
    )
    recovered = ops.ops["z"]
    assert str(recovered) == "((u.sin() + (w ** 2)) / x) + (y / v)"


def test_parse_6b():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u.sin()"})
    recovered = ops.ops["z"]
    assert str(recovered) == "u.sin()"


def test_parse_7():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u & v & w"})
    recovered = ops.ops["z"]
    assert str(recovered) == "(u & v) & w"


def test_parse_8():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u + v + w"})
    recovered = ops.ops["z"]
    assert str(recovered) == "(u + v) + w"


def test_parse_9():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u + v / w"})
    recovered = ops.ops["z"]
    assert str(recovered) == "u + (v / w)"


def test_parse_10():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u - v + w"})
    recovered = ops.ops["z"]
    assert str(recovered) == "(u - v) + w"


def test_parse_11():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u /v"})
    recovered = ops.ops["z"]
    assert str(recovered) == "u / v"


def test_parse_12():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u * (v + w)"})
    recovered = ops.ops["z"]
    assert str(recovered) == "u * (v + w)"


def test_parse_13():
    ops = TableDescription("d", ["u", "v", "w", "x", "y"]).extend({"z": "u * v + w"})
    recovered = ops.ops["z"]
    assert str(recovered) == "(u * v) + w"


def test_parse_14():
    ops = TableDescription("d", ["x"]).extend({"z": "(1).sum()"})
    recovered = ops.ops["z"]
    assert str(recovered) == "(1).sum()"
