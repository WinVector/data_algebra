
import pytest

import data_algebra.util
from data_algebra.data_ops import *

def test_parse():
    q = 4
    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    with pytest.raises(NameError):
        TableDescription("d", ["x", "y"]).extend({"z": "1/q + x"})

    TableDescription("d", ["x", "y"]).extend({
        "z": "x.is_null()",
        "q": "x.is_bad()"
    })
