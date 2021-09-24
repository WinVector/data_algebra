import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *

import pytest


def test_eq_types_1():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError):
        describe_table(d, table_name="d").extend({"eq": 'x == "1"'})


def test_eq_types_2():
    d = data_algebra.default_data_model.pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    with pytest.raises(TypeError):
        describe_table(d, table_name="d").extend({"eq": "x == y"})
