import pytest

import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex


def test_calc_warnings_errors():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with pytest.raises(ValueError):
        describe_table(d).extend({"x": "x+1", "y": "x+2"})

    describe_table(d).extend({"x": "x+1", "y": "2"})

    with pytest.raises(NameError):
        describe_table(d).extend({"x": "z+1"})

    with pytest.raises(ValueError):
        describe_table(d).extend([("x", 1), ("x", 2)])

    with pytest.raises(ValueError):
        describe_table(d).project({"x": "x.max()", "y": "x.max()"})

    with pytest.raises(NameError):
        describe_table(d).project({"x": "z.max()"})

    with pytest.raises(ValueError):
        describe_table(d).project([("x", "x.max()"), ("x", "x.min()")])

    with pytest.raises(ValueError):
        describe_table(d).project([("x", 1)])

    describe_table(d).project({"x2": "x.max()", "y": "x.min()"})
