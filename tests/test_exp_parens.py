import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import data, descr, describe_table, ex


def test_exp_parens():
    d_local = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3]})
    ops = data(d_local).extend({"p": "(x+1).exp()"})
    ops_str = str(ops)
    assert "((" not in ops_str
