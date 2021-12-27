
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.util


def test_uniform_1():
    # some example data
    d = data_algebra.default_data_model.pd.DataFrame(
        {
            "ID": [1, 2, 3],
        }
    )
    # ops = (
    #     descr(d=d).extend({'r': '_uniform()'})
    # )