
import data_algebra

import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.test_util import formats_to_self
import data_algebra.util


def test_expr_parse():
    # check some differences in back to Python versus sending to Pandas
    d = data_algebra.pd.DataFrame({"a": [True, False], "b": [1, 2], "c": [3, 4]})

    ops0 = TableDescription("d", ["a", "b", "c"]).extend({"d": "a + 1"})

    assert formats_to_self(ops0)

    res0 = ops0.transform(d)
    expect0 = data_algebra.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [2, 1],}
    )
    assert data_algebra.test_util.equivalent_frames(res0, expect0)

    ops1 = TableDescription("d", ["a", "b", "c"]).extend({"d": "a.if_else(1, c)"})

    assert formats_to_self(ops1)

    res1 = ops1.transform(d)
    expect1 = data_algebra.pd.DataFrame(
        {"a": [True, False], "b": [1, 2], "c": [3, 4], "d": [1, 4],}
    )
    assert data_algebra.test_util.equivalent_frames(res1, expect1)

    # # TODO: implement and test
    # ops2 = TableDescription('d', ['a', 'b', 'c']). \
    #     extend({'d': 'b.fmax(1.5)'})
    #
    # assert formats_to_self(ops2)
    #
    # res2 = ops2.transform(d)
    # expect2 = data_algebra.pd.DataFrame({
    #     'a': [True, False],
    #     'b': [1, 2],
    #     'c': [3, 4],
    #     'd': [1.5, 2],
    #     })
