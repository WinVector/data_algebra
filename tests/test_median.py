
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import pytest


def test_median_one():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
        'x': [1., 1., 2., 30., 30.],
    })
    ops = descr(d=d).project({'mx': 'x.median()'})
    expect = pd.DataFrame({'mx': [2.0]})
    with pytest.warns(UserWarning):  # TODO: remove this grab of the warnings, not what we want
        data_algebra.test_util.check_transform(
            ops=ops,
            data=d,
            expect=expect,
            models_to_skip={},
            valid_for_empty=False,
            empty_produces_empty=False,
        )
