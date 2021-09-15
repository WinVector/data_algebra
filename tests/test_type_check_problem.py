
import data_algebra
from data_algebra.data_ops import *
import data_algebra.util
import data_algebra.test_util


def test_type_check_problem_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2]
    })
    assert not isinstance(d['x'][0], data_algebra.default_data_model.pd.core.series.Series)
    d2 = data_algebra.default_data_model.pd.concat([d, d])
    # pandas concat mucks columns into series (reset_index() fixes that)
    # check that we see that problem
    assert isinstance(d2['x'][0], data_algebra.default_data_model.pd.core.series.Series)
    # reset index can clear the problem
    d3 = d2.reset_index(drop=True, inplace=False)
    assert not isinstance(d3['x'][0], data_algebra.default_data_model.pd.core.series.Series)
    # see what type inspection we get
    td2 = describe_table(d2, table_name='d2')
    td3 = describe_table(d2, table_name='d3')
    assert td3.column_types['x'] == int
    assert td2.column_types == td3.column_types  # don't pick up Series types!
