
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.test_util
import data_algebra.SQLite
import data_algebra.MySQL
import data_algebra.SparkSQL
import data_algebra.PostgreSQL
import data_algebra.BigQuery


def test_any_project_value():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'x': [.1, .1, .3, .4],
        'g': ['a', 'a', 'b', 'ccc'],
    })
    ops = (
        descr(d=d)
            .project(
                {'new_column': 'x.any_value()'},
                group_by=['g'])
            .order_rows(['g'])
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'g': ['a', 'b', 'ccc'],
        'new_column': [0.1, 0.3, 0.4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        valid_for_empty=False,
        )
    bq_sql = data_algebra.BigQuery.BigQueryModel().to_sql(ops)
    assert 'ANY_VALUE(`x`)' in bq_sql


def test_any_project_logical():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'a': [False, False],
        'b': [False, True],
        'c': [True, False],
        'd': [True, True],
    })
    ops = (
        descr(d=d)
            .project(
                {
                    'any_a': 'a.any()',
                    'all_a': 'a.all()',
                    'any_b': 'b.any()',
                    'all_b': 'b.all()',
                    'any_c': 'c.any()',
                    'all_c': 'c.all()',
                    'any_d': 'd.any()',
                    'all_d': 'd.all()',
                },
                group_by=[])
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'any_a': [False],
        'all_a': [False],
        'any_b': [True],
        'all_b': [False],
        'any_c': [True],
        'all_c': [False],
        'any_d': [True],
        'all_d': [True],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        valid_for_empty = False)


def test_any_project_scalar_produced():
    # On Python 3.10.8
    # Pandas 1.5.1
    # on 2022-11-20
    # the following creates scalars during the project, needing the 
    # promote_scalar_to_array() treatment
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({'a': [False, False]})
    ops = (
        descr(d=d)
            .project({'any_a': 'a.any()'})
        )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'any_a': [False]
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
