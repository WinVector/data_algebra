
# import packages
import string
import numpy
import numpy.random

import data_algebra
from data_algebra.data_ops import *
from data_algebra.cdata import *
import data_algebra.test_util
import data_algebra.util
import data_algebra.SQLite


def test_compare_data_frames():
    # from https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/comparing_two_dataframes.ipynb
    # build synthetic example data

    # seed the pseudo-random generator for repeatability
    numpy.random.seed(1999)

    # choose our simulated number of observations
    n_obs = 100
    symbols = list(string.ascii_lowercase)

    d1 = data_algebra.default_data_model.pd.DataFrame({
        'group': numpy.random.choice(symbols, size=n_obs, replace=True),
    })

    d2 = data_algebra.default_data_model.pd.DataFrame({
        'group': numpy.random.choice(symbols, size=n_obs, replace=True),
    })

    # which columns we consider to be row keys
    # can be more than one column
    grouping_columns = ['group']

    summary_ops = (
        descr(d1=d1)
            .project(
                {'d1_count': '(1).sum()'},
                group_by=grouping_columns)
            .natural_join(
                b=descr(d2=d2)
                    .project(
                        {'d2_count': '(1).sum()'},
                        group_by=grouping_columns),
                by=grouping_columns,
                jointype='full')
            .extend({
                'd1_count': 'd1_count.coalesce(0)',
                'd2_count': 'd2_count.coalesce(0)',
                })
    )

    summary_table = summary_ops.eval({'d1': d1, 'd2': d2})

    res = ex(
        data(summary_table)
            .select_rows('(d1_count <= 0) | (d2_count <= 0)')
            .order_rows(grouping_columns)
    )

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['u', 'w'],
        'd1_count': [0.0, 4.0],
        'd2_count': [3.0, 0.0],
        })

    assert data_algebra.test_util.equivalent_frames(res, expect)

    # sqlite_handle = data_algebra.SQLite.example_handle()
    # print(sqlite_handle.to_sql(summary_ops))
    # sqlite_handle.close()

    ops2 = (
        summary_ops
            .select_rows('(d1_count <= 0) | (d2_count <= 0)')
            .order_rows(grouping_columns)
    )
    data_algebra.test_util.check_transform(
        ops=ops2, data={"d1": d1, "d2": d2}, expect=expect
    )
