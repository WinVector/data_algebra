

import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex

import data_algebra
import data_algebra.util
import data_algebra.test_util
import data_algebra.PostgreSQL
import data_algebra.BigQuery
import data_algebra.MySQL
import data_algebra.SparkSQL

import pytest


def test_mod_fns_one():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [2, 2, 3, 3],}
    )
    ops = (
        descr(d=d)
            .extend({
                'p': 'a % b',
                'q': 'a.mod(b)',
                'r': 'a.remainder(b)',
            })
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [2, 2, 3, 3],
        'p': [1, 0, 0, 1],
        'q': [1, 0, 0, 1],
        'r': [1, 0, 0, 1],
        })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            # TODO: run these down
            str(data_algebra.PostgreSQL.PostgreSQLModel()),  # SQLAlchemy throws a type error on conversion
            str(data_algebra.BigQuery.BigQueryModel()),
            str(data_algebra.MySQL.MySQLModel()),
            str(data_algebra.SparkSQL.SparkSQLModel())
        },
        )


def test_mod_fns_one_edited():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [2, 2, 3, 3],}
    )
    ops = (
        descr(d=d)
            .extend({
                'q': 'a.mod(b)',
                'r': 'a.remainder(b)',
            })
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [2, 2, 3, 3],
        'q': [1, 0, 0, 1],
        'r': [1, 0, 0, 1],
        })
    res_pandas = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(res_pandas, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
        models_to_skip={
            # TODO: run these down
            str(data_algebra.PostgreSQL.PostgreSQLModel()),  # REMAINDER not applicable to given column types
            str(data_algebra.BigQuery.BigQueryModel()),  # fn not named REMAINDER
            str(data_algebra.MySQL.MySQLModel()),  # fn not named REMAINDER
            str(data_algebra.SparkSQL.SparkSQLModel()),  # fn not named REMAINDER
        },
    )


def test_mod_fns_percent_notation():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'row_id': [0, 1, 2, 3],
        'a': [False, False, True, True],
        'b': [False, True, False, True],
        'q': [1, 1, 2, 2],
    })
    ops = (
        descr(d=d)
            .extend({'r': 'row_id % q'})
    )
    res = ops.transform(d)
    expect = pd.DataFrame({
        'row_id': [0, 1, 2, 3],
        'a': [False, False, True, True],
        'b': [False, True, False, True],
        'q': [1, 1, 2, 2],
        'r': [0, 0, 0, 1],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data=d,
        expect=expect,
    )
