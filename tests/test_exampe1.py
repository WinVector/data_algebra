
import sqlite3

import pandas

import pytest

from data_algebra.data_ops import *
import data_algebra.util
import data_algebra.SQLite
import data_algebra.test_util


def test_example1_1():

    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()

    d = pandas.DataFrame({
        'x_s': ['s_03', 's_04', 's_02', 's_01', 's_03', 's_01'],
        'x_n': ['n_13', 'n_48', 'n_77', 'n_29', 'n_91', 'n_93'],
        'y': [1.0312223, -1.3374379, -1.9347144, 1.2772708, -0.1238039, 0.3058670],
    })
    table_desc = db_model.insert_table(conn, d, 'd')

    ops1 = table_desc. \
        extend({
            'y_mean': 'y.mean()'
            },
            partition_by=1). \
        extend({
            'delta': 'y - y_mean'
            }). \
        project({
            'meany': 'delta.mean()',
            },
            group_by=['x_s']
        ). \
        order_rows(['x_s'])
    assert data_algebra.test_util.formats_to_self(ops1)
    res1 = ops1.transform(d)
    expect1 = pandas.DataFrame({
        'x_s': ['s_01', 's_02', 's_03', 's_04'],
        'meany': [0.9218349166666666, -1.8044483833333334, 0.5839752166666667, -1.2071718833333334],
        })
    assert data_algebra.util.equivalent_frames(res1, expect1)

    sql1 = ops1.to_sql(db_model, pretty=True)
    res1db = db_model.read_query(conn, sql1)
    assert data_algebra.util.equivalent_frames(res1db, expect1)

    ops2 = table_desc. \
        extend({
            'y_mean': 'y.mean()'
            },
            partition_by=1). \
        extend({
            'delta': 'y - y_mean'
            }). \
        project({
            'meany': 'delta.mean()',
            },
            group_by=['x_n']
        ). \
        order_rows(['x_n'])
    assert data_algebra.test_util.formats_to_self(ops2)
    res2 = ops2.transform(d)
    expect2 = pandas.DataFrame({
        'x_n': ['n_13', 'n_29', 'n_48', 'n_77', 'n_91', 'n_93'],
        'meany': [1.1614883166666667, 1.4075368166666666, -1.2071718833333334,
                  -1.8044483833333334, 0.006462116666666684, 0.4361330166666667],
        })
    assert data_algebra.util.equivalent_frames(res2, expect2)

    sql2 = ops2.to_sql(db_model, pretty=True)
    res2db = db_model.read_query(conn, sql2)
    assert data_algebra.util.equivalent_frames(res2db, expect2)

    # clean up
    conn.close()


def test_example1_1_early_error():
    d = pandas.DataFrame({
        'x_s': ['s_03', 's_04', 's_02', 's_01', 's_03', 's_01'],
        'x_n': ['n_13', 'n_48', 'n_77', 'n_29', 'n_91', 'n_93'],
        'y': [1.0312223, -1.3374379, -1.9347144, 1.2772708, -0.1238039, 0.3058670],
    })
    table_desc = describe_table(d, table_name='d')

    with pytest.raises(ValueError):
        table_desc. \
            extend({
            'y_mean': 'y.mean() + 1'
            },
            partition_by=1)

def test_example1_1_detect_agg():
    d = pandas.DataFrame({
        'x_s': ['s_03', 's_04', 's_02', 's_01', 's_03', 's_01'],
        'x_n': ['n_13', 'n_48', 'n_77', 'n_29', 'n_91', 'n_93'],
        'y': [1.0312223, -1.3374379, -1.9347144, 1.2772708, -0.1238039, 0.3058670],
    })
    table_desc = describe_table(d, table_name='d')

    ops = table_desc. \
        extend({
        'y_mean': 'y.mean()'
        })
    assert ops.windowed_situation

    ops2 = table_desc. \
        extend({
        'y_mean': 'y+1'
        })
    assert not ops2.windowed_situation
