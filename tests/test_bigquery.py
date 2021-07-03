
import os
import datetime
import sqlite3

from google.cloud import bigquery

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.BigQuery
import data_algebra.SQLite

import pytest


# running bigquery tests depends on environment variable
# example:
#  export GOOGLE_APPLICATION_CREDENTIALS="/Users/johnmount/big_query/big_query_jm.json"
# to clear:
#  unset GOOGLE_APPLICATION_CREDENTIALS
@pytest.fixture(scope='module')
def get_bq_handle():
    # set up big query client
    bq_client = None
    try:
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/johnmount/big_query/big_query_jm.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  # trigger key error if not present
        bq_client = bigquery.Client()
    except KeyError:
        pass
    bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(bq_client)
    data_catalog = 'data-algebra-test'
    data_schema = 'test_1'

    # set up sqlite client
    conn_sqlite = sqlite3.connect(":memory:")
    db_model_sqlite = data_algebra.SQLite.SQLiteModel()
    db_model_sqlite.prepare_connection(conn_sqlite)
    db_handle_sqlite = db_model_sqlite.db_handle(conn_sqlite)

    db_handles = [bq_handle, db_handle_sqlite]

    yield {
        'bq_handle': bq_handle,
        'db_handles': db_handles,
        'data_catalog': data_catalog,
        'data_schema': data_schema,
    }
    # back from yield, clean up
    bq_handle.close()
    db_handle_sqlite.close()


def test_bigquery_1(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    # NOTE: sqllite doesn't allow median as a window function,
    # meaning it can not run this query.
    ops = describe_table(d, table_name=table_name). \
        extend(
            {'med_val': 'median(val)'},
            partition_by=['group']). \
        project(
            {'med_val': 'mean(med_val)'},  # pseudo-aggregator
            group_by=['group'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'b'],
        'med_val': [1.5, 3.5],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_bigquery_2(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    # this is the pattern BigQuery needs to compute
    # median, window function then a pseudo-aggregation
    # refs on BigQuery window fn horseshit:
    #  https://iamhectorotero.github.io/median-and-group-by/
    #  https://chartio.com/resources/tutorials/how-countdistinct-field-works-in-google-bigquery/
    ops = describe_table(d, table_name=table_name). \
        extend({
            'med_1': 'v1.median()',  # median is only a window fn in Big Query
            'med_2': 'v2.median()',  # median is only a window fn in Big Query
            },
            partition_by=['group']). \
        project({
            'med_1': 'med_1.mean()',  # pseudo aggregator
            'med_2': 'med_2.mean()',  # pseudo aggregator
            'mean_1': 'v1.mean()',
            'mean_2': 'v2.mean()',
            'nu_1': 'v1.nunique()',
            'nu_2': 'v2.nunique()',
            },
            group_by=['group'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'b'],
        'med_1': [2, 0],
        'med_2': [2.0, 4.5],
        'mean_1': [1.66666666667, 0.0],
        'mean_2': [2.0, 4.5],
        'nu_1': [2, 1],
        'nu_2': [3, 2],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_bigquery_insert_raise(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5]})
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_dx'
    bq_handle.drop_table(table_name)
    bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
    with pytest.raises(ValueError):
        bq_handle.insert_table(d, table_name=table_name, allow_overwrite=False)
    bq_handle.drop_table(table_name)


def test_bigquery_date_1(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 0],
        'v2': [1, 2, 3, 4, 5],
        'dt': data_algebra.default_data_model.pd.to_datetime([1490195805, 1490195815, 1490295805, 1490196805, 1490195835], unit='s')
    })
    d['dt_str'] = d.dt.astype(str)

    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'date': bq_handle.fns.datetime_to_date('dt'),
            'date_str': bq_handle.fns.trimstr('dt_str', start=0, stop=10),
         }) . \
        extend({
            'mean_v1': 'v1.mean()',
            'count': '_size()',
            },
            partition_by='group') . \
        drop_columns(['dt'])  # date will comback with UTC and other alterations

    expect = d.copy()
    expect['date'] = expect.dt.dt.date.copy()
    expect['date_str'] = expect.dt_str.str.slice(start=0, stop=10)
    expect['mean_v1'] = [5/3, 5/3, 5/3, 0, 0]
    expect['count'] = [3, 3, 3, 2, 2]
    del expect['dt']

    # res_pandas = ops.transform(d)
    # bq_handle.insert_table(d, table_name=table_name, allow_overwrite=True)
    # res_bq = bq_handle.read_query(ops)
    # assert data_algebra.test_util.equivalent_frames(expect, res_pandas)
    # assert data_algebra.test_util.equivalent_frames(expect, res_bq)

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_big_query_and(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 2],
        'v2': [1, 2, 3, 4, 5],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    # build a description that looks like the BigQuery db handle built it.
    ops = describe_table(d, table_name=table_name) .\
        select_rows("(group == 'a') & (v1 == 2)")

    # see & gets translated to AND
    sql = bq_handle.to_sql(ops)
    assert sql.find('&') < 0
    assert sql.find('AND') > 0

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a'],
        'v1': [2, 2],
        'v2': [2, 3],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_big_query_notor(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b'],
        'v1': [1, 2, 2, 0, 2],
        'v2': [1, 2, 3, 4, 5],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    # build a description that looks like the BigQuery db handle built it.
    ops = describe_table(d, table_name=table_name) .\
        select_rows("not ((group == 'a') or (v1 == 2))")

    # see & gets translated to AND
    sql = bq_handle.to_sql(ops)
    assert sql.find('|') < 0
    assert sql.find('OR') > 0

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['b'],
        'v1': [0],
        'v2': [4],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_TRIMSTR(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
         'nx': bq_handle.fns.trimstr('x', start=0, stop=5)
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['0123456', 'abcdefghijk'],
        'y': ['012345', 'abcdefghij'],
        'nx': ['01234', 'abcde'],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_AS_INT64(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
         'nx': bq_handle.fns.as_int64('x')
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['0123456', '66'],
        'y': ['012345', '77'],
        'nx': [123456, 66]
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_DATE(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': data_algebra.default_data_model.pd.to_datetime([1490196805, 1490195835], unit='s'),
        'y': ['012345', '77'],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
         'nx': bq_handle.fns.datetime_to_date('x')
        }) .\
        extend({
         'nxs': bq_handle.fns.as_str('nx')
        }) .\
        select_columns(['nxs'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'nxs': ['2017-03-22', '2017-03-22']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_COALESCE_0(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, None, 3]
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
         'nx': bq_handle.fns.coalesce_0('x')
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, None, 3],
        'nx': [1, 0, 3]
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_PARSE_DATE(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['2001-01-01', '2020-04-02']
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
         'nx': bq_handle.fns.parse_date('x')
        })
    res = ops.transform(d)
    assert isinstance(res.nx[0], datetime.date)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['2001-01-01', '2020-04-02']
    })
    expect['nx'] = data_algebra.default_data_model.pd.to_datetime(d.x, format="%Y-%m-%d")
    assert data_algebra.test_util.equivalent_frames(res, expect)

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )



def test_DATE_PARTS(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['2001-01-01', '2020-04-02'],
        't': ['2001-01-01 01:33:22', '2020-04-02 13:11:10'],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'nx': bq_handle.fns.parse_date('x', format="%Y-%m-%d"),
            'nt': bq_handle.fns.parse_datetime('t', format="%Y-%m-%d %H:%M:%S"),
            'nd': bq_handle.fns.parse_datetime('x', format="%Y-%m-%d"),
        }) .\
        extend({
            'date2': bq_handle.fns.datetime_to_date('nt'),
            'day_of_week': bq_handle.fns.dayofweek('nx'),
            'day_of_year': bq_handle.fns.dayofyear('nx'),
            'month': bq_handle.fns.month('nx'),
            'day_of_month': bq_handle.fns.dayofmonth('nx'),
            'quarter': bq_handle.fns.quarter('nx'),
            'year': bq_handle.fns.year('nx'),
            'diff': bq_handle.fns.timestamp_diff('nt', 'nd'),
            'sdt': bq_handle.fns.format_datetime('nt', format="%Y-%m-%d %H:%M:%S"),
            'sd': bq_handle.fns.format_date('nx', format="%Y-%m-%d"),
            'dd': bq_handle.fns.date_diff('nx', 'nx'),
        })
    res = ops.transform(d)
    assert isinstance(res.nx[0], datetime.date)
    assert isinstance(res.sdt[0], str)
    assert isinstance(res.sd[0], str)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['2001-01-01', '2020-04-02'],
        't': ['2001-01-01 01:33:22', '2020-04-02 13:11:10'],
        'day_of_week': [2, 5],
        'day_of_year': [1, 93],
        'month': [1, 4],
        'day_of_month': [1, 2],
        'quarter': [1, 2],
        'year': [2001, 2020],
        'dd': [0, 0],
    })
    expect['nx'] = data_algebra.default_data_model.pd.to_datetime(expect.x, format="%Y-%m-%d").dt.date.copy()
    expect['nt'] = data_algebra.default_data_model.pd.to_datetime(expect.t, format="%Y-%m-%d %H:%M:%S")
    expect['nd'] = data_algebra.default_data_model.pd.to_datetime(expect.x, format="%Y-%m-%d")
    expect['date2'] = expect.nt.dt.date.copy()
    expect['diff'] = [
            data_algebra.default_data_model.pd.Timedelta(expect['nt'][i] - expect['nd'][i]).total_seconds()
            for i in range(len(expect['nt']))]
    expect['sdt'] = expect.t
    expect['sd'] = expect.x

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_coalesce(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'a': [1, None, None, None, None, 6, 7, None],
        'b': [10, 20, None, None, None, 60, None, None],
        'c': [None, 200, 300, None, 500, 600, 700, None],
        'd': [1000, None, 3000, 4000, None, 6000, None, None],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name)  .\
        extend({'fixed': bq_handle.fns.coalesce(['a','b', 'c', 'd'])})

    expect = data_algebra.default_data_model.pd.DataFrame({
        'a': [1, None, None, None, None, 6, 7, None],
        'b': [10, 20, None, None, None, 60, None, None],
        'c': [None, 200, 300, None, 500, 600, 700, None],
        'd': [1000, None, 3000, 4000, None, 6000, None, None],
        'fixed': [1, 20, 300, 4000, 500, 6, 7, None],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_base_Sunday(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'date_str': ['2021-04-25', '2021-04-27']
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'dt': bq_handle.fns.parse_date('date_str', format="%Y-%m-%d")
        }) .\
        extend({
            's': bq_handle.fns.base_Sunday('dt')
        }) .\
        drop_columns(['dt']) .\
        extend({
            's': bq_handle.fns.format_date('s', format="%Y-%m-%d")
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'date_str': ['2021-04-25', '2021-04-27'],
        's': ['2021-04-25', '2021-04-25']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_bq_concat_rows(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'd': [1, 2]
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({'d': 'd + 1'}) .\
        concat_rows(b=describe_table(d, table_name=table_name))

    expect = data_algebra.default_data_model.pd.DataFrame({
        'd': [2, 3, 1, 2],
        'source_name': ['a', 'a', 'b', 'b']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=[bq_handle],
        check_parse=False,
    )


def test_bq_join_rows(get_bq_handle):
    bq_handle = get_bq_handle['bq_handle']
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d1 = data_algebra.default_data_model.pd.DataFrame({
        'k': ['a', 'b'],
        'd': [1, 2]
    })
    table_name_d1 = f'{data_catalog}.{data_schema}.pytest_temp_d1'
    d2 = data_algebra.default_data_model.pd.DataFrame({
        'k': ['a', 'b'],
        'e': [4, 5]
    })
    table_name_d2 = f'{data_catalog}.{data_schema}.pytest_temp_d2'

    ops = describe_table(d1, table_name=table_name_d1) .\
        extend({'d': 'd + 1'}) .\
        natural_join(b=describe_table(d2, table_name=table_name_d2),
                     by=['k'],
                     jointype='inner')

    expect = data_algebra.default_data_model.pd.DataFrame({
        'k': ['a', 'b'],
        'd': [2, 3],
        'e': [4, 5],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data={table_name_d1: d1, table_name_d2: d2},
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )
