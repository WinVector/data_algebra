
import os
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


def test_ideom_extend_one_count(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({
            'one': 1
        }) .\
        project({
            'count': 'one.sum()'
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'count': [4]
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_extend_special_count(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        project({
            'count': '_count()'
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'count': [4]
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


# previously forbidden
def test_ideom_forbidden_extend_test_trinary(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'select': '(val > 2.5).if_else("high", "low")'
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
        'select': ['low', 'low', 'high', 'high']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_extend_test_trinary(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
    })
    table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'select': '(val > 2.5)'
        }) .\
        extend({
            'select': 'select.if_else("high", "low")'
        })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1, 2, 3, 4],
        'select': ['low', 'low', 'high', 'high']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_simulate_cross_join(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']


    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    e = data_algebra.default_data_model.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'

    ops = describe_table(d, table_name=table_name_d) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'one': 1
        }) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e) . \
                extend({  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    'one': 1
                }),
            by=['one'],
            jointype='left'
        ) .\
        drop_columns(['one'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data={table_name_d: d, table_name_e: e},
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_simulate_cross_join_select(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    e = data_algebra.default_data_model.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'

    ops = describe_table(d, table_name=table_name_d) .\
        extend({ # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
            'one': 1
        }) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e) . \
                extend({  # {'select': '(val > 2.5).if_else("high", "low")' } # doesn't work in Pandas
                    'one': 1
                }),
            by=['one'],
            jointype='left'
        ) .\
        select_columns(['x', 'y'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data={table_name_d: d, table_name_e: e},
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_cross_join(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 2, 3, 4],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    e = data_algebra.default_data_model.pd.DataFrame({
        'y': ['a', 'b', 'c'],
    })
    table_name_e = f'{data_catalog}.{data_schema}.pytest_temp_e'

    ops = describe_table(d, table_name=table_name_d) .\
        natural_join(
            b=describe_table(e, table_name=table_name_e),
            by=[],
            jointype='cross'
        )

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'y': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data={table_name_d: d, table_name_e: e},
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


# Note: switching from _row_number to _count
def test_ideom_row_number(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'i': [1, 3, 2, 4, 5],
        'g': [1, 2, 2, 1, 1],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name_d) .\
        extend({
            'one': 1
            }) .\
        extend({
            'n': 'one.cumsum()'
            },
            partition_by=['g'],
            order_by=['i'],
            ) .\
        drop_columns(['one']) .\
        order_rows(['i'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'i': [1, 2, 3, 4, 5],
        'g': [1, 2, 2, 1, 1],
        'n': [1, 1, 2, 2, 3],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_sum_cumsum(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'i': [1, 2, 3, 4, 5],
        'o': [1, 1, 1, 1, 1],
        'g': [1, 2, 2, 1, 1],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name=table_name_d). \
                extend({
                's2': 'o.sum()',
                },
                partition_by=['g'],
                order_by=['i'],
            )

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name=table_name_d). \
                extend({
                's2': 'o.cumsum()',
                },
                partition_by=['g'],
            )

    ops = describe_table(d, table_name=table_name_d). \
        extend({
            's': '(1).cumsum()',
            },
            partition_by=['g'],
            order_by=['i'],
            ). \
        extend({
            'n': 's.max()',  # max over cumsum to get sum!
            'n2': '(1).sum()',  # no order present, so meaning is non-cumulative.
            },
            partition_by=['g']
        ). \
        order_rows(['i'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'i':  [1, 2, 3, 4, 5],
        'o':  [1, 1, 1, 1, 1],
        'g':  [1, 2, 2, 1, 1],
        'n':  [3, 2, 2, 3, 3],
        'n2': [3, 2, 2, 3, 3],
        's':  [1, 1, 2, 2, 3],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_project_sum(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'i': [1, 2, 3, 4, 5],
        'g': [1, 2, 2, 1, 1],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name_d). \
        project({
            's': '(1).sum()',
            },
            group_by=['g'],
            ). \
        order_rows(['g'])

    expect = data_algebra.default_data_model.pd.DataFrame({
        'g':  [1, 2],
        's':  [3, 2],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_concat_op(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['a', 'b', 'c'],
        'y': ['1', '2', '3'],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name_d). \
        extend({
            'z': 'x %+% y %+% + x'
            })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['a', 'b', 'c'],
        'y': ['1', '2', '3'],
        'z': ['a1a', 'b2b', 'c3c']
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )


def test_ideom_coalesce_op(get_bq_handle):
    db_handles = get_bq_handle['db_handles']
    data_catalog = get_bq_handle['data_catalog']
    data_schema = get_bq_handle['data_schema']

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': ['a', 'b', None, None],
        'y': ['1', None, '3', None],
    })
    table_name_d = f'{data_catalog}.{data_schema}.pytest_temp_d'

    ops = describe_table(d, table_name=table_name_d). \
        extend({
            'z': 'x %?% y'
            })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'x': ['a', 'b', None, None],
        'y': ['1', None, '3', None],
        'z': ['a', 'b', '3', None],
    })

    data_algebra.test_util.check_transform_on_handles(
        ops=ops,
        data=d,
        expect=expect,
        db_handles=db_handles,
        check_parse=False,
    )
