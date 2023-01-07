
import data_algebra
from data_algebra.data_ops import descr
import data_algebra.solutions
import data_algebra.test_util
import data_algebra.SparkSQL
import data_algebra.MySQL
import data_algebra.PostgreSQL
import data_algebra.BigQuery


def test_braid():
    pd = data_algebra.data_model.default_data_model().pd
    d_state = pd.DataFrame({
        't': [1., 3., 5.],
        'state': ['a', 'b', 'c'],
    })
    d_event = pd.DataFrame({
        't': [1., 4.],
        'value': [10., 20.],
    })
    ops = data_algebra.solutions.braid_data(
        d_state=descr(d_state=d_state),
        d_event=descr(d_event=d_event),
        order_by=['t'],
        partition_by=[],
        state_value_column_name='state',
        event_value_column_names=['value'],
        stand_in_values={'state': '""', 'value': 0.0},
    ).order_rows(['t', 'record_type', 'state'], reverse=['record_type'])
    res = ops.eval({'d_state': d_state, 'd_event': d_event})
    # print(data_algebra.util.pandas_to_example_str(res))
    expect = pd.DataFrame({
        't': [1., 1., 3., 4., 5.],
        'state': ['a', 'a', 'b', 'b', 'c'],
        'value': [None, 10., None, 20., None],
        'record_type': ['state_row', 'event_row', 'state_row', 'event_row', 'state_row'],
        })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data={'d_state': d_state, 'd_event': d_event},
        expect=expect,
        models_to_skip={  # TODO: vet on more dbs
            data_algebra.SparkSQL.SparkSQLModel(),
            data_algebra.MySQL.MySQLModel(),
            data_algebra.PostgreSQL.PostgreSQLModel(),
            data_algebra.BigQuery.BigQueryModel(),
        },
    )
