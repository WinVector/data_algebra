
import data_algebra
import data_algebra.util
import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.cdata import *


def test_one_row_cdata_convert():
    # test some conversions related to:
    # https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/solving_problems_using_data_algebra.ipynb
    a = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'a'],
        'sensor': ['s1', 's2'],
        'group_sensor_mean': [-0.103881, 0.018839],
        'group_sensor_est_var': [0.006761, 0.004844],
    })
    b = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a'],
        'group_sensor_mean_s1': [-0.103881],
        'group_sensor_mean_s2': [0.018839],
        'group_sensor_est_var_s1': [0.006761],
        'group_sensor_est_var_s2': [0.004844],
    })
    record_in = data_algebra.default_data_model.pd.DataFrame({
        'sensor': ['s1', 's2'],
        'group_sensor_mean': ['group_sensor_mean_s1', 'group_sensor_mean_s2'],
        'group_sensor_est_var': ['group_sensor_est_var_s1', 'group_sensor_est_var_s2'],
    })
    record_out = data_algebra.default_data_model.pd.DataFrame({
        'group_sensor_mean_s1': ['group_sensor_mean_s1'],
        'group_sensor_mean_s2': ['group_sensor_mean_s2'],
        'group_sensor_est_var_s1': ['group_sensor_est_var_s1'],
        'group_sensor_est_var_s2': ['group_sensor_est_var_s2'],
    })

    # no output row version
    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=record_in,
            record_keys=['group'],
        ),
    )
    f_a = record_map.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a)

    # explicit output row version
    record_map_e = RecordMap(
        blocks_in=RecordSpecification(
            control_table=record_in,
            record_keys=['group'],
        ),
        blocks_out=RecordSpecification(
            control_table=record_out,
            control_table_keys=[],
            record_keys=['group'],
        )
    )
    f_a_e = record_map_e.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a_e)

    # explicit output row version, 2
    record_map_e2 = RecordMap(
        blocks_in=RecordSpecification(
            control_table=record_in,
            record_keys=['group'],
        ),
        blocks_out=RecordSpecification(
            control_table=record_out,
            record_keys=['group'],
        )
    )
    f_a_e2 = record_map_e2.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a_e2)

    # TODO: test inverses and DB paths
