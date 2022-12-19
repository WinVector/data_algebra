import data_algebra
import data_algebra.util
import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.cdata import *
import data_algebra.SQLite


def test_one_row_cdata_convert():
    # test some conversions related to:
    # https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/solving_problems_using_data_algebra.ipynb
    # Note: in converting to row recs we currently prefer no table for the single row case, but
    # making sure this case works on all code paths.
    a = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a", "a"],
            "sensor": ["s1", "s2"],
            "group_sensor_mean": [-0.103881, 0.018839],
            "group_sensor_est_var": [0.006761, 0.004844],
        }
    )
    b = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a"],
            "group_sensor_mean_s1": [-0.103881],
            "group_sensor_mean_s2": [0.018839],
            "group_sensor_est_var_s1": [0.006761],
            "group_sensor_est_var_s2": [0.004844],
        }
    )
    record_in = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "sensor": ["s1", "s2"],
            "group_sensor_mean": ["group_sensor_mean_s1", "group_sensor_mean_s2"],
            "group_sensor_est_var": [
                "group_sensor_est_var_s1",
                "group_sensor_est_var_s2",
            ],
        }
    )
    record_out = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group_sensor_mean_s1": ["group_sensor_mean_s1"],
            "group_sensor_mean_s2": ["group_sensor_mean_s2"],
            "group_sensor_est_var_s1": ["group_sensor_est_var_s1"],
            "group_sensor_est_var_s2": ["group_sensor_est_var_s2"],
        }
    )

    # no output row version
    record_map = RecordMap(
        blocks_in=RecordSpecification(control_table=record_in, record_keys=["group"],),
    )
    f_a = record_map.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a)
    f_b = record_map.inverse().transform(b)
    assert data_algebra.test_util.equivalent_frames(a, f_b)

    # explicit output row version
    record_map_e = RecordMap(
        blocks_in=RecordSpecification(control_table=record_in, record_keys=["group"],),
        blocks_out=RecordSpecification(
            control_table=record_out, control_table_keys=[], record_keys=["group"],
        ),
    )
    f_a_e = record_map_e.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a_e)
    f_b_e = record_map_e.inverse().transform(b)
    assert data_algebra.test_util.equivalent_frames(a, f_b_e)

    # explicit output row version, 2
    record_map_e2 = RecordMap(
        blocks_in=RecordSpecification(control_table=record_in, record_keys=["group"],),
        blocks_out=RecordSpecification(
            control_table=record_out, record_keys=["group"],
        ),
    )
    f_a_e2 = record_map_e2.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a_e2)
    f_b_e2 = record_map_e2.inverse().transform(b)
    assert data_algebra.test_util.equivalent_frames(a, f_b_e2)

    # test db paths

    ops1 = describe_table(a, table_name="a").convert_records(record_map)
    data_algebra.test_util.check_transform(ops=ops1, data=a, expect=b,
        try_on_Polars=False,  # TODO: turn this on
    )
    ops1_r = describe_table(b, table_name="b").convert_records(record_map.inverse())
    data_algebra.test_util.check_transform(ops=ops1_r, data=b, expect=a,
        try_on_Polars=False,  # TODO: turn this on
    )

    db_model = data_algebra.SQLite.SQLiteModel()

    ops1_e = describe_table(a, table_name="a").convert_records(record_map_e)
    data_algebra.test_util.check_transform(ops=ops1_e, data=a, expect=b,
        try_on_Polars=False,  # TODO: turn this on
    )
    example_sql = db_model.to_sql(ops1_e)
    assert isinstance(example_sql, str)
    # print(example_sql)
    ops1_e_r = describe_table(b, table_name="b").convert_records(record_map_e.inverse())
    data_algebra.test_util.check_transform(ops=ops1_e_r, data=b, expect=a,
        try_on_Polars=False,  # TODO: turn this on
    )

    ops1_e2 = describe_table(a, table_name="a").convert_records(record_map_e2)
    data_algebra.test_util.check_transform(ops=ops1_e2, data=a, expect=b,
        try_on_Polars=False,  # TODO: turn this on
    )
    ops1_e2_r = describe_table(b, table_name="b").convert_records(
        record_map_e2.inverse()
    )
    data_algebra.test_util.check_transform(ops=ops1_e2_r, data=b, expect=a,
        try_on_Polars=False,  # TODO: turn this on
    )
