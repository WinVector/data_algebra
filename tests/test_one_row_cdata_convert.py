import data_algebra
import data_algebra.util
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.cdata import pivot_blocks_to_rowrecs,  pivot_rowrecs_to_blocks, pivot_specification, unpivot_specification, RecordMap, RecordSpecification 


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

    # no output row version
    record_map = RecordMap(
        blocks_in=RecordSpecification(control_table=record_in, record_keys=["group"], control_table_keys=["sensor"]),
    )
    f_a = record_map.transform(a)
    assert data_algebra.test_util.equivalent_frames(b, f_a)
    f_b = record_map.inverse().transform(b)
    assert data_algebra.test_util.equivalent_frames(a, f_b)

    # test db paths

    ops1 = describe_table(a, table_name="a").convert_records(record_map)
    data_algebra.test_util.check_transform(ops=ops1, data=a, expect=b,
    )
    ops1_r = describe_table(b, table_name="b").convert_records(record_map.inverse())
    data_algebra.test_util.check_transform(ops=ops1_r, data=b, expect=a,
    )
