
import data_algebra
from data_algebra.data_ops import data, descr, describe_table, ex

import data_algebra.SQLite
import data_algebra.test_util


def test_cross_project_join_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2, 3],})
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({"y": ["a", "b", "c", "d"],})
    ops = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=[], jointype="CROSS"
    )
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "x": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "y": ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d",],
        }
    )
    sql = data_algebra.SQLite.SQLiteModel().to_sql(ops)
    assert isinstance(sql, str)
    data_algebra.test_util.check_transform(
        ops=ops, data={"d1": d1, "d2": d2}, expect=expect,
        valid_for_empty=False  # TODO: think about putting this check back in
    )
