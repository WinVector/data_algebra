
import data_algebra

from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.test_util


def test_select_values_db_test_1():
    db_handles = data_algebra.test_util.get_test_dbs()

    d = data_algebra.default_data_model.pd.DataFrame({
        'c1': ['A', 'C', 'E'],
        'c2': [1, 2, 3],
        })

    for db_handle in db_handles:
        # print(db_handle)
        sql = db_handle.table_values_to_sql(d)
        # print(sql)
        res = db_handle.read_query(sql)
        assert data_algebra.test_util.equivalent_frames(res, d)

    for db_handle in db_handles:
        db_handle.close()

