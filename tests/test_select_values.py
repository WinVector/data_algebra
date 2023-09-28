import data_algebra

from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.test_util


def test_select_values_db_test_1():
    db_handles = data_algebra.test_util.get_test_dbs()
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c1": ["A", "C", "E"], "c2": [1, 2, 3],}
    )
    for db_handle in db_handles:
        # print(db_handle)
        sql = "\n".join(db_handle.table_values_to_sql_str_list(d)) + "\n"
        # print(sql)
        res = db_handle.read_query(sql)
        assert data_algebra.test_util.equivalent_frames(res, d)
    for db_handle in db_handles:
        db_handle.close()


def test_select_values_select_rows_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c1": ["A", "A", "E"], "c2": [1, 2, 2],}
    )

    ops0 = describe_table(d, table_name="d").select_rows([])
    assert isinstance(ops0, TableDescription)
    res0 = ops0.transform(d)
    assert data_algebra.test_util.equivalent_frames(res0, d)

    ops1a = describe_table(d, table_name="d").select_rows('c1 == "A"')
    res1a = ops1a.transform(d)
    expect1a = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c1": ["A", "A"], "c2": [1, 2],}
    )
    assert data_algebra.test_util.equivalent_frames(res1a, expect1a)

    ops1 = describe_table(d, table_name="d").select_rows(['c1 == "A"'])
    res1 = ops1.transform(d)
    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"c1": ["A", "A"], "c2": [1, 2],}
    )
    assert data_algebra.test_util.equivalent_frames(res1, expect1)

    ops2 = describe_table(d, table_name="d").select_rows(['c1 == "A"', "c2 == 2",])
    res2 = ops2.transform(d)
    expect2 = data_algebra.data_model.default_data_model().pd.DataFrame({"c1": ["A"], "c2": [2],})
    assert data_algebra.test_util.equivalent_frames(res2, expect2)
