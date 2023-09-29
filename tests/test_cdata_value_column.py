
import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.cdata import RecordMap, RecordSpecification


def test_cdata_value_column_1():
    pd = data_algebra.data_model.default_data_model().pd
    # from:
    #  https://github.com/WinVector/Examples/blob/main/calling_R_from_Python/sig_pow.ipynb
    rs = RecordSpecification(
        pd.DataFrame({
            'group': ['treatment', 'control'],
            'y': ['treatment', 'control'],
            'tail': ['treatment_tail', 'control_tail'],
        }),
        record_keys=['x'],
        control_table_keys=['group'],
        )
    assert rs.row_version() == ['x', 'treatment', 'control', 'treatment_tail', 'control_tail']
    assert rs.value_column_form() == rs.value_column_form().value_column_form()
    assert isinstance(rs.value_column_form(), RecordSpecification)
    map_from_rows = rs.map_from_rows()
    assert isinstance(map_from_rows, RecordMap)
    map_to_rows = rs.map_to_rows()
    assert isinstance(map_to_rows, RecordMap)
    map_from_keyed_column = rs.map_from_keyed_column()
    assert isinstance(map_from_keyed_column, RecordMap)
    map_to_keyed_column = rs.map_to_keyed_column()
    assert isinstance(map_to_keyed_column, RecordMap)
    d_row_form = pd.DataFrame({
        'x': [-0.14, -0.2],
        'control': [4.96, 5.21],
        'treatment': [1.069, 1.16196],
        'control_tail': [2, 3],
        'treatment_tail': [19, 11],
        })
    d_records = map_from_rows(d_row_form)
    d_column = map_to_keyed_column(d_records)
    d_back = map_from_keyed_column(d_column)
    d_rows_back = map_to_rows(d_records)
    assert data_algebra.test_util.equivalent_frames(d_row_form, d_rows_back)
    assert data_algebra.test_util.equivalent_frames(d_records, d_back)
    s2 = rs.row_record_form()
    assert isinstance(s2, RecordSpecification)
    s2_map_from_keyed_column = s2.map_from_keyed_column()
    assert isinstance(s2_map_from_keyed_column, RecordMap)
    s2_map_to_keyed_column = s2.map_to_keyed_column()
    assert isinstance(s2_map_to_keyed_column, RecordMap)
    d_column_s2 = s2_map_to_keyed_column(d_row_form)
    assert data_algebra.test_util.equivalent_frames(d_column, d_column_s2)
    d_rows_back_d2 = s2_map_from_keyed_column(d_column)
    assert data_algebra.test_util.equivalent_frames(d_row_form, d_rows_back_d2)
    for obj in [
        rs, 
        map_from_rows, map_to_rows,
        map_from_keyed_column, map_to_keyed_column,
        s2, 
        s2_map_from_keyed_column, s2_map_to_keyed_column]:
        assert isinstance(str(obj), str)
        assert isinstance(repr(obj), str)
        assert isinstance(obj.fmt(), str)
        assert isinstance(obj._repr_html_(), str)
