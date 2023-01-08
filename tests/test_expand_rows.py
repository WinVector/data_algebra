
import data_algebra
import data_algebra.MySQL
import data_algebra.test_util
from data_algebra.data_ops import descr
from data_algebra.solutions import replicate_rows_query


def test_replicate_rows_query():
    # get a pandas namespace
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'key': ['a', 'b', 'c'],
        'n': [1, 2, 3],
    })
    d_descr = descr(d=d)
    ops, rt = replicate_rows_query(
        d_descr,
        count_column_name='n',
        seq_column_name='i',
        join_temp_name='rt',
        max_count=3,
        )
    res = ops.eval({'d': d, 'rt': rt})
    expect = pd.DataFrame({
        'key': ['a', 'b', 'b', 'c', 'c', 'c'],
        'n': [1, 2, 2, 3, 3, 3],
        'i': ['0', '0', '1', '0', '1', '2'],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
    data_algebra.test_util.check_transform(
        ops=ops,
        data={'d': d, 'rt': rt},
        expect=expect,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )
