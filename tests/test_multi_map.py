
from data_algebra.data_ops import descr
from data_algebra.solutions import def_multi_column_map
import data_algebra.test_util

# From:
# https://github.com/WinVector/data_algebra/blob/main/Examples/MultiJoin/MultiJoin.ipynb
def test_multi_map():
    pd = data_algebra.data_model.default_data_model().pd  # Pandas

    d = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'va': ['a', 'b', 'a', 'c'],
        'vb': ['e', 'e', 'g', 'f'],
    })

    m = pd.DataFrame({
        'column_name': ['va', 'va', 'vb', 'vb'],
        'column_value': ['a', 'b', 'e', 'f'],
        'mapped_value': [1., 2., 3., 4.],
    })

    row_keys = ['id']
    cols_to_map = ['va', 'vb']
    ops = def_multi_column_map(
        descr(d=d),
        mapping_table=descr(m=m),
        row_keys=row_keys,
        cols_to_map=cols_to_map,
        coalesce_value=0.0,
        cols_to_map_back=[f'{c}_mapped' for c in cols_to_map],
    )
    res = ops.eval({'d': d, 'm': m})

    expect = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'va_mapped': [1.0, 2.0, 1.0, 0.0],
        'vb_mapped': [3.0, 3.0, 0.0, 4.0],
    })

    assert data_algebra.test_util.equivalent_frames(res, expect)

    data_algebra.test_util.check_transform(ops, data={"d": d, "m": m}, expect=expect, valid_for_empty=False,
    )
