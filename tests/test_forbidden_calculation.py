
import pytest

from data_algebra.data_ops import *

def test_forbidden_calculation():
    td = TableDescription(table_name='d', column_names=['a', 'b', 'c'])

    # test using undefined column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'x'})

    # test colliding with know column
    with pytest.raises(ValueError):
        td.rename_columns({'a': 'b'})

    # test swaps don't show up in forbidden
    ops1 = td.rename_columns({'a': 'b', 'b': 'a'})
    f1 = ops1.forbidden_columns()
    assert set(f1['d']) == set()

    # test new column creation triggers forbidden annotation
    ops2 = td.rename_columns({'e': 'a'})
    f2 = ops2.forbidden_columns()
    assert set(['e']) == f2['d']

    # test merge
    ops3 = td.rename_columns({'e': 'a'}).rename_columns({'f': 'b'})
    f3 = ops3.forbidden_columns()
    assert set(['e', 'f']) == f3['d']

    # test composition
    ops4 = td.rename_columns({'e': 'a'}).rename_columns({'a': 'b'})
    f4 = ops4.forbidden_columns()
    assert set(['e']) == f4['d']
