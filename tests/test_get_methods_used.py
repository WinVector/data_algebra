
from data_algebra.data_ops import TableDescription


def test_get_methods_used():
    ops = (
        TableDescription(table_name='d', column_names=['a', 'b', 'c'])
            .extend({'p': 'a**2'})
            .extend({'q': 'a/2'})
            .project({'q': 'q.max()', 'p': 'p.min()'})
            .select_rows('q.sqrt() < 2')
            .extend({'z': 'p + q'})
    )
    found = ops.methods_used()
    assert found == {'**', '+', '/', '<', 'min', 'max'}
