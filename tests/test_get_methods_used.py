
from data_algebra.data_ops import TableDescription


def test_get_methods_used():
    ops = (
        TableDescription(table_name='d', column_names=['a', 'b', 'c'])
            .extend({'p': 'a**2'})
            .project({'q': 'b.max()'})
            .select_rows('q.sqrt() < 2')
    )
    found = ops.methods_used()
    assert found == {'**', '<', 'max'}
