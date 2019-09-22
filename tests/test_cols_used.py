
import pytest
import pandas
import data_algebra.util
from data_algebra.data_ops import *
import data_algebra.PostgreSQL
from data_algebra.util import od
import data_algebra.yaml


def test_cols_used():
    table = TableDescription('d', ['a', 'b', 'c', 'd'])

    ops = table .\
           select_columns(['a', 'b']) .\
           natural_join(b=
              table .\
                 select_columns(['a', 'c']),
              by=['a'], jointype='INNER')

    used = ops.columns_used()
    d_used = used['d']

    assert set(['a', 'b', 'c']) == d_used


    ops2 = TableDescription(table_name='d', column_names=['a', 'b', 'c', 'd']) .\
           select_columns(['a', 'b']) .\
           natural_join(b=
              TableDescription(table_name='d', column_names=['a', 'b', 'c', 'd']) .\
                 select_columns(['a', 'c']),
              by=['a'], jointype='INNER')

    used2 = ops2.columns_used()
    d_used2 = used2['d']

    assert set(['a', 'b', 'c']) == d_used2
