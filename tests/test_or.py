
import numpy
import pandas

from data_algebra.data_ops import *
import data_algebra.util
import data_algebra.test_util

import pytest

# TODO: SQL tests and doc!

def test_or_1():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        select_rows('(ID == 3) | (ID == 4)')
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [3, 4, 4, 4, 4],
        'OP': ['D', 'C', 'A', 'D', 'B'],
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)


def test_or_2():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    with pytest.raises(ValueError):
        ops = describe_table(d, table_name='d'). \
            select_rows('(ID == 3) or (ID == 4)')


def test_in_1():
    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    })

    ops = describe_table(d, table_name='d'). \
        select_rows('ID in [3, 4]')
    d2 = ops.transform(d)

    expect = pandas.DataFrame({
        'ID': [3, 4, 4, 4, 4],
        'OP': ['D', 'C', 'A', 'D', 'B'],
    })

    assert data_algebra.test_util.equivalent_frames(expect, d2)