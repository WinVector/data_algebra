
import sqlite3

import numpy
import pandas

import pytest

from data_algebra.data_ops import *
import data_algebra.SQLite


def test_u_1():
    db_model = data_algebra.SQLite.SQLiteModel()

    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00',
                 '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

    trim_date = user_fn(lambda x: x.str.slice(start=0, stop=10),
                        'DATE',
                        name='trim_date_zz',
                        sql_name='SUBSTR', sql_suffix=', 1, 10')

    ops = describe_table(d, table_name='d'). \
        extend({'date_trimmed': trim_date})

    res_1 = ops.transform(d)

    q = ops.to_sql(db_model)

    with sqlite3.connect(':memory:') as con:
        d.to_sql(name='d', con=con)
        res_db = pandas.read_sql(q, con=con)


def test_u_2():
    db_model = data_algebra.SQLite.SQLiteModel()

    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00',
                 '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

    with pytest.raises(ValueError):
        trim_date = user_fn(lambda x: x.str.slice(start=0, stop=10),
                            'DATE',
                            sql_name='SUBSTR', sql_suffix=', 1, 10')


def test_u_3():
    db_model = data_algebra.SQLite.SQLiteModel()

    # some example data
    d = pandas.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00',
                 '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

    def trim_date_f(x):
        return x.str.slice(start=0, stop=10)
    trim_date = user_fn(trim_date_f,
                    'DATE',
                    sql_name='SUBSTR', sql_suffix=', 1, 10')

    ops = describe_table(d, table_name='d'). \
        extend({'date_trimmed': trim_date})

    res_1 = ops.transform(d)

    q = ops.to_sql(db_model)

    with sqlite3.connect(':memory:') as con:
        d.to_sql(name='d', con=con)
        res_db = pandas.read_sql(q, con=con)

