
import sqlite3

import data_algebra.util
import data_algebra.test_util
from data_algebra.cdata import *
from data_algebra.data_ops import *
import data_algebra.SQLite


def test_ordered_agg_group():
    expect = data_algebra.pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'DATE1': ['2001-01-02 00:00:00', '2000-04-01 00:00:00', '2014-04-07 00:00:00', '2005-06-16 00:00:00',
                  '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
        'OP1': ['A', 'A', 'D', 'A', 'B', 'B'],
        'DATE2': ['2015-04-25 00:00:00', None, None, '2009-01-20 00:00:00', '2010-10-10 00:00:00', None],
        'OP2': ['B', None, None, 'B', 'A', None],
        'DATE3': [None, None, None, '2009-01-20 00:00:00', None, None],
        'OP3': [None, None, None, 'D', None, None],
    })

    d = data_algebra.pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00',
                 '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

    # d

    # %%

    diagram = data_algebra.pd.DataFrame({
        'rank': ['1', '2', '3'],
        'DATE': ['DATE1', 'DATE2', 'DATE3'],
        'OP': ['OP1', 'OP2', 'OP3']
    })

    # diagram

    # %%

    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=diagram,
            record_keys=['ID']
        ))

    str(record_map)

    # %%

    ops = describe_table(d, table_name='d'). \
        extend({'rank': '_row_number()'},
               partition_by=['ID'],
               order_by=['DATE', 'OP']). \
        convert_records(record_map). \
        order_rows(['ID'])

    res1 = ops.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect, res1)

    # %%

    db_model = data_algebra.SQLite.SQLiteModel()

    sql = ops.to_sql(db_model, pretty=True)

    # print(sql)

    # %%

    con = sqlite3.connect(":memory:")
    db_model.prepare_connection(con)
    d.to_sql('d', con, if_exists='replace')

    res_db = data_algebra.pd.read_sql_query(sql, con)
    con.close()

    # res_db
    assert data_algebra.test_util.equivalent_frames(expect, res_db)

    # %%

    # proves we could pass a lambda into agg (need to extend framework to allow this)
    r = d.groupby(['ID', 'DATE']).agg(lambda vals: ', '.join(sorted([str(vi) for vi in set(vals)])))
    r.reset_index(inplace=True, drop=False)

    ops2 = describe_table(r, table_name='r'). \
        extend({'rank': '_row_number()'},
               partition_by=['ID'],
               order_by=['DATE']). \
        convert_records(record_map). \
        order_rows(['ID'])

    res_ag = ops2.transform(r)

    expect_ag = data_algebra.pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'DATE1': ['2001-01-02 00:00:00', '2000-04-01 00:00:00', '2014-04-07 00:00:00', '2005-06-16 00:00:00',
                  '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
        'OP1': ['A', 'A', 'D', 'A', 'B', 'B'],
        'DATE2': ['2015-04-25 00:00:00', None, None, '2009-01-20 00:00:00', '2010-10-10 00:00:00', None],
        'OP2': ['B', None, None, 'B, D', 'A', None],
        'DATE3': [None, None, None, '2012-12-01 00:00:00', None, None],
        'OP3': [None, None, None, 'C', None, None],
        })

    assert data_algebra.test_util.equivalent_frames(expect_ag, res_ag)
