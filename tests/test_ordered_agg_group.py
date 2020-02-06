
import sqlite3

import data_algebra.util
import data_algebra.test_util
from data_algebra.cdata import *
from data_algebra.data_ops import *
import data_algebra.SQLite


def test_ordered_agg_group():
    expect = data_algebra.default_data_model.pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'DATE1': ['2001-01-02 00:00:00', '2000-04-01 00:00:00', '2014-04-07 00:00:00', '2005-06-16 00:00:00',
                  '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
        'OP1': ['A', 'A', 'D', 'A', 'B', 'B'],
        'DATE2': ['2015-04-25 00:00:00', None, None, '2009-01-20 00:00:00', '2010-10-10 00:00:00', None],
        'OP2': ['B', None, None, 'B', 'A', None],
        'DATE3': [None, None, None, '2009-01-20 00:00:00', None, None],
        'OP3': [None, None, None, 'D', None, None],
    })

    d = data_algebra.default_data_model.pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00',
                 '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

    # d

    # %%

    diagram = data_algebra.default_data_model.pd.DataFrame({
        'rank': [1, 2, 3],
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
        select_columns(['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
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

    res_db = data_algebra.default_data_model.pd.read_sql_query(sql, con)

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
        select_columns(['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
        order_rows(['ID'])

    res_ag = ops2.transform(r)

    expect_ag = data_algebra.default_data_model.pd.DataFrame({
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

    # proves we could pass a lambda into agg (need to extend framework to allow this)
    def sorted_concat(vals):
        return ', '.join(sorted([str(vi) for vi in set(vals)]))

    ops3 = describe_table(d, table_name='d'). \
        project({'OP': sorted_concat},
                group_by=['ID', 'DATE']). \
        extend({'rank': '_row_number()'},
               partition_by=['ID'],
               order_by=['DATE']). \
        convert_records(record_map). \
        select_columns(['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
        order_rows(['ID'])

    res_ag3 = ops3.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect_ag, res_ag3)

    ops4 = describe_table(d, table_name='d'). \
        project({'OP': user_fn(sorted_concat, 'OP')},
                group_by=['ID', 'DATE']). \
        extend({'rank': '_row_number()'},
               partition_by=['ID'],
               order_by=['DATE']). \
        convert_records(record_map). \
        select_columns(['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
        order_rows(['ID'])

    res_ag4 = ops4.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect_ag, res_ag4)

    class SortedConcat:
        def __init__(self):
            self.accum = set()

        def step(self, value):
            self.accum.add(str(value))

        def finalize(self):
            return ', '.join(sorted([v for v in self.accum]))

    # sqlite has group_concat https://pythontic.com/database/sqlite/aggregate%20functions
    sql4 = ops4.to_sql(db_model)
    # https://docs.python.org/2/library/sqlite3.html
    con.create_aggregate("sorted_concat", 1, SortedConcat)
    res_db4 = data_algebra.default_data_model.pd.read_sql_query(sql4, con)
    assert data_algebra.test_util.equivalent_frames(expect_ag, res_db4,
                                                    check_column_order=True, check_row_order=True)

    ops5 = describe_table(d, table_name='d'). \
        project({'OP': user_fn('lambda vals: ", ".join(sorted([str(vi) for vi in set(vals)]))', 'OP')},
                group_by=['ID', 'DATE']). \
        extend({'rank': '_row_number()'},
               partition_by=['ID'],
               order_by=['DATE']). \
        convert_records(record_map). \
        select_columns(['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
        order_rows(['ID'])

    res_ag5 = ops5.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect_ag, res_ag5)

    con.close()
