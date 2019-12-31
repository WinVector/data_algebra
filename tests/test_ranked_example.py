
import pandas

import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.cdata import *

def test_ranked_example():
    # from https://community.rstudio.com/t/tidying-data-reorganizing-tibble/48292
    d = data_algebra.pd.DataFrame({
        'ID': [1, 1, 2, 3, 4, 4, 4, 5, 5, 6],
        'OP': ['A', 'B', 'A', 'C', 'C', 'A', 'D', 'A', 'B', 'B'],
        'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00',
                 '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00',
                 '2009-01-20 00:00:00', '2010-10-10 00:00:00', '2003-11-09 00:00:00',
                 '2004-01-09 00:00:00'],
        })

    ops = describe_table(d, 'd'). \
        extend({'rank': '_row_number()'},
               partition_by="ID", order_by="DATE")
    d = ops.transform(d)

    diagram = data_algebra.pd.DataFrame({
        'rank':  [1, 2, 3, 4, 5],
        'DATE': ['DATE1', 'DATE2', 'DATE3', 'DATE4', 'DATE5'],
        'OP': ['OP1', 'OP2', 'OP3', 'OP4', 'OP5']
    })

    record_spec = RecordSpecification(
        control_table=diagram,
        control_table_keys=['rank'],
        record_keys=['ID']
    )
    mapping = RecordMap(blocks_in=record_spec)

    res = mapping.transform(d)

    expect = data_algebra.pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'DATE1': ['2001-01-02 00:00:00', '2000-04-01 00:00:00', '2014-04-07 00:00:00', '2005-06-16 00:00:00', '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
        'OP1': ['A', 'A', 'C', 'A', 'B', 'B'],
        'DATE2': ['2015-04-25 00:00:00', None, None, '2009-01-20 00:00:00', '2010-10-10 00:00:00', None],
        'OP2': ['B', None, None, 'D', 'A', None],
        'DATE3': [None, None, None, '2012-12-01 00:00:00', None, None],
        'OP3': [None, None, None, 'C', None, None],
        'DATE4': [None, None, None, None, None, None],
        'OP4': [None, None, None, None, None, None],
        'DATE5': [None, None, None, None, None, None],
        'OP5': [None, None, None, None, None, None],
        })

    assert data_algebra.test_util.equivalent_frames(res, expect)
