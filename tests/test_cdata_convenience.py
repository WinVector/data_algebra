
import re
import io
import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.cdata import *
from data_algebra.data_ops import *


def test_cdata_convenience_1():
    def parse_table(s):
        return data_algebra.default_data_model.pd.read_csv(
            io.StringIO(re.sub("[ \\t]+", "", s))
        )

    d = parse_table(
        """
            n,n_A,n_B,T,E,T+E
            0,10000,9000.0,1000.0,0.005455,0.002913,0.008368
            1,23778,21400.2,2377.8,0.003538,0.001889,0.005427
            2,37556,33800.4,3755.6,0.002815,0.001503,0.004318
            3,51335,46201.5,5133.5,0.002408,0.001286,0.003693
        """
    )
    expect_2 = parse_table(
        """
            n,curve,effect_size
            10000,T,0.005455
            10000,T+E,0.008368
            23778,T,0.003538
            23778,T+E,0.005427
            37556,T,0.002815
            37556,T+E,0.004318
            51335,T,0.002408
            51335,T+E,0.003693
        """
    )
    expect_2.columns = ["n", "curve", "effect_size"]
    expect_3 = parse_table(
        """
            n,T,T+E
            10000,0.005455,0.008368
            23778,0.003538,0.005427
            37556,0.002815,0.004318
            51335,0.002408,0.003693
        """
    )

    mp = pivot_rowrecs_to_blocks(
        attribute_key_column="curve",
        attribute_value_column="effect_size",
        record_keys=["n"],
        record_value_columns=["T", "T+E"],
    )
    d2 = mp.transform(d)
    assert data_algebra.test_util.equivalent_frames(d2, expect_2)

    mpi = pivot_blocks_to_rowrecs(
        attribute_key_column="curve",
        attribute_value_column="effect_size",
        record_keys=["n"],
        record_value_columns=["T", "T+E"],
    )
    d3 = mpi.transform(d2)
    assert data_algebra.test_util.equivalent_frames(d3, expect_3)

    db_model = data_algebra.SQLite.SQLiteModel()

    ops1 = describe_table(d, table_name='d') .\
        convert_records(mp)
    d2p = ops1.transform(d)
    assert data_algebra.test_util.equivalent_frames(d2p, expect_2)

    with sqlite3.connect(":memory:") as conn:
        db_model.prepare_connection(conn)
        db_handle = db_model.db_handle(conn)
        db_handle.insert_table(d, table_name='d', allow_overwrite=True)
        db_handle.insert_table(d, table_name='d2', allow_overwrite=True)

        temp_tables1 = dict()
        sql_regular = db_handle.to_sql(ops1, pretty=True, use_with=False, annotate=False, temp_tables=temp_tables1)
        for k, v in temp_tables1.items():
            db_handle.insert_table(v, table_name=k, allow_overwrite=True)
        res_regular = db_handle.read_query(sql_regular)

        temp_tables1 = dict()
        sql_regular_a = db_handle.to_sql(ops1, pretty=True, use_with=False, annotate=True, temp_tables=temp_tables1)
        for k, v in temp_tables1.items():
            db_handle.insert_table(v, table_name=k, allow_overwrite=True)
        res_regular_a = db_handle.read_query(sql_regular_a)

        temp_tables1 = dict()
        sql_with = db_handle.to_sql(ops1, pretty=True, use_with=True, annotate=False, temp_tables=temp_tables1)
        for k, v in temp_tables1.items():
            db_handle.insert_table(v, table_name=k, allow_overwrite=True)
        res_with = db_handle.read_query(sql_with)

        temp_tables1 = dict()
        sql_with_a = db_handle.to_sql(ops1, pretty=True, use_with=True, annotate=True, temp_tables=temp_tables1)
        for k, v in temp_tables1.items():
            db_handle.insert_table(v, table_name=k, allow_overwrite=True)
        res_with_a = db_handle.read_query(sql_with_a)

    assert data_algebra.test_util.equivalent_frames(res_regular, expect_2)
    assert data_algebra.test_util.equivalent_frames(res_regular_a, expect_2)
    assert data_algebra.test_util.equivalent_frames(res_with, expect_2)
    assert data_algebra.test_util.equivalent_frames(res_with_a, expect_2)
