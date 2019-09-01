import re

import sqlite3

import pandas

import data_algebra.data_ops

class RecordSpecification:

    def __init__(self,
                 control_table,
                 *,
                 record_keys=None,
                 control_table_keys=None):
        if not isinstance(control_table, pandas.DataFrame):
            raise Exception("control_table should be a pandas.DataFrame")
        self.control_table = control_table.copy()
        self.control_table.reset_index(inplace=True, drop=True)
        if record_keys is None:
            record_keys = []
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            control_table_keys = [control_table.columns[0]]
        self.control_table_keys = [k for k in control_table_keys]

    def fmt(self):
        s = (
                'RecordSpecification\n'
                + '   record_keys: ' + str(self.record_keys) + '\n'
                + '   control_table_keys: ' + str(self.control_table_keys) + '\n'
                + '   control_table:\n'
                + '   ' + re.sub('\n', '\n   ', str(self.control_table))
        )
        return s

    def __repr__(self):
        return self.fmt()

    def __str__(self):
        return self.fmt()


class RecordMap:

    def __init__(self, *, blocks_in=None, blocks_out=None):
        if blocks_in is not None:
            if not isinstance(blocks_in, RecordSpecification):
                raise Exception("blocks_in should be a data_algebra.cdata.RecordSpecificaton")
        if blocks_out is not None:
            if not isinstance(blocks_out, RecordSpecification):
                raise Exception("blocks_out should be a data_algebra.cdata.RecordSpecificaton")
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a pandas.DataFrame")
        X = X.copy()
        X.reset_index(inplace=True, drop=True)
        if (self.blocks_in is None) and (self.blocks_out is None):
            return X
        db_model = data_algebra.SQLite.SQLiteModel()
        with sqlite3.connect(':memory:') as conn:
            if self.blocks_in is not None:
                x_descr = data_algebra.data_ops.describe_pandas_table(X, table_name='X')
                temp_table = data_algebra.data_ops.TableDescription(
                    'blocks_in',
                    self.blocks_in.control_table.columns
                )
                db_model.insert_table(conn, X, 'X')
                db_model.insert_table(conn, self.blocks_in.control_table, temp_table.table_name)
                to_rows_sql = db_model.row_recs_to_blocks_query(x_descr, self.blocks_in, temp_table)
                X = db_model.read_query(conn, to_rows_sql)
            if self.blocks_out is not None:
                x_descr = data_algebra.data_ops.describe_pandas_table(X, table_name='X')
                db_model.insert_table(conn, X, 'X')
                to_blocks_sql = db_model.blocks_to_row_recs_query(x_descr, self.blocks_out)
                X = db_model.read_query(conn, to_blocks_sql)
        return X

    def fmt(self):
        if (self.blocks_in is None) and (self.blocks_out is None):
            return 'RecordMap(no-op)'
        return str(type(self))

    def __repr__(self):
        return self.fmt()

    def __str__(self):
        return self.fmt()
