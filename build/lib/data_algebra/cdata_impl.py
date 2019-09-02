
import sqlite3
import collections

import pandas

import data_algebra.cdata
import data_algebra.SQLite


class RecordMap:

    def __init__(self, *, blocks_in=None, blocks_out=None):
        if blocks_in is not None:
            if not isinstance(blocks_in, data_algebra.cdata.RecordSpecification):
                raise Exception("blocks_in should be a data_algebra.cdata.RecordSpecificaton")
        if blocks_out is not None:
            if not isinstance(blocks_out, data_algebra.cdata.RecordSpecification):
                raise Exception("blocks_out should be a data_algebra.cdata.RecordSpecificaton")
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out
        self.fmt_string = self.fmt()

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a pandas.DataFrame")
        X = X.copy()
        X.reset_index(inplace=True, drop=True)
        if (self.blocks_in is None) and (self.blocks_out is None):
            return X
        db_model = data_algebra.SQLite.SQLiteModel()
        with sqlite3.connect(':memory:') as conn:
            # convert to row-records
            if self.blocks_in is not None:
                x_descr = data_algebra.data_ops.describe_pandas_table(X, table_name='X')
                db_model.insert_table(conn, X, 'X')
                to_blocks_sql = db_model.blocks_to_row_recs_query(x_descr, self.blocks_in)
                X = db_model.read_query(conn, to_blocks_sql)
            # convert to block records
            if self.blocks_out is not None:
                x_descr = data_algebra.data_ops.describe_pandas_table(X, table_name='X')
                temp_table = data_algebra.data_ops.TableDescription(
                    'blocks_out',
                    self.blocks_out.control_table.columns
                )
                db_model.insert_table(conn, X, 'X')
                db_model.insert_table(conn, self.blocks_out.control_table, temp_table.table_name)
                to_rows_sql = db_model.row_recs_to_blocks_query(x_descr, self.blocks_out, temp_table)
                X = db_model.read_query(conn, to_rows_sql)

        return X

    def fmt(self):
        if (self.blocks_in is None) and (self.blocks_out is None):
            return 'RecordMap(no-op)'
        if (self.blocks_in is not None) and (self.blocks_out is not None):
            s = ("Transform block records of structure:\n"
                 + self.blocks_in.fmt()
                 + "to block records of structure:\n"
                 + self.blocks_out.fmt())
            return s
        if self.blocks_in is not None:
            s = ("Transform block records of structure:\n"
                 + self.blocks_in.fmt()
                 + "to row records of the form:\n"
                 + '  record_keys: ' + str(self.blocks_in.record_keys) + '\n'
                 + ' ' + str(self.blocks_in.row_version(include_record_keys=True)) + '\n'
                 )
            return s
        else:
            s = ("Transform row records of the form:\n"
                 + '  record_keys: ' + str(self.blocks_out.record_keys) + '\n'
                 + ' ' + str(self.blocks_out.row_version(include_record_keys=True)) + '\n'
                 + "to block records of structure:\n"
                 + self.blocks_out.fmt())
            return s


    def __repr__(self):
        return self.fmt_string

    def __str__(self):
        return self.fmt_string

    def to_simple_obj(self):
        obj = collections.OrderedDict()
        obj['type'] = 'data_algebra.cdata_impl.RecordMap'
        if self.blocks_in is not None:
            obj['blocks_in'] = self.blocks_in.to_simple_obj()
        if self.blocks_out is not None:
            obj['blocks_out'] = self.blocks_out.to_simple_obj()
        return obj

def record_map_from_simple_obj(obj):
    blocks_in = None
    blocks_out = None
    if 'blocks_in' in obj.keys():
        blocks_in = data_algebra.cdata.record_spec_from_simple_obj(obj['blocks_in'])
    if 'blocks_out' in obj.keys():
        blocks_out = data_algebra.cdata.record_spec_from_simple_obj(obj['blocks_out'])
    return RecordMap(blocks_in=blocks_in, blocks_out=blocks_out)