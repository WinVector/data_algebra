import sqlite3
import collections

import data_algebra.data_types
import data_algebra.data_ops
import data_algebra.cdata
import data_algebra.SQLite


def table_is_keyed_by_columns(table, column_names):
    # check for ill-condition
    missing_columns = set(column_names) - set([c for c in table.columns])
    if len(missing_columns) > 0:
        raise KeyError("missing columns: " + str(missing_columns))
    # get rid of some corner cases
    if table.shape[0] < 2:
        return True
    if len(column_names) < 1:
        return False
    ops = (
        data_algebra.data_ops.describe_table(table, "table")
        .select_columns(column_names)
        .extend({"cdata_temp_one": 1})
        .project({"_cdata_temp_sum": "cdata_temp_one.sum()"}, group_by=column_names)
        .project({"_cdata_temp_sum": "_cdata_temp_sum.max()"})
    )
    t2s = ops.transform(table)
    return t2s.iloc[0, 0] <= 1


class RecordMap:
    def __init__(self, *, blocks_in=None, blocks_out=None):
        if blocks_in is not None:
            if not isinstance(blocks_in, data_algebra.cdata.RecordSpecification):
                raise TypeError(
                    "blocks_in should be a data_algebra.cdata.RecordSpecification"
                )
        if blocks_out is not None:
            if not isinstance(blocks_out, data_algebra.cdata.RecordSpecification):
                raise TypeError(
                    "blocks_out should be a data_algebra.cdata.RecordSpecification"
                )
        if (blocks_in is None) and (blocks_out is None):
            raise ValueError(
                "At least one of blocks_in or blocks_out should not be None"
            )
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out
        self.fmt_string = self.fmt()

    # noinspection PyPep8Naming
    def transform(
        self, X, *, check_blocks_in_keying=True, check_blocks_out_keying=False
    ):
        X = data_algebra.data_types.convert_to_pandas_dataframe(X, "X")
        X = X.reset_index(drop=True)
        db_model = data_algebra.SQLite.SQLiteModel()
        if self.blocks_in is not None:
            x1_descr = data_algebra.data_ops.describe_table(X, table_name="x_blocks_in")
            missing_cols = set(self.blocks_in.control_table_keys).union(
                self.blocks_in.record_keys
            ) - set(x1_descr.column_names)
            if len(missing_cols) > 0:
                raise KeyError("missing required columns: " + str(missing_cols))
            # convert to row-records
            if check_blocks_in_keying:
                # table should be keyed by record_keys + control_table_keys
                if not table_is_keyed_by_columns(
                    X, self.blocks_in.record_keys + self.blocks_in.control_table_keys
                ):
                    raise ValueError(
                        "table is not keyed by blocks_in.record_keys + blocks_in.control_table_keys"
                    )
            with sqlite3.connect(":memory:") as conn:
                db_model.insert_table(conn, X, x1_descr.table_name)
                to_blocks_sql = db_model.blocks_to_row_recs_query(
                    x1_descr, self.blocks_in
                )
                X = db_model.read_query(conn, to_blocks_sql)
        if self.blocks_out is not None:
            x2_descr = data_algebra.data_ops.describe_table(
                X, table_name="x_blocks_out"
            )
            missing_cols = set(self.blocks_out.record_keys) - set(x2_descr.column_names)
            if len(missing_cols) > 0:
                raise KeyError("missing required columns: " + str(missing_cols))
            if check_blocks_out_keying:
                # table should be keyed by record_keys
                if not table_is_keyed_by_columns(X, self.blocks_out.record_keys):
                    raise ValueError("table is not keyed by blocks_out.record_keys")
            # convert to block records
            with sqlite3.connect(":memory:") as conn:
                temp_table = data_algebra.data_ops.TableDescription(
                    "blocks_out", self.blocks_out.control_table.columns
                )
                db_model.insert_table(conn, X, x2_descr.table_name)
                db_model.insert_table(
                    conn, self.blocks_out.control_table, temp_table.table_name
                )
                to_rows_sql = db_model.row_recs_to_blocks_query(
                    x2_descr, self.blocks_out, temp_table
                )
                X = db_model.read_query(conn, to_rows_sql)
        return X

    def fmt(self):
        if (self.blocks_in is None) and (self.blocks_out is None):
            return "RecordMap(no-op)"
        if (self.blocks_in is not None) and (self.blocks_out is not None):
            s = (
                "Transform block records of structure:\n"
                + self.blocks_in.fmt()
                + "to block records of structure:\n"
                + self.blocks_out.fmt()
            )
            return s
        if self.blocks_in is not None:
            s = (
                "Transform block records of structure:\n"
                + self.blocks_in.fmt()
                + "to row records of the form:\n"
                + "  record_keys: "
                + str(self.blocks_in.record_keys)
                + "\n"
                + " "
                + str(self.blocks_in.row_version(include_record_keys=True))
                + "\n"
            )
            return s
        else:
            s = (
                "Transform row records of the form:\n"
                + "  record_keys: "
                + str(self.blocks_out.record_keys)
                + "\n"
                + " "
                + str(self.blocks_out.row_version(include_record_keys=True))
                + "\n"
                + "to block records of structure:\n"
                + self.blocks_out.fmt()
            )
            return s

    def __repr__(self):
        return self.fmt_string

    def __str__(self):
        return self.fmt_string

    def to_simple_obj(self):
        """Create an object for YAML encoding"""

        obj = collections.OrderedDict()
        obj["type"] = "data_algebra.cdata_impl.RecordMap"
        if self.blocks_in is not None:
            obj["blocks_in"] = self.blocks_in.to_simple_obj()
        if self.blocks_out is not None:
            obj["blocks_out"] = self.blocks_out.to_simple_obj()
        return obj


def record_map_from_simple_obj(obj):
    blocks_in = None
    blocks_out = None
    if "blocks_in" in obj.keys():
        blocks_in = data_algebra.cdata.record_spec_from_simple_obj(obj["blocks_in"])
    if "blocks_out" in obj.keys():
        blocks_out = data_algebra.cdata.record_spec_from_simple_obj(obj["blocks_out"])
    return RecordMap(blocks_in=blocks_in, blocks_out=blocks_out)
