import sqlite3
import collections

import pandas

import data_algebra.data_types
import data_algebra.data_ops
import data_algebra.cdata
import data_algebra.SQLite


class RecordMap:
    def __init__(self, *, blocks_in=None, blocks_out=None, strict=False):
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
        if (blocks_in is not None) and (blocks_out is not None):
            unknown = set(blocks_out.record_keys) - set(blocks_in.record_keys)
            if len(unknown) > 0:
                raise ValueError("unknown outgoing record_keys:" + str(unknown))
            unknown = set(blocks_out.content_keys) - set(blocks_in.content_keys)
            if len(unknown) > 0:
                raise ValueError("unknown outgoing content_keys" + str(unknown))
            if strict:
                if set(blocks_in.record_keys) != set(blocks_out.record_keys):
                    raise ValueError(
                        "record keys must match when using both blocks in and blocks out"
                    )
                if set(blocks_in.content_keys) != set(blocks_out.content_keys):
                    raise ValueError(
                        "content keys must match when using both blocks in and blocks out"
                    )
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out
        if self.blocks_in is not None:
            self.columns_needed = self.blocks_in.block_columns
        else:
            self.columns_needed = self.blocks_out.row_columns
        if self.blocks_out is not None:
            self.columns_produced = self.blocks_out.block_columns
        else:
            self.columns_produced = self.blocks_in.row_columns
        self.fmt_string = self.fmt()

    def record_keys(self):
        if self.blocks_in is not None:
            return self.blocks_in.record_keys.copy()
        if self.blocks_out is not None:
            return self.blocks_out.record_keys.copy()
        return None

    def example_input(self):
        if self.blocks_in is not None:
            example = self.blocks_in.control_table.copy()
            nrow = example.shape[0]
            for rk in self.blocks_in.record_keys:
                example[rk] = [rk] * nrow
            return example
        if self.blocks_in is not None:
            example = pandas.DataFrame()
            for k in self.blocks_out.row_columns:
                example[k] = [k]
            return example
        return None

    # noinspection PyPep8Naming
    def transform(
        self, X, *, check_blocks_in_keying=True, check_blocks_out_keying=False
    ):
        X = data_algebra.data_types.convert_to_pandas_dataframe(X, "X")
        unknown = set(self.columns_needed) - set(X.columns)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        X = X.reset_index(drop=True)
        db_model = data_algebra.SQLite.SQLiteModel()
        if self.blocks_in is not None:
            x1_descr = data_algebra.data_ops.describe_table(X, table_name="x_blocks_in")
            x1_sql = x1_descr.to_sql(db_model)
            missing_cols = set(self.blocks_in.control_table_keys).union(
                self.blocks_in.record_keys
            ) - set(x1_descr.column_names)
            if len(missing_cols) > 0:
                raise KeyError("missing required columns: " + str(missing_cols))
            # convert to row-records
            if check_blocks_in_keying:
                # table should be keyed by record_keys + control_table_keys
                if not data_algebra.cdata.table_is_keyed_by_columns(
                    X, self.blocks_in.record_keys + self.blocks_in.control_table_keys
                ):
                    raise ValueError(
                        "table is not keyed by blocks_in.record_keys + blocks_in.control_table_keys"
                    )
            with sqlite3.connect(":memory:") as conn:
                db_model.insert_table(conn, X, x1_descr.table_name)
                to_blocks_sql = db_model.blocks_to_row_recs_query(
                    x1_sql, self.blocks_in
                )
                X = db_model.read_query(conn, to_blocks_sql)
        if self.blocks_out is not None:
            x2_descr = data_algebra.data_ops.describe_table(
                X, table_name="x_blocks_out"
            )
            x2_sql = x2_descr.to_sql(db_model)
            missing_cols = set(self.blocks_out.record_keys) - set(x2_descr.column_names)
            if len(missing_cols) > 0:
                raise KeyError("missing required columns: " + str(missing_cols))
            if check_blocks_out_keying:
                # table should be keyed by record_keys
                if not data_algebra.cdata.table_is_keyed_by_columns(
                    X, self.blocks_out.record_keys
                ):
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
                    x2_sql, self.blocks_out, temp_table
                )
                X = db_model.read_query(conn, to_rows_sql)
        return X

    def compose(self, other):
        """
        Experimental method to compose transforms
        (self.compose(other)).transform(data) == self.transform(other.transform(data))

        :param other: another data_algebra.cdata_impl.RecordMap
        :return:
        """

        if not isinstance(other, RecordMap):
            raise TypeError("expected other to be data_algebra.cdata_impl.RecordMap")
        # (s2.compose(s1)).transform(data) == s2.transform(s1.transform(data))
        s1 = other
        s2 = self
        rk = s1.record_keys()
        if set(rk) != set(s2.record_keys()):
            raise ValueError("can only compose operations with matching record_keys")
        inp = s1.example_input()
        out = s2.transform(s1.transform(inp))
        rsi = inp.drop(rk, axis=1, inplace=False)
        rso = out.drop(rk, axis=1, inplace=False)
        if inp.shape[0] < 2:
            if out.shape[0] < 2:
                return None
            else:
                return RecordMap(
                    blocks_out=data_algebra.cdata.RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                    )
                )
        else:
            if out.shape[0] < 2:
                return RecordMap(
                    blocks_in=data_algebra.cdata.RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                    )
                )
            else:
                return RecordMap(
                    blocks_in=data_algebra.cdata.RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                    ),
                    blocks_out=data_algebra.cdata.RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                    ),
                )

    def __rrshift__(self, other):  # override other >> self
        if other is None:
            return self
        if isinstance(other, RecordMap):
            # (data >> other) >> self == data >> (other >> self)
            return self.compose(other)
        return self.transform(other)

    def inverse(self):
        return RecordMap(blocks_in=self.blocks_out, blocks_out=self.blocks_in)

    def fmt(self):
        if (self.blocks_in is None) and (self.blocks_out is None):
            return "RecordMap(no-op)"
        if (self.blocks_in is not None) and (self.blocks_out is not None):
            s = (
                "Transform block records of structure:\n"
                + str(self.blocks_in)
                + "to block records of structure:\n"
                + str(self.blocks_out)
            )
            return s
        if self.blocks_in is not None:
            s = (
                "Transform block records of structure:\n"
                + str(self.blocks_in)
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
                + str(self.blocks_out)
            )
            return s

    def __repr__(self):
        s = (
            "data_algebra.cdata_impl.RecordMap("
            + "\n    blocks_in="
            + self.blocks_in.__repr__()
            + ",\n    blocks_out="
            + self.blocks_out.__repr__()
            + ")"
        )
        return s

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
