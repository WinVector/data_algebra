import re
import collections

import sqlite3

import pandas

import data_algebra.util
import data_algebra.data_types
import data_algebra.SQLite
import data_algebra.data_ops


class RecordSpecification:
    def __init__(
        self, control_table, *, record_keys=None, control_table_keys=None, strict=False
    ):
        control_table = data_algebra.data_types.convert_to_pandas_dataframe(
            control_table, "control_table"
        )
        control_table = control_table.reset_index(inplace=False, drop=True)
        if control_table.shape[0] < 1:
            raise ValueError("control table should have at least 1 row")
        self.control_table = control_table.reset_index(drop=True)
        if record_keys is None:
            record_keys = []
        if isinstance(record_keys, str):
            record_keys = [record_keys]
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            control_table_keys = [control_table.columns[0]]
        if isinstance(control_table_keys, str):
            record_keys = [control_table_keys]
        self.control_table_keys = [k for k in control_table_keys]
        unknown = set(self.control_table_keys) - set(control_table.columns)
        if len(unknown) > 0:
            raise ValueError(
                "control table keys that are not in the control table: " + str(unknown)
            )
        confused = set(record_keys).intersection(control_table_keys)
        if len(confused) > 0:
            raise ValueError(
                "columns common to record_keys and control_table_keys: " + str(confused)
            )
        if strict:
            if not data_algebra.util.table_is_keyed_by_columns(
                self.control_table, self.control_table_keys
            ):
                raise ValueError("control table wasn't keyed by control table keys")
        self.block_columns = self.record_keys + [c for c in self.control_table.columns]
        cvs = []
        for c in self.control_table:
            if c not in self.control_table_keys:
                col = self.control_table[c]
                isnull = col.isnull()
                if all(isnull):
                    raise ValueError("column " + c + " was all null")
                for i in range(len(col)):
                    if not isnull[i]:
                        v = col[i]
                        if v not in cvs:
                            cvs.append(v)
        confused = set(record_keys).intersection(cvs)
        if len(confused) > 0:
            raise ValueError(
                "control table entries confused with row keys or control table keys"
            )
        if strict:
            if len(set(cvs)) != len(cvs):
                raise ValueError("duplicate content keys")
        self.content_keys = cvs
        self.row_columns = self.record_keys + cvs

    def row_version(self, *, include_record_keys=True):
        cols = []
        if include_record_keys:
            cols = cols + self.record_keys
        cols = cols + self.content_keys
        return cols

    def __repr__(self):
        s = (
            "data_algebra.cdata.RecordSpecification(\n"
            + "    record_keys="
            + self.record_keys.__repr__()
            + ",\n    control_table="
            + data_algebra.util.pandas_to_example_str(self.control_table)
            + ",\n    control_table_keys="
            + self.control_table_keys.__repr__()
            + ")"
        )
        return s

    def fmt(self):
        s = (
            "RecordSpecification\n"
            + "   record_keys: "
            + str(self.record_keys)
            + "\n"
            + "   control_table_keys: "
            + str(self.control_table_keys)
            + "\n"
            + "   control_table:\n"
            + "   "
            + re.sub("\n", "\n   ", str(self.control_table))
            + "\n"
        )
        return s

    def __str__(self):
        return self.fmt()

    def to_simple_obj(self):
        """Create an object for YAML encoding"""

        obj = collections.OrderedDict()
        obj["type"] = "data_algebra.cdata.RecordSpecification"
        obj["record_keys"] = self.record_keys.copy()
        obj["control_table_keys"] = self.control_table_keys.copy()
        tbl = collections.OrderedDict()
        for k in self.control_table.columns:
            tbl[k] = [v for v in self.control_table[k]]
        obj["control_table"] = tbl
        return obj


def record_spec_from_simple_obj(obj):
    control_table = pandas.DataFrame()
    for k in obj["control_table"].keys():
        control_table[k] = obj["control_table"][k]

    def maybe_get_list(omap, key):
        try:
            v = omap[key]
            if v is None:
                return []
            if isinstance(v, str):
                v = [v]
            return v
        except KeyError:
            return []

    return RecordSpecification(
        control_table,
        record_keys=maybe_get_list(obj, "record_keys"),
        control_table_keys=maybe_get_list(obj, "control_table_keys"),
    )


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
                if not data_algebra.util.table_is_keyed_by_columns(
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
                if not data_algebra.util.table_is_keyed_by_columns(
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

        :param other: another data_algebra.cdata.RecordMap
        :return:
        """

        if not isinstance(other, RecordMap):
            raise TypeError("expected other to be data_algebra.cdata.RecordMap")
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
                + str(self.blocks_in.row_version(include_record_keys=False))
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
                + str(self.blocks_out.row_version(include_record_keys=False))
                + "\n"
                + "to block records of structure:\n"
                + str(self.blocks_out)
            )
            return s

    def __repr__(self):
        s = (
            "data_algebra.cdata.RecordMap("
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
        obj["type"] = "data_algebra.cdata.RecordMap"
        if self.blocks_in is not None:
            obj["blocks_in"] = self.blocks_in.to_simple_obj()
        if self.blocks_out is not None:
            obj["blocks_out"] = self.blocks_out.to_simple_obj()
        return obj
