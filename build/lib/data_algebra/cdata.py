"""
Class for representing record structure transformations.
"""


import re
from typing import Iterable, List, Optional

import data_algebra.data_model
import data_algebra.util


class RecordSpecification:
    """
    Class to represent a multi-row data record.
    """

    def __init__(
        self,
        control_table,
        *,
        record_keys=None,
        control_table_keys=None,
        strict=False,
        local_data_model=None
    ):
        """
        :param control_table: data.frame describing record layout
        :param record_keys: array of record key column names
               defaults to no columns.
        :param control_table_keys: array of control_table key column names,
               defaults to first column for non-trivial blocks and no columns for rows.
        :param strict: logical, if True more checks on transform
        :param local_data_model: data.frame data model
        """
        if local_data_model is None:
            local_data_model = data_algebra.data_model.default_data_model()
        assert isinstance(local_data_model, data_algebra.data_model.DataModel)
        assert local_data_model.is_appropriate_data_instance(control_table)
        control_table = local_data_model.clean_copy(control_table)
        if control_table.shape[0] < 1:
            raise ValueError("control table should have at least 1 row")
        if len(control_table.columns) != len(set(control_table.columns)):
            raise ValueError("control table columns should be unique")
        self.control_table = control_table
        assert self.control_table.shape[0] > 0
        if record_keys is None:
            record_keys = []
        if isinstance(record_keys, str):
            record_keys = [record_keys]
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            if self.control_table.shape[0] > 1:
                control_table_keys = [self.control_table.columns[0]]
            else:
                control_table_keys = []  # single row records don't need to be keyed
        if isinstance(control_table_keys, str):
            control_table_keys = [control_table_keys]
        if strict and (self.control_table.shape[0] > 1):
            if len(control_table_keys) <= 0:
                raise ValueError(
                    "multi-row records must have at least one control table key"
                )
        self.control_table_keys = [k for k in control_table_keys]
        unknown = set(self.control_table_keys) - set(control_table.columns)
        if len(unknown) > 0:
            raise ValueError(
                "control table keys that are not in the control table: " + str(unknown)
            )
        if len(self.control_table_keys) >= control_table.shape[1]:
            raise ValueError("control table columns must not all be keys")
        confused = set(record_keys).intersection(control_table_keys)
        if len(confused) > 0:
            raise ValueError(
                "columns common to record_keys and control_table_keys: " + str(confused)
            )
        for ck in self.control_table_keys:
            if any(local_data_model.bad_column_positions(control_table[ck])):
                raise ValueError("NA/NaN/inf/None not allowed as control table keys")
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

    def row_version(self, *, include_record_keys: bool = True) -> List[str]:
        """
        Return copy of record as a row record.

        :param include_record_keys: logical, if True include record keys as columns
        :return: column list
        """
        cols: List[str] = []
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

    def __eq__(self, other):
        if not isinstance(other, RecordSpecification):
            return False
        return self.__repr__() == other.__repr__()

    def fmt(self):
        """
        Prepare for printing

        :return: multi line string representation.
        """
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

    def map_to_rows(self):
        """
        Build a RecordMap mapping this RecordSpecification to rowrecs

        :return: RecordMap
        """

        return RecordMap(blocks_in=self)

    def map_from_rows(self):
        """
        Build a RecordMap mapping this RecordSpecification from rowrecs

        :return: RecordMap
        """

        return RecordMap(blocks_out=self)


class RecordMap:
    """
    Class for specifying general record to record transforms.
    """

    def __init__(
        self,
        *,
        blocks_in: Optional[RecordSpecification] = None,
        blocks_out: Optional[RecordSpecification] = None
    ):
        """
        Build the transform specification. At least one of blocks_in or blocks_out must not be None.

        :param blocks_in: incoming record specification, None for row-records.
        :param blocks_out: outgoing record specification, None for row-records.
        """
        if blocks_in is not None:
            assert isinstance(blocks_in, RecordSpecification)
            ck = [k for k in blocks_in.content_keys if k is not None]
            if len(ck) != len(set(ck)):
                raise ValueError("blocks_in can not have duplicate content keys")
        if blocks_out is not None:
            assert isinstance(blocks_out, RecordSpecification)
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
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out
        if self.blocks_in is not None:
            self.columns_needed = self.blocks_in.block_columns
        else:
            assert self.blocks_out is not None  # type hint
            self.columns_needed = self.blocks_out.row_columns
        if self.blocks_out is not None:
            self.columns_produced = self.blocks_out.block_columns
        else:
            assert self.blocks_in is not None  # type hint
            self.columns_produced = self.blocks_in.row_columns
        self.fmt_string = self.fmt()

    def __eq__(self, other):
        if not isinstance(other, RecordMap):
            return False
        if (self.blocks_in is None) != (other.blocks_in is None):
            return False
        if (self.blocks_out is None) != (other.blocks_out is None):
            return False
        if self.blocks_in is not None:
            if self.blocks_in != other.blocks_in:
                return False
        if self.blocks_in is not None:
            if self.blocks_out != other.blocks_out:
                return False
        return True

    def record_keys(self):
        """Return keys specifying which set of rows are in a record."""
        if self.blocks_in is not None:
            return self.blocks_in.record_keys.copy()
        if self.blocks_out is not None:
            return self.blocks_out.record_keys.copy()
        return None

    def example_input(self, *, local_data_model=None):
        """
        Return example output record.

        :param local_data_model: optional Pandas data model.
        :return: example result data frame.
        """
        if local_data_model is None:
            local_data_model = data_algebra.data_model.default_data_model()
        assert isinstance(local_data_model, data_algebra.data_model.DataModel)
        if self.blocks_in is not None:
            example = self.blocks_in.control_table.copy()
            nrow = example.shape[0]
            for rk in self.blocks_in.record_keys:
                example[rk] = [rk] * nrow
            return example
        if self.blocks_in is not None:
            example = local_data_model.data_frame()
            for k in self.blocks_out.row_columns:
                example[k] = [k]
            return example
        return None

    # noinspection PyPep8Naming
    def transform(
        self, X, *, check_blocks_out_keying: bool = False, local_data_model=None
    ):
        """
        Transform X records.

        :param X: data frame to be transformed.
        :param check_blocks_out_keying: logical, if True check output key constraints.
        :param local_data_model: pandas data model.
        :return: transformed data frame.
        """
        unknown = set(self.columns_needed) - set(X.columns)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        if local_data_model is None:
            local_data_model = data_algebra.data_model.default_data_model()
        assert isinstance(local_data_model, data_algebra.data_model.DataModel)
        assert local_data_model.is_appropriate_data_instance(X)
        X = local_data_model.clean_copy(X)
        if self.blocks_in is not None:
            X = local_data_model.blocks_to_rowrecs(
                X, blocks_in=self.blocks_in
            )
        if self.blocks_out is not None:
            X = local_data_model.rowrecs_to_blocks(
                X,
                blocks_out=self.blocks_out,
                check_blocks_out_keying=check_blocks_out_keying,
            )
        return X

    def compose(self, other):
        """
        Experimental method to compose transforms
        (self.compose(other)).transform(data) == self.transform(other.transform(data))

        :param other: another data_algebra.cdata.RecordMap
        :return:
        """

        assert isinstance(other, RecordMap)
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
                    blocks_out=RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                    )
                )
        else:
            if out.shape[0] < 2:
                return RecordMap(
                    blocks_in=RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                    )
                )
            else:
                return RecordMap(
                    blocks_in=RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                    ),
                    blocks_out=RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                    ),
                )

    def inverse(self):
        """
        Return inverse transform.
        """
        return RecordMap(blocks_in=self.blocks_out, blocks_out=self.blocks_in)

    def fmt(self) -> str:
        """Format for informal presentation."""
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
        if self.blocks_out is not None:
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
        raise ValueError("should not be reached")

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

    # more of the sklearn step API

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        """No-op (sklearn pipeline interface)"""
        pass

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        """transform() (sklearn pipeline interface)"""
        return self.transform(X)

    # noinspection PyUnusedLocal
    def get_feature_names(self, input_features=None):
        """Return columns produced (sklearn pipeline interface)"""
        return self.columns_produced.copy()

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        """Return emtpy dictionary (sklearn pipeline interface)"""
        return dict()

    def set_params(self, **params):
        """No-op (sklearn pipeline interface)"""
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        """Perform inverse transform (sklearn pipeline interface)"""
        return self.inverse().transform(X)


def pivot_blocks_to_rowrecs(
    *,
    attribute_key_column,
    attribute_value_column,
    record_keys,
    record_value_columns,
    local_data_model=None
):
    """
    Build a block records to row records map. This is very similar to a SQL pivot.

    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param local_data_model: data.frame data model
    :return: RecordMap
    """

    if local_data_model is None:
        local_data_model = data_algebra.data_model.default_data_model()
    assert isinstance(local_data_model, data_algebra.data_model.DataModel)
    control_table = local_data_model.data_frame(
        {
            attribute_key_column: record_value_columns,
            attribute_value_column: record_value_columns,
        }
    )
    ct = RecordSpecification(
        control_table,
        record_keys=record_keys,
        control_table_keys=[attribute_key_column],
        local_data_model=local_data_model,
    )
    return ct.map_to_rows()


def pivot_rowrecs_to_blocks(
    *,
    attribute_key_column,
    attribute_value_column,
    record_keys,
    record_value_columns,
    local_data_model=None
):
    """
    Build a row records to block records map. This is very similar to a SQL unpivot.

    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param local_data_model: data.frame data model
    :return: RecordMap
    """

    if local_data_model is None:
        local_data_model = data_algebra.data_model.default_data_model()
    assert isinstance(local_data_model, data_algebra.data_model.DataModel)
    control_table = local_data_model.data_frame(
        {
            attribute_key_column: record_value_columns,
            attribute_value_column: record_value_columns,
        }
    )
    ct = RecordSpecification(
        control_table,
        record_keys=record_keys,
        control_table_keys=[attribute_key_column],
        local_data_model=local_data_model,
    )
    return ct.map_from_rows()


def pivot_specification(
    *,
    row_keys: Iterable[str],
    col_name_key: str = "column_name",
    col_value_key: str = "column_value",
    value_cols: Iterable[str]
) -> RecordMap:
    """
    Specify the cdata transformation that pivots records from a single column of values into collected rows.
    https://en.wikipedia.org/wiki/Pivot_table#History

    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to take the names of columns as a column
    :param col_value_key: column name to take the values in columns as a column
    :param value_cols: columns to place values in
    :return: RecordSpecification
    """
    assert not isinstance(value_cols, str)
    value_cols = list(value_cols)
    assert not isinstance(row_keys, str)
    row_keys = list(row_keys)
    assert isinstance(col_name_key, str)
    assert isinstance(col_value_key, str)
    known_cols = row_keys + [col_name_key, col_value_key] + value_cols
    assert len(known_cols) == len(set(known_cols))
    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=data_algebra.data_model.default_data_model().pd.DataFrame(
                {
                    col_name_key: value_cols,
                    col_value_key: value_cols,
                }
            ),
            record_keys=row_keys,
            control_table_keys=[col_name_key],
        )
    )
    return record_map


def unpivot_specification(
    *,
    row_keys: Iterable[str],
    col_name_key: str = "column_name",
    col_value_key: str = "column_value",
    value_cols: Iterable[str]
) -> RecordMap:
    """
    Specify the cdata transformation that un-pivots records into a single column of values plus keys.
    https://en.wikipedia.org/wiki/Pivot_table#History

    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to land the names of columns as a column
    :param col_value_key: column name to land the values in columns as a column
    :param value_cols: columns to take values from
    :return: RecordSpecification
    """
    assert not isinstance(value_cols, str)
    value_cols = list(value_cols)
    assert not isinstance(row_keys, str)
    row_keys = list(row_keys)
    assert isinstance(col_name_key, str)
    assert isinstance(col_value_key, str)
    known_cols = row_keys + [col_name_key, col_value_key] + value_cols
    assert len(known_cols) == len(set(known_cols))
    record_map = RecordMap(
        blocks_out=RecordSpecification(
            control_table=data_algebra.data_model.default_data_model().pd.DataFrame(
                {
                    col_name_key: value_cols,
                    col_value_key: value_cols,
                }
            ),
            record_keys=row_keys,
            control_table_keys=[col_name_key],
        )
    )
    return record_map
