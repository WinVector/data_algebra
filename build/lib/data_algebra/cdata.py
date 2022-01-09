"""
Class for representing record structure transformations.
"""


import re
from typing import Iterable, List, Optional

import numpy

import data_algebra
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
            local_data_model = data_algebra.default_data_model
        control_table = control_table.reset_index(inplace=False, drop=True)
        if control_table.shape[0] < 1:
            raise ValueError("control table should have at least 1 row")
        if len(control_table.columns) != len(set(control_table.columns)):
            raise ValueError("control table columns should be unique")
        self.control_table = control_table.reset_index(drop=True, inplace=False)
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


def blocks_to_rowrecs(data, *, blocks_in: RecordSpecification, local_data_model=None):
    """
    Convert a block record (record spanning multiple rows) into a rowrecord (record in a single row).

    :param data: data frame to be transformed
    :param blocks_in: record specification
    :param local_data_model: pandas model.
    :return: transformed data frame
    """
    assert isinstance(blocks_in, RecordSpecification)
    ck = [k for k in blocks_in.content_keys if k is not None]
    if len(ck) != len(set(ck)):
        raise ValueError("blocks_in can not have duplicate content keys")
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    data = data.reset_index(drop=True)
    missing_cols = set(blocks_in.control_table_keys).union(blocks_in.record_keys) - set(
        data.columns
    )
    if len(missing_cols) > 0:
        raise KeyError("missing required columns: " + str(missing_cols))
    # table must be keyed by record_keys + control_table_keys
    if not data_algebra.util.table_is_keyed_by_columns(
        data, blocks_in.record_keys + blocks_in.control_table_keys
    ):
        raise ValueError(
            "table is not keyed by blocks_in.record_keys + blocks_in.control_table_keys"
        )
    # convert to row-records
    # regularize/complete records
    dtemp = data.copy()  # TODO: select down columns
    dtemp["FALSE_AGG_KEY"] = 1
    if len(blocks_in.record_keys) > 0:
        ideal = dtemp[blocks_in.record_keys + ["FALSE_AGG_KEY"]].copy()
        res = ideal.groupby(blocks_in.record_keys)["FALSE_AGG_KEY"].agg("sum")
        ideal = local_data_model.data_frame(res).reset_index(drop=False)
        ideal["FALSE_AGG_KEY"] = 1
        ctemp = blocks_in.control_table[blocks_in.control_table_keys].copy()
        ctemp["FALSE_AGG_KEY"] = 1
        ideal = ideal.merge(ctemp, how="outer", on="FALSE_AGG_KEY")
        ideal = ideal.reset_index(drop=True)
        dtemp = ideal.merge(
            right=dtemp,
            how="left",
            on=blocks_in.record_keys + blocks_in.control_table_keys + ["FALSE_AGG_KEY"],
        )
    dtemp.sort_values(
        by=blocks_in.record_keys + blocks_in.control_table_keys, inplace=True
    )
    dtemp = dtemp.reset_index(drop=True)
    # start building up result frame
    if len(blocks_in.record_keys) > 0:
        res = dtemp.groupby(blocks_in.record_keys)["FALSE_AGG_KEY"].agg("sum")
    else:
        res = dtemp.groupby("FALSE_AGG_KEY")["FALSE_AGG_KEY"].agg("sum")
    res = local_data_model.data_frame(res).reset_index(drop=False)
    res.sort_values(by=blocks_in.record_keys, inplace=True)
    res = local_data_model.data_frame(res).reset_index(drop=True)
    del res["FALSE_AGG_KEY"]
    # now fill in columns
    ckeys = blocks_in.control_table_keys
    value_keys = [k for k in blocks_in.control_table.columns if k not in set(ckeys)]
    donor_cols = set(dtemp.columns)
    for i in range(blocks_in.control_table.shape[0]):
        want = numpy.ones((dtemp.shape[0],), dtype=bool)
        for ck in ckeys:
            want = numpy.logical_and(want, dtemp[ck] == blocks_in.control_table[ck][i])
        if numpy.any(want):
            for vk in value_keys:
                if vk in donor_cols:
                    dcol = blocks_in.control_table[vk][i]
                    res[dcol] = numpy.asarray(dtemp.loc[want, vk])
    # fill in any missed columns
    colset = set(res.columns)
    for c in blocks_in.row_version():
        if c not in colset:
            res[c] = None
    if data.shape[0] <= 0:
        res = res.loc[range(0), :]
        res = res.reset_index(inplace=False, drop=True)
    return res


def rowrecs_to_blocks(
    data,
    *,
    blocks_out: RecordSpecification,
    check_blocks_out_keying: bool = False,
    local_data_model=None
):
    """
    Convert rowrecs (single row records) into block records (multiple row records).

    :param data: data frame to transform.
    :param blocks_out: record specification.
    :param check_blocks_out_keying: logical, if True confirm keying
    :param local_data_model: pandas data model
    :return: transformed data frame
    """
    assert isinstance(blocks_out, RecordSpecification)
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    data = data.reset_index(drop=True)
    missing_cols = set(blocks_out.record_keys) - set(data.columns)
    if len(missing_cols) > 0:
        raise KeyError("missing required columns: " + str(missing_cols))
    if check_blocks_out_keying:
        # prefer table be keyed by record_keys
        if not data_algebra.util.table_is_keyed_by_columns(
            data, blocks_out.record_keys
        ):
            raise ValueError("table is not keyed by blocks_out.record_keys")
    # convert to block records, first build up parallel structures
    rv = [k for k in blocks_out.row_version(include_record_keys=True) if k is not None]
    if len(rv) != len(set(rv)):
        raise ValueError("duplicate row columns")
    dtemp_cols = [
        k
        for k in rv
        if k is not None and k in set(blocks_out.record_keys + blocks_out.content_keys)
    ]
    dtemp = data[dtemp_cols].copy()
    dtemp.sort_values(by=blocks_out.record_keys, inplace=True)
    dtemp = dtemp.reset_index(drop=True)
    if len(dtemp.columns) != len(set(dtemp.columns)):
        raise ValueError("targeted data columns not unique")
    ctemp = blocks_out.control_table.copy()
    dtemp["FALSE_JOIN_KEY"] = 1
    ctemp["FALSE_JOIN_KEY"] = 1
    res = dtemp[blocks_out.record_keys + ["FALSE_JOIN_KEY"]].merge(
        ctemp, how="outer", on=["FALSE_JOIN_KEY"]
    )
    del res["FALSE_JOIN_KEY"]
    ckeys = blocks_out.control_table_keys
    res.sort_values(by=blocks_out.record_keys + ckeys, inplace=True)
    res = res.reset_index(drop=True)
    del ctemp["FALSE_JOIN_KEY"]
    del dtemp["FALSE_JOIN_KEY"]
    value_keys = [k for k in ctemp.columns if k not in set(ckeys)]
    donor_cols = set(dtemp.columns)
    for vk in value_keys:
        res[vk] = None
    # we now have parallel structures to copy between
    for i in range(ctemp.shape[0]):
        want = numpy.ones((res.shape[0],), dtype=bool)
        for ck in ckeys:
            want = numpy.logical_and(want, res[ck] == ctemp[ck][i])
        if numpy.any(want):
            for vk in value_keys:
                dcol = ctemp[vk][i]
                if dcol in donor_cols:
                    nvals = numpy.asarray(dtemp[dcol])
                    if len(nvals) < 1:
                        nvals = [None] * numpy.sum(want)
                    res.loc[want, vk] = nvals
    # see about promoting composite columns to numeric
    for vk in set(value_keys):
        converted = local_data_model.to_numeric(res[vk], errors="coerce")
        if numpy.all(
            local_data_model.isnull(converted) == local_data_model.isnull(res[vk])
        ):
            res[vk] = converted
    if data.shape[0] < 1:
        # empty input produces emtpy output (with different column structure)
        res = res.iloc[range(0), :].reset_index(drop=True)
    if data.shape[0] <= 0:
        res = res.loc[range(0), :]
        res = res.reset_index(inplace=False, drop=True)
    return res


class RecordMap:
    """
    Class for specifying general record to record transforms.
    """
    def __init__(
            self,
            *,
            blocks_in: Optional[RecordSpecification] = None,
            blocks_out: Optional[RecordSpecification] = None):
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
            local_data_model = data_algebra.default_data_model
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
            self,
            X,
            *,
            check_blocks_out_keying: bool = False,
            local_data_model=None):
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
            local_data_model = data_algebra.default_data_model
        X = X.reset_index(drop=True)
        if self.blocks_in is not None:
            X = blocks_to_rowrecs(
                X, blocks_in=self.blocks_in, local_data_model=local_data_model
            )
        if self.blocks_out is not None:
            X = rowrecs_to_blocks(
                X,
                blocks_out=self.blocks_out,
                check_blocks_out_keying=check_blocks_out_keying,
                local_data_model=local_data_model,
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

    # noinspection PyTypeChecker
    def __rrshift__(self, other):  # override other >> self
        if other is None:
            return self
        if isinstance(other, RecordMap):
            # (data >> other) >> self == data >> (other >> self)
            return self.compose(other)
        return self.transform(other)

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
        local_data_model = data_algebra.default_data_model
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
        local_data_model = data_algebra.default_data_model
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
        col_name_key: str = 'column_name',
        col_value_key: str = 'column_value',
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
            control_table=data_algebra.pandas_model.pd.DataFrame({
                col_name_key: value_cols,
                col_value_key: value_cols,
            }),
            record_keys=row_keys,
            control_table_keys=[col_name_key])
        )
    return record_map


def unpivot_specification(
        *,
        row_keys: Iterable[str],
        col_name_key: str = 'column_name',
        col_value_key: str = 'column_value',
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
            control_table=data_algebra.pandas_model.pd.DataFrame({
                col_name_key: value_cols,
                col_value_key: value_cols,
            }),
            record_keys=row_keys,
            control_table_keys=[col_name_key])
        )
    return record_map
