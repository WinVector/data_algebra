import re
import collections

import numpy

import data_algebra
import data_algebra.util


class RecordSpecification:
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
        Class to represent a multi-row data record.

        :param control_table: data.frame describing record layout
        :param record_keys: array of record key column names
        :param control_table_keys: array of control_table key column names
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
        self.control_table = control_table.reset_index(drop=True)
        if record_keys is None:
            record_keys = []
        if isinstance(record_keys, str):
            record_keys = [record_keys]
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            control_table_keys = [control_table.columns[0]]
        if len(control_table_keys) <= 0:
            raise ValueError("must have at least one control table key")
        if isinstance(control_table_keys, str):
            control_table_keys = [control_table_keys]
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

    def map_to_rows(self, *, strict=False):
        """
        Build a RecordMap mapping this RecordSpecification to rowrecs

        :param strict:
        :return: RecordMap
        """

        return RecordMap(blocks_in=self, strict=strict)

    def map_from_rows(self, *, strict=False):
        """
        Build a RecordMap mapping this RecordSpecification from rowrecs

        :param strict:
        :return: RecordMap
        """

        return RecordMap(blocks_out=self, strict=strict)


def blocks_to_rowrecs(data, *, blocks_in, local_data_model=None):
    if not isinstance(blocks_in, data_algebra.cdata.RecordSpecification):
        raise TypeError("blocks_in should be a data_algebra.cdata.RecordSpecification")
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
    return res


def rowrecs_to_blocks(
    data, *, blocks_out, check_blocks_out_keying=False, local_data_model=None
):
    if not isinstance(blocks_out, data_algebra.cdata.RecordSpecification):
        raise TypeError("blocks_out should be a data_algebra.cdata.RecordSpecification")
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
                    res.loc[want, vk] = numpy.asarray(dtemp[dcol])
    # see about promoting composite columns to numeric
    for vk in set(value_keys):
        converted = local_data_model.to_numeric(res[vk], errors="coerce")
        if numpy.all(
            local_data_model.isnull(converted) == local_data_model.isnull(res[vk])
        ):
            res[vk] = converted
    return res


class RecordMap:
    def __init__(self, *, blocks_in=None, blocks_out=None, strict=False):
        if blocks_in is not None:
            if not isinstance(blocks_in, data_algebra.cdata.RecordSpecification):
                raise TypeError(
                    "blocks_in should be a data_algebra.cdata.RecordSpecification"
                )
            ck = [k for k in blocks_in.content_keys if k is not None]
            if len(ck) != len(set(ck)):
                raise ValueError("blocks_in can not have duplicate content keys")
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
        if self.blocks_in is not None:
            return self.blocks_in.record_keys.copy()
        if self.blocks_out is not None:
            return self.blocks_out.record_keys.copy()
        return None

    def example_input(self, *, local_data_model=None):
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
    def transform(self, X, *, check_blocks_out_keying=False, local_data_model=None):
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

    # noinspection PyTypeChecker
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

    # more of the sklearn step API

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        pass

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        return self.transform(X)

    # noinspection PyUnusedLocal
    def get_feature_names(self, input_features=None):
        return self.columns_produced.copy()

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        return dict()

    def set_params(self, **params):
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        return self.inverse().transform(X)


def pivot_blocks_to_rowrecs(
    *,
    attribute_key_column,
    attribute_value_column,
    record_keys,
    record_value_columns,
    strict=False,
    local_data_model=None
):
    """
    Build a block records to row records map.

    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param strict: logical, if True more checks on transform
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
        strict=strict,
        local_data_model=local_data_model,
    )
    return ct.map_to_rows(strict=strict)


def pivot_rowrecs_to_blocks(
    *,
    attribute_key_column,
    attribute_value_column,
    record_keys,
    record_value_columns,
    strict=False,
    local_data_model=None
):
    """
    Build a row records to block records map.

    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param strict: logical, if True more checks on transform
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
        strict=strict,
        local_data_model=local_data_model,
    )
    return ct.map_from_rows(strict=strict)
