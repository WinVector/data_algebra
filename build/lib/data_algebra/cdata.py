"""
Class for representing record structure transformations.
"""


import re
import html
from typing import Iterable, List, Optional

import data_algebra.data_model
import data_algebra.util
import data_algebra.data_ops
from data_algebra.shift_pipe_action import ShiftPipeAction


def _str_list_to_html(lst: Iterable[str]) -> str:
    return "[" + ", ".join([html.escape(k.__repr__()) for k in lst]) + "]"


def _format_table(
    d,
    *,
    record_id_cols: Iterable[str],
    control_id_cols: Iterable[str],
    add_style: bool = True,
) -> str:
    local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
    d = local_data_model.to_pandas(d)
    pd = data_algebra.data_model.lookup_data_model_for_dataframe(d).pd
    record_id_cols = list(record_id_cols)
    control_id_cols = list(control_id_cols)
    record_id_col_pairs = [
        ("record id", c) for c in d.columns if c in set(record_id_cols)
    ]
    control_id_col_pairs = [
        ("record structure", c) for c in d.columns if c in set(control_id_cols)
    ]
    value_id_col_pairs = [
        ("value", c)
        for c in d.columns
        if (c not in set(record_id_cols)) and (c not in set(control_id_cols))
    ]
    d = pd.DataFrame(
        {
            (cc, cn): d[cn]
            for (cc, cn) in record_id_col_pairs
            + control_id_col_pairs
            + value_id_col_pairs
        }
    )
    if add_style:
        d = d.style.set_properties(
            **{"background-color": "#FFE4C4"}, subset=record_id_col_pairs
        ).set_properties(**{"background-color": "#7FFFD4"}, subset=control_id_col_pairs)
    return d._repr_html_()


class RecordSpecification:
    """
    Class to represent a data record. 
    For single row data records use None as the specification.
    """

    row_columns: List[str]  # columns when in row form
    block_columns: List[str]  # columns when in block form
    strict: bool  # do or do not allow repeated value names

    def __init__(
        self,
        control_table,
        *,
        record_keys=None,
        control_table_keys=None,
        strict: bool = True,
        local_data_model=None,
    ):
        """
        :param control_table: data.frame describing record layout
        :param record_keys: array of record key column names
               defaults to no columns.
        :param control_table_keys: array of control_table key column names,
               defaults to first column for non-trivial blocks and no columns for rows.
        :param strict: if True don't allow duplicate value names
        :param local_data_model: data.frame data model
        """
        assert isinstance(strict, bool)
        self.strict = strict
        if local_data_model is None:
            local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(
                control_table
            )
        assert isinstance(local_data_model, data_algebra.data_model.DataModel)
        assert local_data_model.is_appropriate_data_instance(control_table)
        control_table = local_data_model.clean_copy(control_table)
        if control_table.shape[0] < 1:
            raise ValueError("control table should have at least 1 row")
        if control_table.shape[1] < 2:
            raise ValueError("control table must have at least 2 columns")
        if len(control_table.columns) != len(set(control_table.columns)):
            raise ValueError("control table columns should be unique")
        if control_table_keys is None:
            if control_table.shape[0] > 1:
                control_table_keys = [control_table.columns[0]]
            else:
                control_table_keys = []  # single row records don't need to be keyed
        if isinstance(control_table_keys, str):
            control_table_keys = [control_table_keys]
        else:
            control_table_keys = list(control_table_keys)
        assert isinstance(control_table_keys, List)
        if strict:
            if control_table.shape[0] > 1:
                assert len(control_table_keys) > 0
                assert local_data_model.table_is_keyed_by_columns(
                    control_table, column_names=control_table_keys
                )
        self.control_table = control_table
        assert self.control_table.shape[0] > 0
        if record_keys is None:
            record_keys = []
        elif isinstance(record_keys, str):
            record_keys = [record_keys]
        else:
            record_keys = list(record_keys)
        assert isinstance(record_keys, list)
        self.record_keys = record_keys
        if isinstance(control_table_keys, str):
            control_table_keys = [control_table_keys]
        if self.control_table.shape[0] > 1:
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
        if not local_data_model.table_is_keyed_by_columns(
            self.control_table, column_names=self.control_table_keys
        ):
            raise ValueError("control table wasn't keyed by control table keys")
        self.block_columns = self.record_keys + [c for c in self.control_table.columns]
        cvs = []
        for c in self.control_table.columns:
            if c not in self.control_table_keys:
                col = self.control_table[c]
                isnull = local_data_model.bad_column_positions(col)
                if any(isnull):
                    raise ValueError("column " + c + " has null(s)")
                for i in range(len(col)):
                    v = col[i]
                    assert isinstance(v, str)
                    assert len(v) > 0
                    cvs.append(v)
        confused = set(record_keys).intersection(cvs)
        if len(confused) > 0:
            raise ValueError(
                "control table entries confused with row keys or control table keys"
            )
        if len(set(cvs)) != len(cvs):
            if strict:
                raise ValueError("duplicate content keys")
            cvs_orig = cvs
            cvs = []
            cvs_seen = set()
            for v in cvs_orig:
                if v not in cvs_seen:
                    cvs.append(v)
                    cvs_seen.add(v)
        self.content_keys = cvs
        # columns in the row-record form
        self.row_columns = self.record_keys + cvs
        # columns in the block-record form
        self.block_columns = self.record_keys + list(self.control_table.columns)

    def row_version(self, *, include_record_keys: bool = True) -> List[str]:
        """
        Return copy of record as a row record.

        :param include_record_keys: logical, if True include record keys as columns
        :return: column list
        """
        assert isinstance(include_record_keys, bool)
        cols: List[str] = []
        if include_record_keys:
            cols = cols + self.record_keys
        cols = cols + self.content_keys
        return cols
    
    def row_record_form(self):
        """
        Return specification of matching row record form.
        Note: prefer using None to specify row records specs.
        """
        local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(self.control_table)
        row_vals = self.row_version(include_record_keys=False)
        ct = local_data_model.data_frame({
            k: [k] for k in row_vals
        })
        v_set = set(self.row_version(include_record_keys=False))
        return RecordSpecification(
            ct,
            record_keys=self.record_keys,
            control_table_keys=[],
            strict=self.strict,
            local_data_model=local_data_model,
        )

    def value_column_form(self, *, key_column_name: str = "measure", value_column_name: str = "value"):
        """
        Return specification of the matching value column form.
        Note: for type safety prefer map_to_rows() to map_to_keyed_column().

        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column
        """
        assert isinstance(key_column_name, str)
        assert isinstance(value_column_name, str)
        local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(self.control_table)
        ct = local_data_model.data_frame({
            key_column_name: self.row_version(include_record_keys=False),
            value_column_name: self.row_version(include_record_keys=False),
        })
        return RecordSpecification(
            ct,
            record_keys=self.record_keys,
            control_table_keys=[key_column_name],
            strict=self.strict,
            local_data_model=local_data_model,
        )
        
    def __repr__(self):
        s = (
            "data_algebra.cdata.RecordSpecification(\n"
            + "    record_keys="
            + self.record_keys.__repr__()
            + ",\n    control_table="
            + data_algebra.util.pandas_to_example_str(self.control_table)
            + ",\n    control_table_keys="
            + self.control_table_keys.__repr__()
            + ",\n    strict="
            + self.strict.__repr__()
            + ")"
        )
        return s

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

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        if cycle:
            p.text("RecordSpecification()")
        else:
            p.text(self.fmt())

    def _repr_html_(self):
        s = (
            "<p>RecordSpecification<br>\n"
            + "<ul>"
            + "<li>record_keys: "
            + _str_list_to_html(self.record_keys)
            + "</li>\n"
            + "<li>control_table_keys: "
            + _str_list_to_html(self.control_table_keys)
            + "</li>\n"
            + "<li>control_table:<br>\n"
            + _format_table(
                self.control_table,
                record_id_cols=self.record_keys,
                control_id_cols=self.control_table_keys,
                add_style=True,
            )
            + "</li>\n"
            + "</ul>"
            + "</p>\n"
        )
        return s

    def __eq__(self, other):
        if not isinstance(other, RecordSpecification):
            return False
        return self.__repr__() == other.__repr__()

    def map_to_rows(self):
        """
        Build a RecordMap mapping this RecordSpecification to rowrecs

        :return: RecordMap
        """
        if self.control_table.shape[0] <= 1:
            raise ValueError("already in row record format")
        return RecordMap(blocks_in=self, strict=self.strict)

    def map_from_rows(self):
        """
        Build a RecordMap mapping this RecordSpecification from rowrecs

        :return: RecordMap
        """
        if self.control_table.shape[0] <= 1:
            raise ValueError("already in row record format")
        return RecordMap(blocks_out=self, strict=self.strict)
    
    def map_to_keyed_column(self, *, key_column_name: str = "measure", value_column_name: str = "value"):
        """
        Build a RecordMap mapping this RecordSpecification to a table
        where only one column holds values.
        Note: for type safety prefer map_to_rows() to map_to_keyed_column().

        
        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column
        :return: Record map
        """
        return RecordMap(blocks_in=self, blocks_out=self.value_column_form(), strict=self.strict)
    
    def map_from_keyed_column(self, *, key_column_name: str = "measure", value_column_name: str = "value"):
        """
        Build a RecordMap mapping this RecordSpecification from a table
        where only one column holds values.

        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column
        :return: Record map
        """
        return RecordMap(blocks_in=self.value_column_form(), blocks_out=self, strict=self.strict)


class RecordMap(ShiftPipeAction):
    """
    Class for specifying general record to record transforms.
    """

    blocks_in: Optional[RecordSpecification]
    blocks_out: Optional[RecordSpecification]
    strict: bool

    def __init__(
        self,
        *,
        blocks_in: Optional[RecordSpecification] = None,
        blocks_out: Optional[RecordSpecification] = None,
        strict: bool = True,
    ):
        """
        Build the transform specification. At least one of blocks_in or blocks_out must not be None.

        :param blocks_in: incoming record specification, None for row-records.
        :param blocks_out: outgoing record specification, None for row-records.
        :param strict: if True insist block be strict, and in and out blocks agree on row-form columns.∂
        """
        ShiftPipeAction.__init__(self)
        assert isinstance(strict, bool)
        self.strict = strict
        if blocks_in is not None:
            assert isinstance(blocks_in, RecordSpecification)
            if strict:
                assert blocks_in.strict
            if blocks_in.control_table.shape[0] <= 1:
                blocks_in = None
        if blocks_in is not None:
            ck = [k for k in blocks_in.content_keys if k is not None]
            if len(ck) != len(set(ck)):
                raise ValueError("blocks_in can not have duplicate content keys")
            if blocks_in.control_table.shape[0] <= 1:
                raise ValueError("for row records use None specification")
        if blocks_out is not None:
            assert isinstance(blocks_out, RecordSpecification)
            if strict:
                assert blocks_out.strict
            if blocks_out.control_table.shape[0] <= 1:
                blocks_out = None
        if (blocks_in is None) and (blocks_out is None):
            raise ValueError(
                "At least one of blocks_in or blocks_out should not be None or a non-row record"
            )
        if (blocks_in is not None) and (blocks_out is not None):
            unknown = set(blocks_out.record_keys) - set(blocks_in.record_keys)
            if len(unknown) > 0:
                raise ValueError("unknown outgoing record_keys:" + str(unknown))
            unknown = set(blocks_out.content_keys) - set(blocks_in.content_keys)
            if len(unknown) > 0:
                raise ValueError("unknown outgoing content_keys" + str(unknown))
            if strict:
                assert set(blocks_out.record_keys) == set(blocks_in.record_keys)
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

    def record_keys(self) -> List[str]:
        """Return keys specifying which set of rows are in a record."""
        if self.blocks_in is not None:
            if self.blocks_in.record_keys is None:
                return []
            return list(self.blocks_in.record_keys)
        if self.blocks_out is not None:
            if self.blocks_out.record_keys is None:
                return []
            return list(self.blocks_out.record_keys)
        raise ValueError("null transform")

    def input_control_table_key_columns(self) -> List[str]:
        if (self.blocks_in is None) or (self.blocks_in.control_table_keys is None):
            return []
        return list(self.blocks_in.control_table_keys)

    def output_control_table_key_columns(self) -> List[str]:
        if (self.blocks_out is None) or (self.blocks_out.control_table_keys is None):
            return []
        return list(self.blocks_out.control_table_keys)

    def example_input(
        self,
        *,
        local_data_model=None,
        value_suffix: str = " value",
        record_key_suffix: str = " record key",
    ):
        """
        Return example output record.

        :param local_data_model: optional Pandas data model.
        :param value_suffix: suffix to identify values
        :param record_key_suffix: suffix to identify record keys
        :return: example result data frame.
        """
        assert isinstance(value_suffix, str)
        assert isinstance(record_key_suffix, str)
        rk = self.record_keys()
        if self.blocks_in is not None:
            if local_data_model is None:
                local_data_model = (
                    data_algebra.data_model.lookup_data_model_for_dataframe(
                        self.blocks_in.control_table
                    )
                )
            assert isinstance(local_data_model, data_algebra.data_model.DataModel)
            example = local_data_model.clean_copy(self.blocks_in.control_table)
            cols = list(example.columns)
            for j in range(example.shape[1]):
                if (self.blocks_in.control_table_keys is None) or (
                    cols[j] not in self.blocks_in.control_table_keys
                ):
                    renames = [
                        f"{local_data_model.get_cell(d=example, row=i, colname=cols[j])}{value_suffix}"
                        for i in range(example.shape[0])
                    ]
                    example = local_data_model.set_col(
                        d=example, colname=cols[j], values=renames
                    )
        elif self.blocks_out is not None:
            if local_data_model is None:
                local_data_model = (
                    data_algebra.data_model.lookup_data_model_for_dataframe(
                        self.blocks_out.control_table
                    )
                )
            assert isinstance(local_data_model, data_algebra.data_model.DataModel)
            example = local_data_model.data_frame(
                {
                    k: [f"{k}{value_suffix}"]
                    for k in self.blocks_out.row_columns
                    if k not in rk
                }
            )
        else:
            raise ValueError("null transform")
        if len(rk) > 0:
            nrow = example.shape[0]
            row_key_frame = local_data_model.data_frame(
                {k: [f"{k}{record_key_suffix}"] * nrow for k in rk}
            )
            example = local_data_model.concat_columns([row_key_frame, example])
        return example

    # noinspection PyPep8Naming
    def transform(self, X, *, local_data_model=None):
        """
        Transform X records.

        :param X: data frame to be transformed.
        :param local_data_model: pandas data model.
        :return: transformed data frame.
        """
        unknown = set(self.columns_needed) - set(X.columns)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        if local_data_model is None:
            local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(
                X
            )
        assert isinstance(local_data_model, data_algebra.data_model.DataModel)
        assert local_data_model.is_appropriate_data_instance(X)
        X = local_data_model.clean_copy(X)
        if self.blocks_in is not None:
            X = local_data_model.blocks_to_rowrecs(X, blocks_in=self.blocks_in)
        if self.blocks_out is not None:
            X = local_data_model.rowrecs_to_blocks(
                X,
                blocks_out=self.blocks_out,
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
        strict = self.strict and other.strict
        if inp.shape[0] < 2:
            if out.shape[0] < 2:
                return None
            else:
                return RecordMap(
                    blocks_out=RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                        strict=strict,
                    ),
                    strict=strict,
                )
        else:
            if out.shape[0] < 2:
                return RecordMap(
                    blocks_in=RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                        strict=strict,
                    ),
                    strict=strict,
                )
            else:
                return RecordMap(
                    blocks_in=RecordSpecification(
                        control_table=rsi,
                        record_keys=rk,
                        control_table_keys=s1.blocks_in.control_table_keys,
                        strict=strict,
                    ),
                    blocks_out=RecordSpecification(
                        control_table=rso,
                        record_keys=rk,
                        control_table_keys=s2.blocks_out.control_table_keys,
                        strict=strict,
                    ),
                    strict=strict,
                )

    def as_pipeline(
        self,
        *,
        table_name: Optional[str] = None,
        local_data_model=None,
        value_suffix: str = " value",
        record_key_suffix: str = " record key",
    ):
        """
        Build into processing pipeline.

        :param table_name: name for input table.
        :param local_data_model: optional Pandas data model.
        :param value_suffix: suffix to identify values
        :param record_key_suffix: suffix to identify record keys
        :return: cdata operator as a pipeline
        """
        assert isinstance(value_suffix, str)
        assert isinstance(record_key_suffix, str)
        exp_inp = self.example_input(
            local_data_model=local_data_model,
            value_suffix=value_suffix,
            record_key_suffix=record_key_suffix,
        )
        ops = data_algebra.data_ops.describe_table(
            exp_inp,
            table_name=table_name,
            row_limit=None,
            keep_sample=True,
            keep_all=True,
        ).convert_records(self)
        return ops

    def inverse(self):
        """
        Return inverse transform, if there is such (duplicate value keys or mis-matching
        row representations can prevent this).
        """
        assert self.strict
        return RecordMap(
            blocks_in=self.blocks_out, blocks_out=self.blocks_in, strict=True
        )

    def act_on(self, b, *, correct_ordered_first_call: bool = False):
        assert isinstance(correct_ordered_first_call, bool)
        if isinstance(b, RecordMap):
            return self.compose(b)
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            return b.convert_records(self)
        if correct_ordered_first_call and isinstance(b, ShiftPipeAction):
            return b.act_on(
                self, correct_ordered_first_call=False
            )  # fall back to peer's action
        # assume table like
        return self.transform(b)

    def fmt(self) -> str:
        """Format for informal presentation."""
        if (self.blocks_in is None) and (self.blocks_out is None):
            return "RecordMap(no-op)"
        example_input = self.example_input()
        example_input_formatted = _format_table(
            example_input,
            record_id_cols=set(self.record_keys()),
            control_id_cols=self.input_control_table_key_columns(),
            add_style=False,
        )
        example_output = self.transform(example_input)
        example_output_formatted = _format_table(
            example_output,
            record_id_cols=set(self.record_keys()),
            control_id_cols=self.output_control_table_key_columns(),
            add_style=False,
        )
        s = (
            "RecordMap: transforming records of the form:\n"
            + str(example_input_formatted)
            + "\nto records of the form:\n"
            + str(example_output_formatted)
        )
        return s

    def __repr__(self):
        s = (
            "data_algebra.cdata.RecordMap("
            + "\n    blocks_in="
            + self.blocks_in.__repr__()
            + ",\n    blocks_out="
            + self.blocks_out.__repr__()
            + ",\n    strict="
            + self.strict.__repr__()
            + ")"
        )
        return s

    def __str__(self):
        return self.fmt()

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        if cycle:
            p.text("RecordMap()")
        else:
            p.text(self.fmt())

    def _repr_html_(self):
        if (self.blocks_in is None) and (self.blocks_out is None):
            return "RecordMap(no-op)"
        example_input = self.example_input()
        example_input_formatted = _format_table(
            example_input,
            record_id_cols=set(self.record_keys()),
            control_id_cols=self.input_control_table_key_columns(),
        )
        example_output = self.transform(example_input)
        example_output_formatted = _format_table(
            example_output,
            record_id_cols=set(self.record_keys()),
            control_id_cols=self.output_control_table_key_columns(),
        )
        s = (
            "RecordMap: transforming records of the form:<br>\n"
            + example_input_formatted
            + "<br>\nto records of the form:<br>\n"
            + example_output_formatted
        )
        return s


def pivot_blocks_to_rowrecs(
    *,
    attribute_key_column,
    attribute_value_column,
    record_keys,
    record_value_columns,
    local_data_model=None,
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
    local_data_model=None,
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
    value_cols: Iterable[str],
    local_data_model=None,
) -> RecordMap:
    """
    Specify the cdata transformation that pivots records from a single column of values into collected rows.
    https://en.wikipedia.org/wiki/Pivot_table#History

    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to take the names of columns as a column
    :param col_value_key: column name to take the values in columns as a column
    :param value_cols: columns to place values in
    :param local_data_model: data.frame data model
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
    if local_data_model is None:
        local_data_model = data_algebra.data_model.default_data_model()
    assert isinstance(local_data_model, data_algebra.data_model.DataModel)
    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=local_data_model.pd.DataFrame(
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
    value_cols: Iterable[str],
    local_data_model=None,
) -> RecordMap:
    """
    Specify the cdata transformation that un-pivots records into a single column of values plus keys.
    https://en.wikipedia.org/wiki/Pivot_table#History

    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to land the names of columns as a column
    :param col_value_key: column name to land the values in columns as a column
    :param value_cols: columns to take values from
    :param local_data_model: data.frame data model
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
    if local_data_model is None:
        local_data_model = data_algebra.data_model.default_data_model()
    assert isinstance(local_data_model, data_algebra.data_model.DataModel)
    record_map = RecordMap(
        blocks_out=RecordSpecification(
            control_table=local_data_model.pd.DataFrame(
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
