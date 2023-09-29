Module data_algebra.cdata
=========================
Class for representing record structure transformations.

Functions
---------

    
`pivot_blocks_to_rowrecs(*, attribute_key_column, attribute_value_column, record_keys, record_value_columns, local_data_model=None)`
:   Build a block records to row records map. This is very similar to a SQL pivot.
    
    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param local_data_model: data.frame data model
    :return: RecordMap

    
`pivot_rowrecs_to_blocks(*, attribute_key_column, attribute_value_column, record_keys, record_value_columns, local_data_model=None)`
:   Build a row records to block records map. This is very similar to a SQL unpivot.
    
    :param attribute_key_column: column to identify record attribute keys
    :param attribute_value_column: column for record attribute values
    :param record_keys: names of key columns identifying row record blocks
    :param record_value_columns: names of columns to take row record values from
    :param local_data_model: data.frame data model
    :return: RecordMap

    
`pivot_specification(*, row_keys: Iterable[str], col_name_key: str = 'column_name', col_value_key: str = 'column_value', value_cols: Iterable[str], local_data_model=None) ‑> data_algebra.cdata.RecordMap`
:   Specify the cdata transformation that pivots records from a single column of values into collected rows.
    https://en.wikipedia.org/wiki/Pivot_table#History
    
    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to take the names of columns as a column
    :param col_value_key: column name to take the values in columns as a column
    :param value_cols: columns to place values in
    :param local_data_model: data.frame data model
    :return: RecordSpecification

    
`unpivot_specification(*, row_keys: Iterable[str], col_name_key: str = 'column_name', col_value_key: str = 'column_value', value_cols: Iterable[str], local_data_model=None) ‑> data_algebra.cdata.RecordMap`
:   Specify the cdata transformation that un-pivots records into a single column of values plus keys.
    https://en.wikipedia.org/wiki/Pivot_table#History
    
    :param row_keys: columns that identify rows in the incoming data set
    :param col_name_key: column name to land the names of columns as a column
    :param col_value_key: column name to land the values in columns as a column
    :param value_cols: columns to take values from
    :param local_data_model: data.frame data model
    :return: RecordSpecification

Classes
-------

`RecordMap(*, blocks_in: Optional[data_algebra.cdata.RecordSpecification] = None, blocks_out: Optional[data_algebra.cdata.RecordSpecification] = None, strict: bool = True)`
:   Class for specifying general record to record transforms.
    
    Build the transform specification. At least one of blocks_in or blocks_out must not be None.
    
    :param blocks_in: incoming record specification, None for row-records.
    :param blocks_out: outgoing record specification, None for row-records.
    :param strict: if True insist block be strict, and in and out blocks agree on row-form columns.∂

    ### Ancestors (in MRO)

    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `blocks_in: Optional[data_algebra.cdata.RecordSpecification]`
    :

    `blocks_out: Optional[data_algebra.cdata.RecordSpecification]`
    :

    `strict: bool`
    :

    ### Methods

    `as_pipeline(self, *, table_name: Optional[str] = None, local_data_model=None, value_suffix: str = ' value', record_key_suffix: str = ' record key')`
    :   Build into processing pipeline.
        
        :param table_name: name for input table.
        :param local_data_model: optional Pandas data model.
        :param value_suffix: suffix to identify values
        :param record_key_suffix: suffix to identify record keys
        :return: cdata operator as a pipeline

    `compose(self, other)`
    :   Experimental method to compose transforms
        (self.compose(other)).transform(data) == self.transform(other.transform(data))
        
        :param other: another data_algebra.cdata.RecordMap
        :return:

    `example_input(self, *, local_data_model=None, value_suffix: str = ' value', record_key_suffix: str = ' record key')`
    :   Return example output record.
        
        :param local_data_model: optional Pandas data model.
        :param value_suffix: suffix to identify values
        :param record_key_suffix: suffix to identify record keys
        :return: example result data frame.

    `fmt(self) ‑> str`
    :   Format for informal presentation.

    `input_control_table_key_columns(self) ‑> List[str]`
    :

    `inverse(self)`
    :   Return inverse transform, if there is such (duplicate value keys or mis-matching
        row representations can prevent this).

    `output_control_table_key_columns(self) ‑> List[str]`
    :

    `record_keys(self) ‑> List[str]`
    :   Return keys specifying which set of rows are in a record.

    `transform(self, X, *, local_data_model=None)`
    :   Transform X records.
        
        :param X: data frame to be transformed.
        :param local_data_model: pandas data model.
        :return: transformed data frame.

`RecordSpecification(control_table, *, record_keys=None, control_table_keys=None, strict: bool = True, local_data_model=None)`
:   Class to represent a data record. 
    For single row data records use None as the specification.
    
    :param control_table: data.frame describing record layout
    :param record_keys: array of record key column names
           defaults to no columns.
    :param control_table_keys: array of control_table key column names,
           defaults to first column for non-trivial blocks and no columns for rows.
    :param strict: if True don't allow duplicate value names
    :param local_data_model: data.frame data model

    ### Class variables

    `block_columns: List[str]`
    :

    `row_columns: List[str]`
    :

    `strict: bool`
    :

    ### Methods

    `fmt(self)`
    :   Prepare for printing
        
        :return: multi line string representation.

    `map_from_keyed_column(self, *, key_column_name: str = 'measure', value_column_name: str = 'value')`
    :   Build a RecordMap mapping this RecordSpecification from a table
        where only one column holds values.
        
        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column
        :return: Record map

    `map_from_rows(self)`
    :   Build a RecordMap mapping this RecordSpecification from rowrecs
        
        :return: RecordMap

    `map_to_keyed_column(self, *, key_column_name: str = 'measure', value_column_name: str = 'value')`
    :   Build a RecordMap mapping this RecordSpecification to a table
        where only one column holds values.
        Note: for type safety prefer map_to_rows() to map_to_keyed_column().
        
        
        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column
        :return: Record map

    `map_to_rows(self)`
    :   Build a RecordMap mapping this RecordSpecification to rowrecs
        
        :return: RecordMap

    `row_record_form(self)`
    :   Return specification of matching row record form.
        Note: prefer using None to specify row records specs.

    `row_version(self, *, include_record_keys: bool = True) ‑> List[str]`
    :   Return copy of record as a row record.
        
        :param include_record_keys: logical, if True include record keys as columns
        :return: column list

    `value_column_form(self, *, key_column_name: str = 'measure', value_column_name: str = 'value')`
    :   Return specification of the matching value column form.
        Note: for type safety prefer map_to_rows() to map_to_keyed_column().
        
        :param key_column_name: name for additional keying column
        :param value_column_name: name for value column