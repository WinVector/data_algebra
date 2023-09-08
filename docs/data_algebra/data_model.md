Module data_algebra.data_model
==============================
Interface for realizing the data algebra as a sequence of steps over an object.

Functions
---------

    
`default_data_model() ‑> data_algebra.data_model.DataModel`
:   Get the default (Pandas) data model

    
`lookup_data_model_for_dataframe(d) ‑> data_algebra.data_model.DataModel`
:   

    
`lookup_data_model_for_key(key: str) ‑> data_algebra.data_model.DataModel`
:   

Classes
-------

`DataModel(*, presentation_model_name: str, module)`
:   Interface for realizing the data algebra as a sequence of steps over Pandas like objects.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.pandas_base.PandasModelBase
    * data_algebra.polars_model.PolarsModel

    ### Class variables

    `module: Any`
    :

    `presentation_model_name: str`
    :

    ### Methods

    `bad_column_positions(self, x)`
    :   Return vector indicating which entries are bad (null or nan) (vectorized).

    `blocks_to_rowrecs(self, data, *, blocks_in)`
    :   Convert a block record (record spanning multiple rows) into a rowrecord (record in a single row).
        
        :param data: data frame to be transformed
        :param blocks_in: cdata record specification
        :return: transformed data frame

    `clean_copy(self, df)`
    :   Copy of data frame without indices.

    `concat_columns(self, frame_list)`
    :   Concatenate columns from frame_list

    `concat_rows(self, frame_list: List)`
    :   Concatenate rows from frame_list

    `data_frame(self, arg=None)`
    :   Build a new data frame.
        
        :param arg: optional argument passed to constructor.
        :return: data frame

    `drop_indices(self, df) ‑> None`
    :   Drop indices in place.

    `eval(self, op, *, data_map: Dict[str, Any])`
    :   Implementation of Pandas evaluation of operators
        
        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result

    `get_cell(self, *, d, row: int, colname: str)`
    :   get a value from a cell

    `is_appropriate_data_instance(self, df) ‑> bool`
    :   Check if df is our type of data frame.

    `rowrecs_to_blocks(self, data, *, blocks_out)`
    :   Convert rowrecs (single row records) into block records (multiple row records).
        
        :param data: data frame to transform.
        :param blocks_out: cdata record specification.
        :return: transformed data frame

    `set_col(self, *, d, colname: str, values)`
    :   set column, return ref

    `table_is_keyed_by_columns(self, table, *, column_names: Iterable[str]) ‑> bool`
    :   Check if a table is keyed by a given list of column names.
        
        :param table: DataFrame
        :param column_names: list of column names
        :return: True if rows are uniquely keyed by values in named columns

    `to_pandas(self, df)`
    :   Convert to Pandas