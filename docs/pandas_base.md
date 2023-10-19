Module data_algebra.pandas_base
===============================
Base class for adapters for Pandas-like APIs

Functions
---------

    
`none_mark_scalar_or_length(v) ‑> Optional[int]`
:   Test if item is a scalar (returning None) if it is, else length of object.
    
    :param v: value to test
    :return: None if value is a scalar, else length.

    
`promote_scalar_to_array(vi, *, target_len: int) ‑> List`
:   Convert a scalar into a vector. Pass a non-trivial array through.
    
    :param vi: value to promote to scalar
    :target_len: length for vector
    :return: list

Classes
-------

`PandasModelBase(*, pd: module, presentation_model_name: str)`
:   Base class for implementing the data algebra on pandas-like APIs

    ### Ancestors (in MRO)

    * data_algebra.data_model.DataModel
    * data_algebra.expression_walker.ExpressionWalker
    * abc.ABC

    ### Descendants

    * data_algebra.pandas_model.PandasModel

    ### Class variables

    `impl_map: Dict[str, Callable]`
    :

    `pd: module`
    :

    `transform_op_map: Dict[str, str]`
    :

    `user_fun_map: Dict[str, Callable]`
    :

    ### Methods

    `act_on_column_name(self, *, arg, value)`
    :   Action for a column name.
        
        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on

    `act_on_expression(self, *, arg, values: List, op)`
    :   Action for a column name.
        
        :param arg: item we are acting on
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on

    `add_data_frame_columns_to_data_frame_(self, res, transient_new_frame)`
    :   Add columns from transient_new_frame to res. Res may be altered, and either of res or
        transient_new_frame may be returned.

    `can_convert_col_to_numeric(self, x)`
    :   Return True if column or value can be converted to numeric type.

    `columns_to_frame_(self, cols: Dict[str, Any], *, target_rows: Optional[int] = None)`
    :   Convert a dictionary of column names to series-like objects and scalars into a Pandas data frame.
        Deal with special cases, such as some columns coming in as scalars (often from Panda aggregation).
        
        :param cols: dictionary mapping column names to columns
        :param target_rows: number of rows we are shooting for
        :return: Pandas data frame.

    `data_frame(self, arg=None)`
    :   Build a new emtpy data frame.
        
        :param arg" optional argument passed to constructor.
        :return: data frame

    `eval(self, op, *, data_map: Dict[str, Any])`
    :   Implementation of Pandas evaluation of operators
        
        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames or data sources
        :return: data frame result

    `isinf(self, x)`
    :   Return vector indicating which entries are nan (vectorized).

    `isnan(self, x)`
    :   Return vector indicating which entries are nan (vectorized).

    `isnull(self, x)`
    :   Return vector indicating which entries are null (vectorized).

    `standardize_join_code_(self, jointype)`
    :   Map join names to Pandas names. Internal method.

    `to_numeric(self, x, *, errors='coerce')`
    :   Convert column to numeric.