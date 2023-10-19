Module data_algebra.util
========================
Basic utilities. Not allowed to import many other modules.

Functions
---------

    
`check_columns_appear_compatible(d_left, d_right, *, columns: Optional[Iterable[str]] = None) ‑> Optional[Dict[str, Tuple[type, type]]]`
:   Check if columns have compatible types
    
    :param d_left: pandas dataframe to check
    :param d_right: pandas dataframe to check
    :param columns: columns to check, None means check all columns
    :return: None if compatible, else dictionary of mismatches

    
`compatible_types(types_seen: Iterable[type]) ‑> bool`
:   Check if a set of types are all considered equivalent.
    
    :param types_seen: collection of types seen
    :return: True if types are all compatible, else False.

    
`guess_carried_scalar_type(col) ‑> type`
:   Guess the type of a column or scalar.
    
    :param col: column or scalar to inspect
    :return: type of first non-None entry, if any , else type(None)

    
`guess_column_types(d, *, columns: Optional[Iterable[str]] = None) ‑> Dict[str, type]`
:   Guess column types as type of first non-missing value.
    Will not return series types, as some pandas data frames with non-trivial indexing report this type.
    
    :param d: pandas.DataFrame
    :param columns: list of columns to check, if None all columns are checked
    :return: map of column names to guessed types, empty dict if any column guess fails

    
`map_type_to_canonical(v: type) ‑> type`
:   Map type to a smaller set of considered equivalent types.
    
    :param v: type to map
    :return: type

    
`pandas_to_example_str(obj, *, local_data_model=None) ‑> str`
:   Convert data frame to a Python source code string.
    
    :param obj: data frame to convert.
    :param local_data_model: data model to use.
    :return: Python source code representation of obj.