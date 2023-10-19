Module data_algebra.test_util
=============================
Utils that help with testing. This module is allowed to import many other modules.

Functions
---------

    
`check_transform(ops, data, expect, *, float_tol: float = 1e-08, check_column_order: bool = False, cols_case_sensitive: bool = False, check_row_order: bool = False, check_parse: bool = True, try_on_DBs: bool = True, models_to_skip: Optional[Iterable] = None, valid_for_empty: bool = True, empty_produces_empty: bool = True, local_data_model=None, try_on_Polars: bool = True) ‑> None`
:   Test an operator dag produces the expected result, and parses correctly.
    Assert/raise if there are issues.
    
    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data: pd.DataFrame or map of strings to pd.DataFrame
    :param expect: pd.DataFrame
    :param float_tol: passed to equivalent_frames()
    :param check_column_order: passed to equivalent_frames()
    :param cols_case_sensitive: passed to equivalent_frames()
    :param check_row_order: passed to equivalent_frames()
    :param check_parse: if True check expression parses/formats to self
    :param try_on_DBs: if true, try on databases
    :param models_to_skip: None or set of model names or models to skip testing
    :param valid_for_empty: logical, if True perform tests on empty inputs
    :param empty_produces_empty: logical, if True assume empty inputs should produce empty output
    :param local_data_mode: data model to use
    :param try_on_Polars: try tests again on Polars
    :return: nothing

    
`check_transform_on_data_model(*, ops, data: Dict, expect, float_tol: float = 1e-08, check_column_order: bool = False, cols_case_sensitive: bool = False, check_row_order: bool = False, check_parse: bool = True, local_data_model=None, valid_for_empty: bool = True, empty_produces_empty: bool = True) ‑> None`
:   Test an operator dag produces the expected result, and parses correctly.
    Asserts if there are issues
    
    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data: DataFrame or map of strings to pd.DataFrame
    :param expect: DataFrame
    :param float_tol: passed to equivalent_frames()
    :param check_column_order: passed to equivalent_frames()
    :param cols_case_sensitive: passed to equivalent_frames()
    :param check_row_order: passed to equivalent_frames()
    :param check_parse: if True check expression parses/formats to self
    :param local_data_model: optional alternate evaluation model
    :param valid_for_empty: logical, if True test on empty inputs
    :param empty_produces_empty: logical, if True assume emtpy inputs should produce empty output
    :return: None, assert if there is an issue

    
`equivalent_frames(a, b, *, float_tol: float = 1e-08, check_column_order: bool = False, cols_case_sensitive: bool = False, check_row_order: bool = False) ‑> bool`
:   return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing. None and nan are considered equivalent in numeric contexts.

    
`formats_to_self(ops, *, data_model_map: Optional[Dict[str, Any]] = None) ‑> bool`
:   Check a operator dag formats and parses back to itself.
    Can raise exceptions. Also checks pickling.
    
    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data_model_map: map from abbreviated module names to modules (i.e. define pd)
    :return: logical, True if formats and evals back to self

    
`get_test_dbs()`
:   handles connected to databases for testing.