Module data_algebra.solutions
=============================
data algebra solutions to common data processing problems

Functions
---------

    
`braid_data(*, d_state: data_algebra.view_representations.ViewRepresentation, d_event: data_algebra.view_representations.ViewRepresentation, order_by: Iterable[str], partition_by: Optional[Iterable[str]] = None, state_value_column_name: str, event_value_column_names: Iterable[str], source_id_column: str = 'record_type', state_row_mark: str = 'state_row', event_row_mark: str = 'event_row', stand_in_values: Dict, locf_to_use_column_name: str = 'locf_to_use', locf_non_null_rank_column_name: str = 'locf_non_null_rank', locf_tiebreaker_column_name: str = 'locf_tiebreaker') ‑> data_algebra.view_representations.ViewRepresentation`
:   Mix data from two sources, ordering by order_by columns and carrying forward observations
    on d_state value column.
    
    :param d_state: ViewRepresentation representation of state by order_by.
    :param d_event: ViewRepresentation representation of events by order_by.
    :param order_by: columns to order by (non empty list of column names)
    :param partition_by: optional partitioning column names
    :param state_value_column_name: column to copy from d_state and propagate forward
    :param event_value_column_names: columns to copy from d_event
    :param source_id_column: name for source identification column.
    :param state_row_mark: source annotation of state rows.
    :param event_row_mark: source annotation of event rows.
    :param stand_in_values: dictionary stand in values to use for state_value_column_name and event_value_column_names
            needed to get column types correct, replaced by None and not passed further.
    :param locf_to_use_column_name: name for a temporary values column
    :param locf_non_null_rank_column_name: name for a temporary values column
    :param locf_tiebreaker_column_name: name for a temporary values column
    :return: ops

    
`def_multi_column_map(d: data_algebra.view_representations.ViewRepresentation, *, mapping_table: data_algebra.view_representations.ViewRepresentation, row_keys: Iterable[str], col_name_key: str = 'column_name', col_value_key: str = 'column_value', mapped_value_key: str = 'mapped_value', cols_to_map: Iterable[str], coalesce_value=None, cols_to_map_back: Optional[Iterable[str]] = None) ‑> data_algebra.view_representations.ViewRepresentation`
:   Map all columns in list cols_to_map through the mapping in mapping table (key by column name and value).
    d should be uniquely keyed by row_keys, and mapping table should be uniquely keyed by [col_name_key, col_value_key].
    
    :param d: view to re-map
    :param mapping_table: view to get mappings from
    :param row_keys: columns that uniquely identify rows in d
    :param col_name_key: column name specifying columns in mapping_table
    :param col_value_key: column name specifying pre-map values in mapping table
    :param mapped_value_key: column name specifying post-map values in mapping table
    :param cols_to_map: columns to re-map.
    :param coalesce_value: if not None, coalesce to this value
    :param cols_to_map_back: if not None new names for resulting columns
    :return: operations specifying how to re-map DataFrame

    
`last_observed_carried_forward(d: data_algebra.view_representations.ViewRepresentation, *, order_by: Iterable[str], partition_by: Optional[Iterable[str]] = None, value_column_name: str, selection_predicate: str = 'is_null()', locf_to_use_column_name: str = 'locf_to_use', locf_non_null_rank_column_name: str = 'locf_non_null_rank', locf_tiebreaker_column_name: str = 'locf_tiebreaker') ‑> data_algebra.view_representations.ViewRepresentation`
:   Copy last observed non-null value in column value_column_name forward using order order_by and
    optional partition_by partition.
    
    :param d: ViewRepresentation representation of data to transform.
    :param order_by: columns to order by
    :param partition_by: optional partitioning column
    :param value_column_name: column to alter
    :param selection_predicate: expression to choose values to replace
    :param locf_to_use_column_name: name for a temporary values column
    :param locf_non_null_rank_column_name: name for a temporary values column
    :param locf_tiebreaker_column_name: name for a temporary values column
    :return: ops

    
`rank_to_average(d: data_algebra.view_representations.ViewRepresentation, *, order_by: Iterable[str], partition_by: Optional[Iterable[str]] = None, rank_column_name: str, tie_breaker_column_name: str = 'rank_tie_breaker') ‑> data_algebra.view_representations.ViewRepresentation`
:   Compute rank where the rank of each item is the average of all items with same order
    position. That is rank_to_average([1, 1, 2]) = [1.5, 1.5, 3].
    
    :param d: ViewRepresentation representation of data to transform.
    :param order_by: columns to order by
    :param partition_by: optional partitioning column
    :param rank_column_name: column to land ranks in
    :param tie_breaker_column_name: temp column
    :return: ops

    
`replicate_rows_query(d: data_algebra.view_representations.ViewRepresentation, *, count_column_name: str, seq_column_name: str, join_temp_name: str, max_count: int) ‑> Tuple[data_algebra.view_representations.ViewRepresentation, Any]`
:   Build query to replicate each row by count_column_name copies.
    
    :param d: incoming data description.
    :param count_column_name: name of count column, should be non-negative integers.
    :param seq_column_name: name of colulmn to land sequence in.
    :param join_temp_name: name for join temp table.
    :param max_count: maximum in count column we need to handle, should be a reasonable upper bound.
    :return: ops and table to join against

    
`xicor_query(data: data_algebra.view_representations.ViewRepresentation, *, x_name: str = 'x', y_name: str = 'y', var_keys: Iterable[str] = ()) ‑> data_algebra.view_representations.ViewRepresentation`
:   Build a query computing the xicor of y_name as a function of x_name for each var_keys group of rows.
    Ref: https://arxiv.org/abs/1909.10140
    
    xicor(x, y) : 1 - n sum(i = 0, n-2) |r(i+1) - r(i)| / (2 * sum(i=0, n-1) l(i) (n - l(i)),
    where r(i) is the rank of the i-th Y item when ordered by x, and l(i) is the reverse rank of
    the l-th Y item.
    
    :param data: description of data to transform
    :param x_name: name for explanatory variable column.
    :param y_name: name for dependent variable column.
    :param var_keys: list of names for variable id columns.
    :return: data algebra query computing xicor.

    
`xicor_score_variables_plan(d: data_algebra.view_representations.ViewRepresentation, *, x_vars: Iterable[str], y_name: str, n_rep: int = 25) ‑> Tuple[data_algebra.view_representations.ViewRepresentation, str, Any]`
:   Set up a query to batch compute xicor.
    
    :param d: description of incoming data frame
    :param x_vars: list of explanatory variable names
    :param y_name: name of dependent variable
    :param n_rep: number of times to repeat calculation
    :return: group_calc_ops, rep_frame_name, rep_frame