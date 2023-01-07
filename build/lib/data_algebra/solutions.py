"""
data algebra solutions to common data processing problems
"""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy
import data_algebra.data_model
from data_algebra.data_ops import descr, TableDescription, ViewRepresentation
from data_algebra.cdata import (
    pivot_specification,
    unpivot_specification,
    RecordMap,
    RecordSpecification,
)


def def_multi_column_map(
    d: ViewRepresentation,
    *,
    mapping_table: ViewRepresentation,
    row_keys: Iterable[str],
    col_name_key: str = "column_name",
    col_value_key: str = "column_value",
    mapped_value_key: str = "mapped_value",
    cols_to_map: Iterable[str],
    coalesce_value=None,
    cols_to_map_back: Optional[Iterable[str]] = None,
) -> ViewRepresentation:
    """
    Map all columns in list cols_to_map through the mapping in mapping table (key by column name and value).
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
    """
    assert not isinstance(row_keys, str)
    row_keys = list(row_keys)
    assert len(row_keys) > 0
    assert not isinstance(cols_to_map, str)
    cols_to_map = list(cols_to_map)
    assert len(cols_to_map) > 0
    assert isinstance(col_name_key, str)
    assert isinstance(col_value_key, str)
    assert isinstance(mapped_value_key, str)
    if cols_to_map_back is not None:
        assert not isinstance(cols_to_map_back, str)
        cols_to_map_back = list(cols_to_map_back)
        assert len(cols_to_map_back) == len(cols_to_map)
    pre_col_names = row_keys + cols_to_map
    assert len(pre_col_names) == len(set(pre_col_names))
    mid_col_names = row_keys + [col_name_key, col_value_key, mapped_value_key]
    assert len(mid_col_names) == len(set(mid_col_names))
    if cols_to_map_back is None:
        post_col_names = row_keys + cols_to_map
    else:
        post_col_names = row_keys + cols_to_map_back
    assert len(post_col_names) == len(set(post_col_names))
    record_map_to = unpivot_specification(
        row_keys=row_keys,
        col_name_key=col_name_key,
        col_value_key=col_value_key,
        value_cols=cols_to_map,
    )
    record_map_back = pivot_specification(
        row_keys=row_keys,
        col_name_key=col_name_key,
        col_value_key=mapped_value_key,
        value_cols=cols_to_map,
    )
    ops = (
        d.select_columns(row_keys + cols_to_map)
        .convert_records(record_map_to)
        .natural_join(
            b=mapping_table.select_columns(
                [col_name_key, col_value_key, mapped_value_key]
            ),
            jointype="left",
            on=[col_name_key, col_value_key],
        )
    )
    if coalesce_value is not None:
        ops = ops.extend(
            {mapped_value_key: f"{mapped_value_key}.coalesce({coalesce_value})"}
        )
    ops = ops.convert_records(record_map_back)
    if cols_to_map_back is not None:
        # could do this in the record mapping, but this seems easier to read
        ops = ops.rename_columns(
            {
                new_name: old_name
                for new_name, old_name in zip(cols_to_map_back, cols_to_map)
            }
        )
    return ops


def xicor_query(
    data: ViewRepresentation,
    *,
    x_name: str = "x",
    y_name: str = "y",
    var_keys: Iterable[str] = tuple(),
) -> ViewRepresentation:
    """
    Build a query computing the xicor of y_name as a function of x_name for each var_keys group of rows.
    Ref: https://arxiv.org/abs/1909.10140

    xicor(x, y) : 1 - n sum(i = 0, n-2) |r(i+1) - r(i)| / (2 * sum(i=0, n-1) l(i) (n - l(i)),
    where r(i) is the rank of the i-th Y item when ordered by x, and l(i) is the reverse rank of
    the l-th Y item.

    :param data: description of data to transform
    :param x_name: name for explanatory variable column.
    :param y_name: name for dependent variable column.
    :param var_keys: list of names for variable id columns.
    :return: data algebra query computing xicor.
    """
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    assert not isinstance(var_keys, str)
    var_keys = list(var_keys)
    x_tie_breaker = x_name + "_tie_breaker"
    y_group = y_name + "_group"
    names = [
        x_name,
        y_name,
        x_tie_breaker,
        y_group,
        "l",
        "n",
        "r",
        "rplus",
        "rdiff",
        "lterm",
        "num_sum",
        "den_sum",
        "xicor",
    ] + var_keys
    assert len(names) == len(set(names))
    ops = (
        data.extend(
            {y_group: f"{y_name}.as_str()"}
        )  # Google BigQuery won't group by float
        .extend(
            {  # convert types, and add in tie breaking column
                x_name: f"1.0 * {x_name}",
                y_name: f"1.0 * {y_name}",
                x_tie_breaker: "_uniform()",
            }
        )
        .extend({"n": "(1).sum()"}, partition_by=var_keys)  # annotate in number of rows
        .extend(  # compute y ranks, that we will use to compare rank changes wrt x
            {"r": "(1).cumsum()"}, order_by=[y_name], partition_by=var_keys
        )
        .extend(  # compute reverse y ranks, used to normalize for ties in denominator
            {"l": "(1).cumsum()"},
            order_by=[y_name],
            reverse=[y_name],
            partition_by=var_keys,
        )
        .extend(  # go to max rank of group tie breaking, also why we don't need tiebreaker in cumsums
            {"l": "l.max()", "r": "r.max()"}, partition_by=[y_group] + var_keys
        )
        .extend(  # get y rank and y rank of next x-item into same row so we can take a difference
            {"rplus": "r.shift(1)"},
            order_by=[x_name, x_tie_breaker],
            reverse=[x_name, x_tie_breaker],
            partition_by=var_keys,
        )
        .extend(  # compute numerator and denominator terms
            {"rdiff": "((rplus - r).abs()).coalesce(0)", "lterm": "l * (n - l)"}
        )
        .project(  # aggregate to compute sums in xicor definition
            {
                "num_sum": "rdiff.sum()",
                "den_sum": "lterm.sum()",
                "n": "n.max()",  # pseudo-aggregation n is constant across rows
            },
            group_by=var_keys,
        )
        .extend(  # apply actual xicor formula
            {"xicor": "1.0 - ((n * num_sum) / (2.0 * den_sum))"}
        )
        .select_columns(var_keys + ["xicor"])
    )
    return ops


def xicor_score_variables_plan(
    d: ViewRepresentation,
    *,
    x_vars: Iterable[str],
    y_name: str,
    n_rep: int = 25,
) -> Tuple[ViewRepresentation, str, Any]:
    """
    Set up a query to batch compute xicor.

    :param d: description of incoming data frame
    :param x_vars: list of explanatory variable names
    :param y_name: name of dependent variable
    :param n_rep: number of times to repeat calculation
    :return: group_calc_ops, rep_frame_name, rep_frame
    """
    assert not isinstance(x_vars, str)
    x_vars = list(x_vars)
    assert len(x_vars) > 0
    assert numpy.all([isinstance(c, str) for c in x_vars])
    assert len(x_vars) == len(set(x_vars))
    assert isinstance(y_name, str)
    assert y_name not in x_vars
    d_col_set = set(d.column_names)
    assert y_name in d_col_set
    assert numpy.all([c in d_col_set for c in x_vars])
    assert isinstance(n_rep, int)
    record_map = RecordMap(
        blocks_out=RecordSpecification(
            control_table=data_algebra.data_model.default_data_model().data_frame(
                {
                    "variable_name": x_vars,
                    "x": x_vars,
                    "y": y_name,
                }
            ),
            record_keys=["_da_xicor_tmp_index"],
            control_table_keys=["variable_name"],
            strict=False,
        ),
        strict=False,
    )
    rep_frame = data_algebra.data_model.default_data_model().data_frame({"rep": range(n_rep)})
    grouped_calc = (
        xicor_query(
            d
                .extend({"_da_xicor_tmp_order": "1"})
                .extend({"_da_xicor_tmp_index": "(1).cumsum()"}, order_by=["_da_xicor_tmp_order"])
                .convert_records(
                    record_map
                ).natural_join(  # cross join rows to get experiment repetitions
                    b=descr(rep_frame=rep_frame),
                    on=[],
                    jointype="cross",
                ),
                var_keys=["variable_name", "rep"],
        )
        .project(
            {
                "xicor_mean": "xicor.mean()",
                "xicor_std": "xicor.std()",
            },
            group_by=["variable_name"],
        )
        .order_rows(["variable_name"])
    )
    return grouped_calc, "rep_frame", rep_frame


def last_observed_carried_forward(
    d: ViewRepresentation,
    *,
    order_by: Iterable[str],
    partition_by: Optional[Iterable[str]] = None,
    value_column_name: str,
    selection_predicate: str = "is_null()",
    locf_to_use_column_name: str = "locf_to_use",
    locf_non_null_rank_column_name: str = "locf_non_null_rank",
    locf_tiebreaker_column_name: str = "locf_tiebreaker",
) -> ViewRepresentation:
    """
    Copy last observed non-null value in column value_column_name forward using order order_by and
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
    """
    assert isinstance(d, ViewRepresentation)
    assert isinstance(locf_to_use_column_name, str)
    assert isinstance(locf_non_null_rank_column_name, str)
    cols = [
        locf_to_use_column_name,
        locf_non_null_rank_column_name,
        locf_tiebreaker_column_name,
    ] + list(d.column_names)
    assert len(cols) == len(set(cols))
    assert not isinstance(order_by, str)
    assert isinstance(selection_predicate, str)
    assert isinstance(locf_to_use_column_name, str)
    assert isinstance(locf_non_null_rank_column_name, str)
    assert isinstance(locf_tiebreaker_column_name, str)
    order_by = list(order_by)
    if partition_by is None:
        partition_by = []
    else:
        assert not isinstance(partition_by, str)
        partition_by = list(partition_by)
    d_marked = (
        d.extend(
            {
                locf_to_use_column_name: f"{value_column_name}.{selection_predicate}.where(0, 1)"
            }
        )
        .extend(
            {locf_tiebreaker_column_name: "_row_number()"},
            order_by=partition_by + order_by,
        )
        .extend(
            {locf_non_null_rank_column_name: f"{locf_to_use_column_name}.cumsum()"},
            order_by=order_by + [locf_tiebreaker_column_name],
            partition_by=partition_by,
        )
    )
    ops = d_marked.natural_join(
        b=d_marked.select_rows(f"{locf_to_use_column_name} == 1").select_columns(
            partition_by + [locf_non_null_rank_column_name, value_column_name]
        ),
        on=partition_by + [locf_non_null_rank_column_name],
        jointype="left",
    ).drop_columns(
        [
            locf_to_use_column_name,
            locf_non_null_rank_column_name,
            locf_tiebreaker_column_name,
        ]
    )
    return ops


def braid_data(
    *,
    d_state: ViewRepresentation,
    d_event: ViewRepresentation,
    order_by: Iterable[str],
    partition_by: Optional[Iterable[str]] = None,
    state_value_column_name: str,
    event_value_column_names: Iterable[str],
    source_id_column: str = "record_type",
    state_row_mark: str = "state_row",
    event_row_mark: str = "event_row",
    stand_in_values: Dict,
    locf_to_use_column_name: str = "locf_to_use",
    locf_non_null_rank_column_name: str = "locf_non_null_rank",
    locf_tiebreaker_column_name: str = "locf_tiebreaker",
) -> ViewRepresentation:
    """
    Mix data from two sources, ordering by order_by columns and carrying forward observations
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
    """
    assert isinstance(d_state, ViewRepresentation)
    assert isinstance(d_event, ViewRepresentation)
    assert not isinstance(order_by, str)
    order_by = list(order_by)
    assert len(order_by) > 0
    if partition_by is not None:
        assert not isinstance(partition_by, str)
        partition_by = list(partition_by)
    else:
        partition_by = []
    assert isinstance(state_value_column_name, str)
    assert not isinstance(event_value_column_names, str)
    event_value_column_names = list(event_value_column_names)
    assert isinstance(source_id_column, str)
    assert isinstance(state_row_mark, str)
    assert isinstance(event_row_mark, str)
    assert isinstance(locf_to_use_column_name, str)
    assert isinstance(locf_non_null_rank_column_name, str)
    assert isinstance(locf_tiebreaker_column_name, str)
    assert isinstance(stand_in_values, dict)
    together = (
        d_state.extend({k: stand_in_values[k] for k in event_value_column_names})
        .select_columns(
            partition_by
            + order_by
            + [state_value_column_name]
            + event_value_column_names
        )
        .concat_rows(
            b=(
                d_event.extend(
                    {state_value_column_name: stand_in_values[state_value_column_name]}
                ).select_columns(
                    partition_by
                    + order_by
                    + [state_value_column_name]
                    + event_value_column_names
                )
            ),
            id_column=source_id_column,
            a_name=state_row_mark,
            b_name=event_row_mark,
        )
        # clear out stand-in values
        .extend(
            {
                state_value_column_name: f'({source_id_column} == "{event_row_mark}").if_else(None, {state_value_column_name})'
            }
        )
        .extend(
            {
                k: f'({source_id_column} == "{state_row_mark}").if_else(None, {k})'
                for k in event_value_column_names
            }
        )
    )
    ops = last_observed_carried_forward(
        together,
        order_by=order_by,
        partition_by=partition_by,
        value_column_name=state_value_column_name,
        selection_predicate="is_null()",
        locf_to_use_column_name=locf_to_use_column_name,
        locf_non_null_rank_column_name=locf_non_null_rank_column_name,
        locf_tiebreaker_column_name=locf_tiebreaker_column_name,
    )
    return ops


def rank_to_average(
    d: ViewRepresentation,
    *,
    order_by: Iterable[str],
    partition_by: Optional[Iterable[str]] = None,
    rank_column_name: str,
    tie_breaker_column_name: str = "rank_tie_breaker",
) -> ViewRepresentation:
    """
    Compute rank where the rank of each item is the average of all items with same order
    position. That is rank_to_average([1, 1, 2]) = [1.5, 1.5, 3].

    :param d: ViewRepresentation representation of data to transform.
    :param order_by: columns to order by
    :param partition_by: optional partitioning column
    :param rank_column_name: column to land ranks in
    :param tie_breaker_column_name: temp column
    :return: ops
    """
    assert isinstance(d, ViewRepresentation)
    assert not isinstance(order_by, str)
    order_by = list(order_by)
    if partition_by is None:
        partition_by = []
    else:
        assert not isinstance(partition_by, str)
        partition_by = list(partition_by)
    cols = [rank_column_name, tie_breaker_column_name] + list(d.column_names)
    assert len(cols) == len(set(cols))
    ops = (
        d.extend(  # database sum() is constant per group when partitioned, so cumsum fails without tiebreaker
            {tie_breaker_column_name: "_row_number()"}, order_by=order_by
        )
        .extend(
            {
                rank_column_name: "(1.0).cumsum()",
            },
            order_by=order_by + [tie_breaker_column_name],
            partition_by=partition_by,
        )
        .extend(
            {
                rank_column_name: f"{rank_column_name}.mean()",
            },
            partition_by=partition_by + order_by,
        )
        .drop_columns([tie_breaker_column_name])
    )
    return ops


def replicate_rows_query(
    d: ViewRepresentation,
    *,
    count_column_name: str,
    seq_column_name: str,
    join_temp_name: str,
    max_count: int,
) -> Tuple[ViewRepresentation, Any]:
    """
    Build query to replicate each row by count_column_name copies.

    :param d: incoming data description.
    :param count_column_name: name of count column, should be non-negative integers.
    :param seq_column_name: name of colulmn to land sequence in.
    :param join_temp_name: name for join temp table.
    :param max_count: maximum in count column we need to handle, should be a reasonable upper bound.
    :return: ops and table to join against
    """
    assert isinstance(d, TableDescription)
    assert isinstance(count_column_name, str)
    assert count_column_name in d.column_names
    assert isinstance(seq_column_name, str)
    assert seq_column_name not in d.column_names
    assert isinstance(join_temp_name, str)
    assert isinstance(max_count, int)
    assert max_count > 0
    # reserve a power key column
    power_key_colname = "power"
    assert power_key_colname != count_column_name
    assert power_key_colname not in d.column_names
    # get a pandas namespace
    local_data_model = data_algebra.data_model.default_data_model()
    # build powers of 2 until max_count is met or exceeded
    powers = list(range(int(numpy.ceil(numpy.log(max_count) / numpy.log(2))) + 1))
    # replicate each power the number of times it specifies
    count_frame = local_data_model.concat_rows(
        [
            local_data_model.data_frame(
                {
                    power_key_colname: f"p{p}",
                    seq_column_name: range(int(2**p)),
                }
            )
            for p in powers
        ]
    )
    local_data_model.clean_copy(count_frame)
    # specify ops that produce row replicates
    ops = (
        d
        # specify which power table we want to join with
        .extend(
            {
                power_key_colname: f'"p" %+% ({count_column_name}.log() / (2).log()).ceil().as_int64()'
            }
        )
        # get one row for each number less than or equal to power by under-specified join
        .natural_join(
            b=TableDescription(
                table_name=join_temp_name,
                column_names=[power_key_colname, seq_column_name],
            ),
            on=[power_key_colname],
            jointype="inner",
        )
        # drop rows exceeding desired count
        .select_rows(f"{seq_column_name} < {count_column_name}")
        # drop the power group column id
        .drop_columns([power_key_colname])
    )
    return ops, count_frame
