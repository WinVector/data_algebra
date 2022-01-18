
"""
data algebra solutions to common data processing problems
"""

from typing import Iterable, Optional

import numpy
import data_algebra
from data_algebra.data_ops import descr, ViewRepresentation
from data_algebra.cdata import pivot_specification, unpivot_specification, RecordMap, RecordSpecification


def def_multi_column_map(
        d: ViewRepresentation,
        *,
        mapping_table: ViewRepresentation,
        row_keys: Iterable[str],
        col_name_key: str = 'column_name',
        col_value_key: str = 'column_value',
        mapped_value_key: str = 'mapped_value',
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
        value_cols=cols_to_map)
    record_map_back = pivot_specification(
        row_keys=row_keys,
        col_name_key=col_name_key,
        col_value_key=mapped_value_key,
        value_cols=cols_to_map)
    ops = (
        d
            .select_columns(row_keys + cols_to_map)
            .convert_records(record_map_to)
            .natural_join(
                b=mapping_table
                    .select_columns([col_name_key, col_value_key, mapped_value_key]),
                jointype='left',
                by=[col_name_key, col_value_key],
                )
        )
    if coalesce_value is not None:
        ops = ops.extend({mapped_value_key: f'{mapped_value_key}.coalesce({coalesce_value})'})
    ops = ops.convert_records(record_map_back)
    if cols_to_map_back is not None:
        # could do this in the record mapping, but this seems easier to read
        ops = ops.rename_columns({new_name: old_name for new_name, old_name in zip(cols_to_map_back, cols_to_map)})
    return ops


def xicor_query(
        data: ViewRepresentation,
        *,
        x_name: str = 'x',
        y_name: str = 'y',
        var_keys: Iterable[str] = tuple()):
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
        x_name, y_name, x_tie_breaker, y_group,
        'l', 'n', 'r',
        'rplus', 'rdiff', 'lterm', 'num_sum', 'den_sum',
        'xicor'
        ] + var_keys
    assert(len(names) == len(set(names)))
    ops = (
        data
            .extend({y_group: f"{y_name}.as_str()"})  # Google BigQuery won't group by float
            .extend({    # convert types, and add in tie breaking column
                x_name: f"1.0 * {x_name}",
                y_name: f"1.0 * {y_name}",
                x_tie_breaker: "_uniform()"})
            .extend(
                {"n": "(1).sum()"}, partition_by=var_keys)  # annotate in number of rows
            .extend(  # compute y ranks, that we will use to compare rank changes wrt x
                {"r": "(1).cumsum()"}, order_by=[y_name], partition_by=var_keys)
            .extend(  # compute reverse y ranks, used to normalize for ties in denominator
                {"l": "(1).cumsum()"}, order_by=[y_name], reverse=[y_name], partition_by=var_keys)
            .extend(  # go to max rank of group tie breaking
                {"l": "l.max()", "r": "r.max()"}, partition_by=[y_group] + var_keys)
            .extend(  # get y rank and y rank of next x-item into same row so we can take a difference
                {"rplus": "r.shift(1)"},
                order_by=[x_name, x_tie_breaker],
                reverse=[x_name, x_tie_breaker],
                partition_by=var_keys,
                )
            .extend(  # compute numerator and denominator terms
                {"rdiff": "((rplus - r).abs()).coalesce(0)", "lterm": "l * (n - l)"})
            .project(   # aggregate to compute sums in xicor definition
                {"num_sum": "rdiff.sum()", "den_sum": "lterm.sum()",
                 "n": "n.max()"  # pseudo-aggregation n is constant across rows
                 },
                group_by=var_keys,
                )
            .extend(  # apply actual xicor formula
                {"xicor": "1.0 - ((n * num_sum) / (2.0 * den_sum))"})
            .select_columns(var_keys + ["xicor"])
        )
    return ops


def xicor_score_variables_plan(
        d: ViewRepresentation,
        *,
        x_vars: Iterable[str],
        y_name: str,
        n_rep: int = 25,
):
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
            control_table=data_algebra.pandas_model.pd.DataFrame({
                'variable_name': x_vars,
                'x': x_vars,
                'y': y_name,

            }),
            record_keys=[],
            control_table_keys=['variable_name'])
    )
    rep_frame = data_algebra.default_data_model.pd.DataFrame({'rep': range(n_rep)})
    grouped_calc = (
        xicor_query(
            d
                .convert_records(record_map)
                .natural_join(  # cross join rows to get experiment repetitions
                        b=descr(rep_frame=rep_frame),
                        by=[],
                        jointype='cross',
                    ),
                var_keys=['variable_name', 'rep'])
            .project({
                    'xicor_mean': 'xicor.mean()',
                    'xicor_std': 'xicor.std()',
                },
                group_by=['variable_name'])
            .order_rows(['variable_name'])
    )
    return grouped_calc, 'rep_frame', rep_frame
