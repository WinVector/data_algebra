
"""
data algebra solutions to common data processing problems
"""

from typing import Iterable, Optional
from data_algebra.data_ops import ViewRepresentation
from data_algebra.cdata import melt_specification


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
    record_map_to = melt_specification(
        row_keys=row_keys,
        col_name_key=col_name_key,
        col_value_key=col_value_key,
        value_cols=cols_to_map)
    record_map_back = melt_specification(
        row_keys=row_keys,
        col_name_key=col_name_key,
        col_value_key=mapped_value_key,
        value_cols=cols_to_map).inverse()
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
