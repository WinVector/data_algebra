
"""
data algebra solutions to common data processing problems
"""

from typing import Iterable
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
    col_names = row_keys + [col_name_key, col_value_key, mapped_value_key] + cols_to_map
    assert len(col_names) == len(set(col_names))
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
            .convert_records(record_map_back)
        )
    return ops
