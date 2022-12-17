"""
Basic utilities. Not allowed to import many other modules.
"""

import datetime
import warnings
from typing import Dict, Iterable, Optional, Tuple

import numpy

import data_algebra.data_model


def pandas_to_example_str(obj, *, local_data_model=None) -> str:
    """
    Convert data frame to a Python source code string.

    :param obj: data frame to convert.
    :param local_data_model: data model to use.
    :return: Python source code representation of obj.
    """
    if local_data_model is None:
        local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(obj)
    pd_module_name = local_data_model.presentation_model_name
    if not local_data_model.is_appropriate_data_instance(obj):
        raise TypeError("Expect obj to be local_data_model.pd.DataFrame")
    obj = local_data_model.clean_copy(obj)
    nrow = obj.shape[0]
    pandas_string = pd_module_name + ".DataFrame({"
    for k in obj.columns:
        col = obj[k]
        nulls = local_data_model.bad_column_positions(col)
        cells = ["None" if nulls[i] else col[i].__repr__() for i in range(nrow)]
        pandas_string = (
            pandas_string + "\n    " + k.__repr__() + ": [" + ", ".join(cells) + "],"
        )
    pandas_string = pandas_string + "\n    })"
    return pandas_string


# noinspection PyBroadException
def _mk_type_conversion_table() -> Dict[type, type]:
    """
    Build up conversion from type aliases we do not want into standard types. Eat any errors or warnings during table
    construction.
    """

    type_conversions_table: Dict[type, type] = dict()
    # DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`.
    # To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe.
    # If you specifically wanted the numpy scalar type, use `np.bool_` here.
    #   Deprecated in NumPy 1.20; for more details and guidance:
    #   https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    # (note even numpy.bool_ triggers the above, or the triggering it is in pyspark right now
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            type_conversions_table[numpy.bool_] = bool
        except Exception:
            pass
        try:
            type_conversions_table[numpy.bool] = bool
        except Exception:
            pass
        try:
            type_conversions_table[numpy.int] = int
        except Exception:
            pass
        try:
            type_conversions_table[numpy.int_] = int
        except Exception:
            pass
        try:
            type_conversions_table[numpy.int64] = int
        except Exception:
            pass
        try:
            type_conversions_table[numpy.float64] = float
        except Exception:
            pass
        try:
            type_conversions_table[numpy.float] = float
        except Exception:
            pass
        try:
            type_conversions_table[numpy.float_] = float
        except Exception:
            pass
        try:
            type_conversions_table[numpy.str] = str
        except Exception:
            pass
        try:
            type_conversions_table[numpy.str_] = str
        except Exception:
            pass
    return type_conversions_table


type_conversions = _mk_type_conversion_table()


def map_type_to_canonical(v: type) -> type:
    """
    Map type to a smaller set of considered equivalent types.

    :param v: type to map
    :return: type
    """
    try:
        return type_conversions[v]
    except KeyError:
        pass
    return v


def guess_carried_scalar_type(col) -> type:
    """
    Guess the type of a column or scalar.

    :param col: column or scalar to inspect
    :return: type of first non-None entry, if any , else type(None)
    """
    # check for scalars first
    ct = map_type_to_canonical(type(col))
    if ct in {
        str,
        int,
        float,
        bool,
        type(None),
        numpy.int64,
        numpy.float64,
        datetime.datetime,
        datetime.date,
        datetime.timedelta,
    }:
        return ct
    # look at a list or Series
    if isinstance(col, data_algebra.data_model.default_data_model().pd.core.series.Series):
        col = col.values
    if len(col) < 1:
        return type(None)
    good_idx = numpy.where(
        numpy.logical_not(data_algebra.data_model.default_data_model().pd.isna(col))
    )[0]
    test_idx = 0
    if len(good_idx) > 0:
        test_idx = good_idx[0]
    return map_type_to_canonical(type(col[test_idx]))


def guess_column_types(
    d, *, columns: Optional[Iterable[str]] = None
) -> Dict[str, type]:
    """
    Guess column types as type of first non-missing value.
    Will not return series types, as some pandas data frames with non-trivial indexing report this type.

    :param d: pandas.DataFrame
    :param columns: list of columns to check, if None all columns are checked
    :return: map of column names to guessed types, empty dict if any column guess fails
    """
    if (d.shape[0] <= 0) or (d.shape[1] <= 0):
        return dict()
    if columns is None:
        columns = list(d.columns)
    else:
        columns = list(columns)
    assert len(set(columns) - set(d.columns)) == 0
    if len(columns) <= 0:
        return dict()
    res = dict()
    for c in columns:
        gt = guess_carried_scalar_type(d[c])
        if (
            (gt is None)
            or (not isinstance(gt, type))
            or gt == data_algebra.data_model.default_data_model().pd.core.series.Series
        ):
            # pandas.concat() poisons types with Series, don't allow that
            return dict()
        res[c] = gt
    return res


def compatible_types(types_seen: Iterable[type]) -> bool:
    """
    Check if a set of types are all considered equivalent.

    :param types_seen: collection of types seen
    :return: True if types are all compatible, else False.
    """
    mapped_comparison = {map_type_to_canonical(t) for t in types_seen} - {type(None)}
    if (len(mapped_comparison) > 1) and (mapped_comparison != {int, float}):
        return False
    return True


def check_columns_appear_compatible(
    d_left, d_right, *, columns: Optional[Iterable[str]] = None
) -> Optional[Dict[str, Tuple[type, type]]]:
    """
    Check if columns have compatible types

    :param d_left: pandas dataframe to check
    :param d_right: pandas dataframe to check
    :param columns: columns to check, None means check all columns
    :return: None if compatible, else dictionary of mismatches
    """
    if columns is None:
        columns = [c for c in d_left.columns]
        assert set(d_left.columns) == set(d_right.columns)
    else:
        columns = [c for c in columns]
    assert len(set(columns) - set(d_left.columns)) == 0
    assert len(set(columns) - set(d_right.columns)) == 0
    left_types = data_algebra.util.guess_column_types(d_left, columns=columns)
    if (left_types is None) or (len(left_types) <= 0):
        return None
    right_types = data_algebra.util.guess_column_types(d_right, columns=columns)
    if (right_types is None) or (len(right_types) <= 0):
        return None
    mismatches = dict()
    for c in columns:
        if not compatible_types([left_types[c], right_types[c]]):
            mismatches[c] = (left_types[c], right_types[c])
    if len(mismatches) > 0:
        return mismatches
    return None
