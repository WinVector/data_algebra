
import datetime

import numpy

import data_algebra


def pandas_to_example_str(obj, *, local_data_model=None):
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    pd_module_name = local_data_model.presentation_model_name
    if not local_data_model.is_appropriate_data_instance(obj):
        raise TypeError("Expect obj to be local_data_model.pd.DataFrame")
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


def table_is_keyed_by_columns(table, column_names):
    """

    :param table: pandas DataFrame
    :param column_names: list of column names
    :return: True if rows are uniquely keyed by values in named columns
    """
    # check for ill-condition
    if isinstance(column_names, str):
        column_names = [column_names]
    missing_columns = set(column_names) - set([c for c in table.columns])
    if len(missing_columns) > 0:
        raise KeyError("missing columns: " + str(missing_columns))
    # get rid of some corner cases
    if table.shape[0] < 2:
        return True
    if len(column_names) < 1:
        return False
    counts = table.groupby(column_names).size()
    return max(counts) <= 1


type_conversions = {
    # DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`.
    # To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe.
    # If you specifically wanted the numpy scalar type, use `np.bool_` here.
    #   Deprecated in NumPy 1.20; for more details and guidance:
    #   https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    # (note even numpy.bool_ triggers the above, or the triggering it is in pyspark right now
    numpy.bool_: bool,
    numpy.int64: int,
    numpy.float64: float,
}


def map_type_to_canonical(v):
    try:
        return type_conversions[v]
    except KeyError:
        pass
    return v


def guess_carried_scalar_type(col):
    """
    Guess the type of a column or scalar.

    :param col: column or scalar to inspect
    :return: type of first non-None entry, if any , else type(None)
    """
    # check for scalars first
    ct = map_type_to_canonical(type(col))
    if ct in {str, int, float, bool, type(None), numpy.int64, numpy.float64,
              datetime.datetime, datetime.date, datetime.timedelta}:
        return ct
    # look at a list or Series
    if isinstance(col, data_algebra.default_data_model.pd.core.series.Series):
        col = col.values
    if len(col) < 1:
        return type(None)
    good_idx = numpy.where(numpy.logical_not(data_algebra.default_data_model.pd.isna(col)))[0]
    test_idx = 0
    if len(good_idx) > 0:
        test_idx = good_idx[0]
    return map_type_to_canonical(type(col[test_idx]))


def guess_column_types(d, *, columns=None):
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
        columns = d.columns.copy()
    assert len(set(columns) - set(d.columns)) == 0
    if len(columns) <= 0:
        return dict()
    res = dict()
    for c in columns:
        gt = guess_carried_scalar_type(d[c])
        if (gt is None) or (not isinstance(gt, type)) or gt == data_algebra.default_data_model.pd.core.series.Series:
            # pandas.concat() poisons types with Series, don't allow that
            return dict()
        res[c] = gt
    return res


def compatible_types(types_seen):
    mapped_comparison = {map_type_to_canonical(t) for t in types_seen} - {type(None)}
    if (len(mapped_comparison) > 1) and (mapped_comparison != {int, float}):
        return False
    return True


def check_columns_appear_compatible(d_left, d_right, *, columns=None):
    """
    Check if columns have compatible types

    :param d_left: pandas dataframe to check
    :param d_right: pandas dataframe to check
    :param columns: columns to check, None means check all columns
    :return: None if compatible, else dictionary of mismatches
    """
    if columns is None:
        columns = d_left.columns
        assert set(d_left.columns) == set(d_right.columns)
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
