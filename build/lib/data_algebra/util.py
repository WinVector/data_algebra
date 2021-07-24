
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
        pandas_string = pandas_string + "\n    " + k.__repr__() + ": [" + ", ".join(cells) + "],"
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


def guess_column_types(d, *, columns=None):
    """
    Guess column types as type of first non-missing value

    :param d: pandas.DataFrame
    :param columns: list of columns to check, if None all columns are checked
    :return: map of column names to guessed types
    """
    if d.shape[1] <= 0:
        return dict()
    if columns is None:
        columns = d.columns
    assert len(set(columns) - set(d.columns)) == 0
    if d.shape[0] <= 0:
        return {c: type(None) for c in columns}
    res = dict()
    for c in columns:
        col = d[c]
        idx = col.notna().idxmax()
        if idx is None:
            res[c] = type(col[0])
        else:
            res[c] = type(col[idx])
    return res


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
    right_types = data_algebra.util.guess_column_types(d_right, columns=columns)
    mismatches = dict()
    for c in columns:
        comparison = ({left_types[c]}.union({right_types[c]})) - {type(None)}
        if (len(comparison) > 1) and (comparison != {numpy.float64, numpy.int64}):
            mismatches[c] = (left_types[c], right_types[c])
    if len(mismatches) > 0:
        return mismatches
    return None
