import collections
import numpy

import data_algebra


def od(**kwargs):
    """Capture arguments in order."""
    r = collections.OrderedDict()
    for (k, v) in kwargs.items():
        r[k] = v
    return r


def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        numpy.asarray(x + 0, dtype=float)
        return True
    except TypeError:
        return False


def is_bad(x, *, pd=None):
    """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
    if pd is None:
        pd = data_algebra.pd
    if can_convert_v_to_numeric(x):
        x = numpy.asarray(x + 0, dtype=float)
        return numpy.logical_or(
            pd.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
        )
    return pd.isnull(x)


def pandas_to_example_str(obj, *, pd=None, pd_module_name="data_algebra.pd"):
    if pd is None:
        pd = data_algebra.pd
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Expect obj to be pd.DataFrame")
    pstr = pd_module_name + ".DataFrame({"
    for k in obj.columns:
        cells = ["None" if pd.isnull(v) else v.__repr__() for v in obj[k]]
        pstr = pstr + "\n    " + k.__repr__() + ": [" + ", ".join(cells) + "],"
    pstr = pstr + "\n    })"
    return pstr


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
