import collections
import numpy
import pandas

import data_algebra.data_types


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


def is_bad(x):
    """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
    if can_convert_v_to_numeric(x):
        x = numpy.asarray(x + 0, dtype=float)
        return numpy.logical_or(
            pandas.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
        )
    return pandas.isnull(x)


# for testing


def pandas_to_example_str(obj):
    if not isinstance(obj, pandas.DataFrame):
        raise TypeError("Expect obj to be pandas.DataFrame")
    pstr = "pandas.DataFrame({"
    for k in obj.columns:
        cells = ["None" if pandas.isnull(v) else v.__repr__() for v in obj[k]]
        pstr = pstr + "\n    " + k.__repr__() + ": [" + ", ".join(cells) + "],"
    pstr = pstr + "\n    })"
    return pstr


def equivalent_frames(
    a,
    b,
    *,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False
):
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing."""
    a = data_algebra.data_types.convert_to_pandas_dataframe(a, "a")
    b = data_algebra.data_types.convert_to_pandas_dataframe(b, "b")
    # leave in extra checks as this is usually used by test code
    if not isinstance(a, pandas.DataFrame):
        raise TypeError("Expect a to be pandas.DataFrame")
    if not isinstance(b, pandas.DataFrame):
        raise TypeError("Expect b to be pandas.DataFrame")
    if a.shape != b.shape:
        return False
    if a.shape[1] < 1:
        return True
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if not cols_case_sensitive:
        a.columns = [c.lower() for c in a.columns]
        a = a.reset_index(drop=True)
        b.columns = [c.lower() for c in b.columns]
        b = b.reset_index(drop=True)
    acols = [c for c in a.columns]
    bcols = [c for c in b.columns]
    if set(acols) != set(bcols):
        return False
    if check_column_order:
        if not all([a.columns[i] == b.columns[i] for i in range(a.shape[0])]):
            return False
    else:
        # re-order b into a's column order
        b = b[acols]
        b = b.reset_index(drop=True)
    for j in range(a.shape[1]):
        if can_convert_v_to_numeric(a.iloc[:, j]) != can_convert_v_to_numeric(
            b.iloc[:, j]
        ):
            return False
    if not check_row_order:
        a = a.sort_values(by=acols)
        a = a.reset_index(drop=True)
        b = b.sort_values(by=acols)
        b = b.reset_index(drop=True)
    for j in range(a.shape[1]):
        ca = a.iloc[:, j]
        cb = b.iloc[:, j]
        ca_null = ca.isnull()
        cb_null = cb.isnull()
        if (ca_null is None) != (cb_null is None):
            return False
        if ca is not None:
            if not all([ca_null[i] == cb_null[i] for i in range(a.shape[0])]):
                return False
        if can_convert_v_to_numeric(ca):
            ca = numpy.asarray(ca, dtype=float)
            cb = numpy.asarray(cb, dtype=float)
            dif = ca - cb
            dif = numpy.asarray([abs(d) for d in dif if not pandas.isnull(d)])
            if dif.max() > float_tol:
                return False
        else:
            if not all([ca[i] == cb[i] for i in range(a.shape[0])]):
                return False
    return True



def table_is_keyed_by_columns(table, column_names):
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
    return counts.max() <= 1
