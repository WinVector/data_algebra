import collections
import pandas


def od(**kwargs):
    """Capture arguments in order."""
    r = collections.OrderedDict()
    for (k, v) in kwargs.items():
        r[k] = v
    return r


def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        x + 0.0
        return True
    except TypeError:
        return False


# for testing


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
    if not isinstance(a, pandas.DataFrame):
        raise TypeError("Expect a to be pandas.DataFrame")
    if not isinstance(b, pandas.DataFrame):
        raise TypeError("Expect b to be pandas.DataFrame")
    if a.shape != b.shape:
        return False
    if a.shape[1] < 1:
        return True
    a = a.reset_index(inplace=False, drop=True)
    b = b.reset_index(inplace=False, drop=True)
    if not cols_case_sensitive:
        a.columns = [c.lower() for c in a.columns]
        a.reset_index(inplace=True, drop=True)
        b.columns = [c.lower() for c in b.columns]
        b.reset_index(inplace=True, drop=True)
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
        b.reset_index(inplace=True, drop=True)
    for j in range(a.shape[1]):
        if can_convert_v_to_numeric(a.iloc[:, j]) != can_convert_v_to_numeric(
            b.iloc[:, j]
        ):
            return False
    if not check_row_order:
        a = a.sort_values(by=acols, inplace=False)
        a = a.reset_index(inplace=False, drop=True)
        b = b.sort_values(by=acols, inplace=False)
        b = b.reset_index(inplace=False, drop=True)
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
            dif = ca - cb
            if dif.abs().max() > float_tol:
                return False
        else:
            if not all([ca[i] == cb[i] for i in range(a.shape[0])]):
                return False
    return True
