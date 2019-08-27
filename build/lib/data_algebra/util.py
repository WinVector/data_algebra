
import pandas

def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        x + 0.0
        return True
    except TypeError:
        return False


def equivalent_frames(a, b,
                      *,
                      float_tol=1e-8,
                      check_column_order=False,
                      check_row_order=False):
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering)"""
    if not isinstance(a, pandas.DataFrame):
        raise TypeError("Expect a to be pandas.DataFrame")
    if not isinstance(b, pandas.DataFrame):
        raise TypeError("Expect b to be pandas.DataFrame")
    if a.shape != b.shape:
        return False
    if a.shape[1] < 1:
        return True
    cols = [c for c in a.columns]
    if set(cols) != set([c for c in b.columns]):
        return False
    if check_column_order:
        if not all([a.columns[i] == b.columns[i] for i in range(a.shape[0])]):
            return False
    else:
        b = b[cols]
    for i in range(a.shape[0]):
        if can_convert_v_to_numeric(a.iloc[:, i]) != can_convert_v_to_numeric(b.iloc[:, i]):
            return False
    if not check_row_order:
        a = a.sort_values(by=cols, inplace=False)
        b = b.sort_values(by=cols, inplace=False)
    for i in range(a.shape[0]):
        ca = a.iloc[:, i]
        cb = b.iloc[:, i]
        if can_convert_v_to_numeric(a.iloc[:, i]):
            dif = ca - cb
            if dif.abs().max() > float_tol:
                return False
        else:
            if not all([ca[i] == cb[i] for i in range(a.shape[0])]):
                return False
    return True
