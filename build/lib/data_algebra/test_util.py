# needed for the eval
# noinspection PyUnresolvedReferences
import numpy
# noinspection PyUnresolvedReferences
import pandas
import yaml

# noinspection PyUnresolvedReferences
import data_algebra
from data_algebra.data_ops import *
from data_algebra.util import can_convert_v_to_numeric
from data_algebra.yaml import have_yaml, to_pipeline


def formats_to_self(ops):
    """
    Check a operator dag formats and parses back to itself

    :param ops: data_algebra.data_ops.ViewRepresentation
    :return: logical, True if formats and evals back to self
    """
    str1 = str(ops)
    ops2 = eval(str1)
    str2 = str(ops2)
    return str1 == str2


def check_op_round_trip(o):
    if not isinstance(o, ViewRepresentation):
        raise TypeError("expect o to be a data_algebra.data_ops.ViewRepresentation")
    if not have_yaml:
        raise RuntimeError("yaml/PyYAML not installed")
    strr = o.to_python(strict=True, pretty=False)
    strp = o.to_python(strict=True, pretty=True)
    obj = o.collect_representation()
    back = to_pipeline(obj)
    strr_back = back.to_python(strict=True, pretty=False)
    assert strr == strr_back
    strp_back = back.to_python(strict=True, pretty=True)
    assert strp == strp_back
    dmp = yaml.dump(obj)
    ld = yaml.safe_load(dmp)
    back = to_pipeline(ld)
    if isinstance(o, ExtendNode):
        if len(o.ops) == 1:
            strr_back = back.to_python(strict=True, pretty=False)
            assert strr == strr_back
            strp_back = back.to_python(strict=True, pretty=True)
            assert strp == strp_back


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
            if numpy.max(dif) > float_tol:
                return False
        else:
            if not all([ca[i] == cb[i] for i in range(a.shape[0])]):
                return False
    return True
