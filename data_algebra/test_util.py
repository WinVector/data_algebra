# needed for the eval
# noinspection PyUnresolvedReferences
import numpy

# noinspection PyUnresolvedReferences
import data_algebra

import sqlite3

# noinspection PyUnresolvedReferences
import data_algebra.SQLite
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


def equivalent_frames(
    a,
    b,
    *,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False,
    pd=None,
):
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing."""
    # leave in extra checks as this is usually used by test code
    if pd is None:
        pd = data_algebra.pd
    if not isinstance(a, pd.DataFrame):
        raise TypeError("Expect a to be pd.DataFrame")
    if not isinstance(b, pd.DataFrame):
        raise TypeError("Expect b to be pd.DataFrame")
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
    if a.shape[0] < 1:
        return True
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
            dif = numpy.asarray([abs(d) for d in dif if not pd.isnull(d)])
            if numpy.max(dif) > float_tol:
                return False
        else:
            if not all([ca[i] == cb[i] for i in range(a.shape[0])]):
                return False
    return True


def check_transform(
    ops,
    data,
    expect,
    *,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False,
    pd=None,
):
    """
    Test an operator dag produces the expected result, and parses correctly.

    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data: pd.DataFrame or map of strings to pd.DataFrame
    :param expect: pd.DataFrame
    :param float_tol passed to equivalent_frames()
    :param check_column_order passed to equivalent_frames()
    :param cols_case_sensitive passed to equivalent_frames()
    :param check_row_order passed to equivalent_frames()
    :param pd pandas module (defaults to data_algebra.pd if None)
    :return: None, assert if there is an issue
    """

    if pd is None:
        pd = data_algebra.pd
    if not isinstance(ops, ViewRepresentation):
        raise TypeError("expected ops to be a data_algebra.data_ops.ViewRepresentation")
    if not isinstance(expect, pd.DataFrame):
        raise TypeError("exepcted expect to be a pd.DataFrame")
    cols_used = ops.columns_used()
    if len(cols_used) < 1:
        raise ValueError("no tables used")
    if not formats_to_self(ops):
        raise ValueError("ops did not round-trip format")
    check_op_round_trip(ops)
    if isinstance(data, pd.DataFrame):
        if len(cols_used) != 1:
            raise ValueError("more than one table used, but only one table supplied")
        res = ops.transform(data)
    else:
        if not isinstance(data, Dict):
            raise TypeError(
                "expected data to be a pd.DataFrame or a dictionary of such"
            )
        res = ops.eval_pandas(data_map=data)
    # try pandas path
    if not isinstance(res, pd.DataFrame):
        raise ValueError("expected res to be pd.DataFrame, got: " + str(type(res)))
    if not equivalent_frames(
        res,
        expect,
        float_tol=float_tol,
        check_column_order=check_column_order,
        cols_case_sensitive=cols_case_sensitive,
        check_row_order=check_row_order,
    ):
        raise ValueError("Pandas result did not match expect")
    # try Sqlite path
    conn = sqlite3.connect(":memory:")
    db_model = data_algebra.SQLite.SQLiteModel()
    if isinstance(data, pd.DataFrame):
        table_name = [k for k in cols_used.keys()][0]
        db_model.insert_table(conn, data, table_name=table_name)
    else:
        for k in cols_used.keys():
            v = data[k]
            db_model.insert_table(conn, v, table_name=k)
    sql = ops.to_sql(db_model, pretty=True)
    res_db = db_model.read_query(conn, sql)
    # clean up
    conn.close()
    if not equivalent_frames(
        res_db,
        expect,
        float_tol=float_tol,
        check_column_order=check_column_order,
        cols_case_sensitive=cols_case_sensitive,
        check_row_order=check_row_order,
    ):
        raise ValueError("SQLite result did not match expect")
