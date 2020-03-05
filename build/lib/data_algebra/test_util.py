# needed for the eval
# noinspection PyUnresolvedReferences
import pandas
import numpy

# noinspection PyUnresolvedReferences
import data_algebra

import sqlite3

# noinspection PyUnresolvedReferences
import data_algebra.SQLite
from data_algebra.data_ops import *
from data_algebra.yaml import have_yaml, to_pipeline


def formats_to_self(ops):
    """
    Check a operator dag formats and parses back to itself

    :param ops: data_algebra.data_ops.ViewRepresentation
    :return: logical, True if formats and evals back to self
    """
    str1 = repr(ops)
    ops2 = eval(str1)
    str2 = repr(ops2)
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
    assert o._equiv_nodes(back)


def equivalent_frames(
    a,
    b,
    *,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False,
    local_data_model=None,
):
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing."""
    # leave in extra checks as this is usually used by test code
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    if not isinstance(a, local_data_model.pd.DataFrame):
        raise TypeError("Expect a to be local_data_model.pd.DataFrame")
    if not isinstance(b, local_data_model.pd.DataFrame):
        raise TypeError("Expect b to be local_data_model.pd.DataFrame")
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
    for c in acols:
        if local_data_model.can_convert_col_to_numeric(
            a[c]
        ) != local_data_model.can_convert_col_to_numeric(b[c]):
            return False
    if a.shape[0] < 1:
        return True
    if not check_row_order:
        a = a.sort_values(by=acols)
        a = a.reset_index(drop=True)
        b = b.sort_values(by=acols)
        b = b.reset_index(drop=True)
    for c in acols:
        ca = a[c]
        cb = b[c]
        if (ca is None) != (cb is None):
            return False
        if ca is not None:
            ca_null = ca.isnull()
            cb_null = cb.isnull()
            if (ca_null is None) != (cb_null is None):
                return False
            if not all(ca_null == cb_null):
                return False
            if not all(ca_null):
                ca = ca[~ca_null]
                cb = cb[~cb_null]
                if local_data_model.can_convert_col_to_numeric(a[c]):
                    ca = numpy.asarray(ca, dtype=float)
                    cb = numpy.asarray(cb, dtype=float)
                    dif = abs(ca - cb)
                    if numpy.max(dif) > float_tol:
                        return False
                else:
                    if not all(ca == cb):
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
    local_data_model=None,
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
    :param pd pandas module (defaults to data_algebra.default_data_model if None)
    :return: None, assert if there is an issue
    """

    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    if not isinstance(ops, ViewRepresentation):
        raise TypeError("expected ops to be a data_algebra.data_ops.ViewRepresentation")
    if not local_data_model.is_appropriate_data_instance(expect):
        raise TypeError("exepcted expect to be a local_data_model.pd.DataFrame")
    cols_used = ops.columns_used()
    if len(cols_used) < 1:
        raise ValueError("no tables used")
    if not formats_to_self(ops):
        raise ValueError("ops did not round-trip format")
    check_op_round_trip(ops)
    if isinstance(data, dict):
        res = ops.eval(data_map=data)
    else:
        if len(cols_used) != 1:
            raise ValueError("more than one table used, but only one table supplied")
        if not local_data_model.is_appropriate_data_instance(data):
            raise TypeError("exepcted expect to be a local_data_model.pd.DataFrame")
        res = ops.transform(data)
    # try pandas path
    if not local_data_model.is_appropriate_data_instance(res):
        raise ValueError(
            "expected res to be local_data_model.pd.DataFrame, got: " + str(type(res))
        )
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
    db_model.prepare_connection(conn)
    if isinstance(data, dict):
        for (k, v) in data.items():
            db_model.insert_table(conn, v, table_name=k)
    else:
        table_name = [k for k in cols_used.keys()][0]
        db_model.insert_table(conn, data, table_name=table_name)
    temp_tables = dict()
    sql = ops.to_sql(db_model, pretty=True, temp_tables=temp_tables)
    for (k, v) in temp_tables.items():
        db_model.insert_table(conn, v, table_name=k)
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
