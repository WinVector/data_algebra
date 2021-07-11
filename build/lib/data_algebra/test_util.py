
import numpy

# noinspection PyUnresolvedReferences
import data_algebra

import sqlite3
import pickle

# noinspection PyUnresolvedReferences
import data_algebra.SQLite
import data_algebra.BigQuery
import data_algebra.PostgreSQL
import data_algebra.MySQL
import data_algebra.SparkSQL

from data_algebra.data_ops import *


# controls
test_PostgreSQL = False  # causes an external dependency
test_BigQuery = False  # causes an external dependency
test_MySQL = False  # causes an external dependency
test_Spark = False  # causes an external dependency


def formats_to_self(ops):
    """
    Check a operator dag formats and parses back to itself.
    Can raise exceptions. Also checks pickling.

    :param ops: data_algebra.data_ops.ViewRepresentation
    :return: logical, True if formats and evals back to self
    """
    str1 = repr(ops)
    ops2 = eval(
        str1,
        globals(),
        {'pd': data_algebra.default_data_model.pd}  # make our definition of pandas available
    )
    str2 = repr(ops2)
    strings_match = str1 == str2   # probably too strict
    ops_match = ops == ops2
    if strings_match and (not ops_match):
        raise Exception("strings match, but ops did not")
    pickle_string = pickle.dumps(ops)
    ops_3 = pickle.loads(pickle_string)
    assert ops == ops_3
    return ops_match


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
    a = a.reset_index(drop=True, inplace=False)
    b = b.reset_index(drop=True, inplace=False)
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
                ca = ca[ca_null == False]
                cb = cb[cb_null == False]
                if local_data_model.can_convert_col_to_numeric(a[c]):
                    ca = numpy.asarray(ca, dtype=float)
                    cb = numpy.asarray(cb, dtype=float)
                    dif = numpy.abs(ca - cb) / numpy.maximum(numpy.maximum(ca, cb), 1)
                    if numpy.max(dif) > float_tol:
                        return False
                else:
                    if not all(ca == cb):
                        return False
    return True


def check_transform_on_handles(
    *,
    ops,
    data,
    expect,
    db_handles,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False,
    check_parse=True,
    local_data_model=None,
    allow_pretty=False,
):
    """
    Test an operator dag produces the expected result, and parses correctly.
    Asserts if there are issues

    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data: pd.DataFrame or map of strings to pd.DataFrame
    :param expect: pd.DataFrame
    :param db_handles:  list of database handles to use in testing
    :param float_tol: passed to equivalent_frames()
    :param check_column_order: passed to equivalent_frames()
    :param cols_case_sensitive: passed to equivalent_frames()
    :param check_row_order: passed to equivalent_frames()
    :param check_parse: if True check expression parses/formats to self
    :param local_data_model: optional alternate evaluation model
    :param allow_pretty: if True try pretty printing SQL
    :return: None, assert if there is an issue
    """

    # convert single table to dictionary
    if not isinstance(data, dict):
        cols_used = ops.columns_used()
        table_name = [k for k in cols_used.keys()][0]
        data = {table_name: data}

    assert isinstance(data, dict)
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    if not isinstance(ops, ViewRepresentation):
        raise TypeError("expected ops to be a data_algebra.data_ops.ViewRepresentation")
    if not local_data_model.is_appropriate_data_instance(expect):
        raise TypeError("expected expect to be a local_data_model.pd.DataFrame")
    cols_used = ops.columns_used()
    if len(cols_used) < 1:
        raise ValueError("no tables used")
    # check all needed tables are present
    for k in cols_used.keys():
        v = data[k]
        assert local_data_model.is_appropriate_data_instance(v)
    if check_parse:
        if not formats_to_self(ops):
            raise ValueError("ops did not round-trip format")
    res = ops.eval(data_map=data)
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
        raise ValueError("Pandas eval result did not match expect")
    if len(data) == 1:
        res_t = ops.transform(list(data.values())[0])
        if not equivalent_frames(
                res_t,
                expect,
                float_tol=float_tol,
                check_column_order=check_column_order,
                cols_case_sensitive=cols_case_sensitive,
                check_row_order=check_row_order,
        ):
            raise ValueError("Pandas transform result did not match expect")
    # try any db paths
    if db_handles is not None:
        for db_handle in db_handles:
            to_del = set()
            temp_tables = None
            sql_statements = []
            if allow_pretty:
                pretty_levels = [True, False]
            else:
                pretty_levels = [False]
            for annotate in [True, False]:
                for pretty in pretty_levels:
                    for use_with in [True, False]:
                        temp_tables = dict()
                        sql = db_handle.to_sql(
                            ops,
                            pretty=pretty,
                            annotate=annotate,
                            use_with=use_with,
                            temp_tables=temp_tables)
                        assert isinstance(sql, str)
                        sql_statements.append(sql)
                        # print(sql)
            if db_handle.conn is not None:
                for (k, v) in data.items():
                    db_handle.insert_table(v, table_name=k, allow_overwrite=True)
                    to_del.add(k)
                for (k, v) in temp_tables.items():
                    db_handle.insert_table(v, table_name=k, allow_overwrite=True)
                    to_del.add(k)
                caught = None
                res_db_sql = []
                res_db_ops = None
                run_ops_version = len(temp_tables) <= 0
                try:
                    for sql in sql_statements:
                        res_db_sql_i = db_handle.read_query(sql)
                        res_db_sql.append(res_db_sql_i)
                    if run_ops_version:
                        res_db_ops = db_handle.read_query(ops)
                except Exception as e:
                    caught = e
                for k in to_del:
                    db_handle.drop_table(k)
                if caught is not None:
                    raise caught
                for res in res_db_sql:
                    if not equivalent_frames(
                        res,
                        expect,
                        float_tol=float_tol,
                        check_column_order=check_column_order,
                        cols_case_sensitive=cols_case_sensitive,
                        check_row_order=check_row_order,
                    ):
                        raise ValueError(f"{db_handle} SQL result did not match expect")
                if run_ops_version:
                    if not equivalent_frames(
                        res_db_ops,
                        expect,
                        float_tol=float_tol,
                        check_column_order=check_column_order,
                        cols_case_sensitive=cols_case_sensitive,
                        check_row_order=check_row_order,
                    ):
                        raise ValueError(f"{db_handle} ops result did not match expect")


def check_transform(
    ops,
    data,
    expect,
    *,
    float_tol=1e-8,
    check_column_order=False,
    cols_case_sensitive=False,
    check_row_order=False,
    check_parse=True,
    allow_pretty=True,
    models_to_skip=None,
):
    """
    Test an operator dag produces the expected result, and parses correctly.
    Assert if there are issues.

    :param ops: data_algebra.data_ops.ViewRepresentation
    :param data: pd.DataFrame or map of strings to pd.DataFrame
    :param expect: pd.DataFrame
    :param float_tol: passed to equivalent_frames()
    :param check_column_order: passed to equivalent_frames()
    :param cols_case_sensitive: passed to equivalent_frames()
    :param check_row_order: passed to equivalent_frames()
    :param check_parse: if True check expression parses/formats to self
    :param allow_pretty: if True try pretty printing SQL
    :param models_to_skip: None or set of model names to skip testing
    :return: nothing
    """

    # convert single table to dictionary
    if not isinstance(data, dict):
        cols_used = ops.columns_used()
        table_name = [k for k in cols_used.keys()][0]
        data = {table_name: data}

    db_handles = [
        # non-connected handles, lets us test some of the SQL generation path
        data_algebra.SQLite.SQLiteModel().db_handle(None),
        data_algebra.BigQuery.BigQueryModel().db_handle(None),
        data_algebra.PostgreSQL.PostgreSQLModel().db_handle(None),
        data_algebra.SparkSQL.SparkSQLModel().db_handle(None),
        data_algebra.MySQL.MySQLModel().db_handle(None),
        data_algebra.SQLite.example_handle(),  # actual database instance, not empty
        ]

    if test_PostgreSQL:
        db_handles.append(data_algebra.PostgreSQL.example_handle())
    if test_BigQuery:
        db_handles.append(data_algebra.BigQuery.example_handle())
    if test_MySQL:
        db_handles.append(data_algebra.MySQL.example_handle())
    if test_Spark:
        db_handles.append(data_algebra.SparkSQL.example_handle())

    if models_to_skip is not None:
        db_handles = [h for h in db_handles if str(h.db_model) not in models_to_skip]

    check_transform_on_handles(
        ops=ops,
        data=data,
        expect=expect,
        float_tol=float_tol,
        check_column_order=check_column_order,
        cols_case_sensitive=cols_case_sensitive,
        check_row_order=check_row_order,
        check_parse=check_parse,
        db_handles=db_handles,
        allow_pretty=allow_pretty,
    )

    for handle in db_handles:
        handle.close()
