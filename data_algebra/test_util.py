"""
Utils that help with testing. This module is allowed to import many other modules.
"""

import numpy

import data_algebra

import pickle
import hashlib

import data_algebra.db_model
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

run_direct_ops_path_tests = False

# global test result cache
global_test_result_cache = None


def formats_to_self(ops) -> bool:
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
        {
            "pd": data_algebra.default_data_model.pd
        },  # make our definition of pandas available
    )
    # str2 = repr(ops2)
    # strings_match = str1 == str2  # probably too strict
    ops_match = ops == ops2
    assert ops_match
    pickle_string = pickle.dumps(ops)
    ops_3 = pickle.loads(pickle_string)
    assert ops == ops_3
    return ops_match


def equivalent_frames(
    a,
    b,
    *,
    float_tol: float = 1e-8,
    check_column_order: bool = False,
    cols_case_sensitive: bool = False,
    check_row_order: bool = False,
    local_data_model=None,
) -> bool:
    """return False if the frames are equivalent (up to column re-ordering and possible row-reordering).
    Ignores indexing."""
    # leave in extra checks as this is usually used by test code
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    assert isinstance(a, local_data_model.pd.DataFrame)
    assert isinstance(b, local_data_model.pd.DataFrame)
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
    a_columns = [c for c in a.columns]
    b_columns = [c for c in b.columns]
    if set(a_columns) != set(b_columns):
        return False
    if check_column_order:
        if not numpy.all([a.columns[i] == b.columns[i] for i in range(a.shape[0])]):
            return False
    else:
        # re-order b into a's column order
        b = b[a_columns]
        b = b.reset_index(drop=True)
    for c in a_columns:
        if local_data_model.can_convert_col_to_numeric(
            a[c]
        ) != local_data_model.can_convert_col_to_numeric(b[c]):
            return False
    if a.shape[0] < 1:
        return True
    if not check_row_order:
        a = a.sort_values(by=a_columns)
        a = a.reset_index(drop=True)
        b = b.sort_values(by=a_columns)
        b = b.reset_index(drop=True)
    for c in a_columns:
        ca = a[c]
        cb = b[c]
        if (ca is None) != (cb is None):
            return False
        if ca is not None:
            if len(ca) != len(cb):
                return False
            ca_null = ca.isnull()
            cb_null = cb.isnull()
            if (ca_null is None) != (cb_null is None):
                return False
            if not numpy.all(ca_null == cb_null):
                return False
            if not numpy.all(ca_null):
                ca = ca[ca_null == False]
                cb = cb[cb_null == False]
                if local_data_model.can_convert_col_to_numeric(a[c]):
                    ca = numpy.asarray(ca, dtype=float)
                    cb = numpy.asarray(cb, dtype=float)
                    dif = numpy.abs(ca - cb) / numpy.maximum(numpy.maximum(ca, cb), 1)
                    if numpy.max(dif) > float_tol:
                        return False
                else:
                    if not numpy.all(ca == cb):
                        return False
    return True


def hash_data_frame(d) -> str:
    """
    Get a hash code representing a data frame.

    :param d: data frame
    :return: hash code as a string
    """
    return hashlib.sha256(
        data_algebra.default_data_model.pd.util.hash_pandas_object(d).values
    ).hexdigest()


def _run_handle_experiments(
    *,
    db_handle,
    data: Dict,
    ops: ViewRepresentation,
    sql_statements: Iterable[str],
    expect,
    float_tol: float = 1e-8,
    check_column_order: bool = False,
    cols_case_sensitive: bool = False,
    check_row_order: bool = False,
    test_result_cache: Optional[dict] = None,
    alter_cache: bool = True,
    test_direct_ops_path=False,
):
    """
    Run ops and sql_statements on db_handle, checking if result matches expect.
    """
    assert isinstance(db_handle, data_algebra.db_model.DBHandle)
    assert isinstance(db_handle.db_model, data_algebra.db_model.DBModel)
    assert isinstance(ops, ViewRepresentation)
    assert db_handle.conn is not None
    if isinstance(db_handle.db_model, data_algebra.SQLite.SQLiteModel):
        test_direct_ops_path = True
    db_handle_key = str(db_handle.db_model)
    sql_statements = list(sql_statements)
    res_db_sql = list([None] * len(sql_statements))  # extra list() wrapper for PyCharm's type checker
    res_db_ops = None
    need_to_run = True
    dict_keys = list(data.keys())
    dict_keys.sort()
    data_key = " ".join([k + ":" + hash_data_frame(data[k]) for k in dict_keys])

    def mk_key(ii):
        """
        Build sql statement key.
        """
        return db_handle_key + " " + sql_statements[ii] + " " + data_key

    # inspect result cache for any prior results
    if test_result_cache is not None:
        for i in range(len(sql_statements)):
            try:
                res_db_sql[i] = test_result_cache[mk_key(i)].copy()
            except KeyError:
                pass
        need_to_run = test_direct_ops_path or numpy.any(
            [result_i is None for result_i in res_db_sql]
        )
    # generate any new needed results
    if need_to_run:
        to_del = set()
        caught = None
        for (k, v) in data.items():
            db_handle.insert_table(v, table_name=k, allow_overwrite=True)
            to_del.add(k)
        try:
            if test_direct_ops_path is None:
                res_db_ops = db_handle.read_query(ops)
                assert res_db_ops is not None
            for i in range(len(sql_statements)):
                if res_db_sql[i] is None:
                    res_db_sql[i] = db_handle.read_query(sql_statements[i])
                    if (
                        alter_cache
                        and (test_result_cache is not None)
                        and (res_db_sql[i] is not None)
                    ):
                        test_result_cache[mk_key(i)] = res_db_sql[i].copy()
        except Exception as e:
            caught = e
        for k in to_del:
            db_handle.drop_table(k)
        if caught is not None:
            raise ValueError(f"{db_handle} error in test " + str(caught))
    # check results
    for res in res_db_sql:
        assert res is not None
        if not equivalent_frames(
            res,
            expect,
            float_tol=float_tol,
            check_column_order=check_column_order,
            cols_case_sensitive=cols_case_sensitive,
            check_row_order=check_row_order,
        ):
            raise ValueError(f"{db_handle} SQL result did not match expect")
    if res_db_ops is not None:
        if not equivalent_frames(
            res_db_ops,
            expect,
            float_tol=float_tol,
            check_column_order=check_column_order,
            cols_case_sensitive=cols_case_sensitive,
            check_row_order=check_row_order,
        ):
            raise ValueError(f"{db_handle} ops result did not match expect")


# noinspection PyShadowingNames
def check_transform_on_handles(
    *,
    ops,
    data,
    expect,
    db_handles,
    float_tol: float = 1e-8,
    check_column_order: bool = False,
    cols_case_sensitive: bool = False,
    check_row_order: bool = False,
    check_parse: bool = True,
    local_data_model=None,
    empty_produces_empty: bool = True,
) -> None:
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
    :param empty_produces_empty: logical, if true assume emtpy inputs should produce empty output
    :return: None, assert if there is an issue
    """

    # convert single table to dictionary
    if not isinstance(data, dict):
        cols_used = ops.columns_used()
        table_name = [k for k in cols_used.keys()][0]
        data = {table_name: data}
    assert isinstance(data, dict)
    n_tables = len(data)
    assert n_tables > 0
    if local_data_model is None:
        local_data_model = data_algebra.default_data_model
    assert isinstance(ops, ViewRepresentation)
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
    # try on empty inputs
    empty_map = {
        k: v.iloc[range(0), :].reset_index(drop=True, inplace=False)
        for k, v in data.items()
    }
    empty_res = ops.eval(empty_map)
    assert local_data_model.is_appropriate_data_instance(empty_res)
    assert set(empty_res.columns) == set(res.columns)
    if empty_produces_empty:
        assert empty_res.shape[0] == 0
    else:
        assert empty_res.shape[0] > 0
    # try on combinations of empty and original
    if n_tables == 2:
        keys = [k for k in data.keys()]
        partial_maps = [
            {
                keys[0]: data[keys[0]].reset_index(drop=True, inplace=False),
                keys[1]: data[keys[1]],
            },
            {
                keys[0]: data[keys[0]],
                keys[1]: data[keys[1]].reset_index(drop=True, inplace=False),
            },
        ]
        for pm in partial_maps:
            empty_res_i = ops.eval(pm)
            assert set(empty_res_i.columns) == set(res.columns)
    # try any db paths
    global global_test_result_cache
    global run_direct_ops_path_tests
    if db_handles is not None:
        for db_handle in db_handles:
            sql_statements = set()
            for initial_commas in [True, False]:
                for use_with in [True, False]:
                    for annotate in [True, False]:
                        sql_format_options = data_algebra.db_model.SQLFormatOptions(
                            use_with=use_with,
                            annotate=annotate,
                            initial_commas=initial_commas,
                        )
                        sql = db_handle.to_sql(
                            ops, sql_format_options=sql_format_options,
                        )
                        assert isinstance(sql, str)
                        sql_statements.add(sql)
            if db_handle.conn is not None:
                _run_handle_experiments(
                    db_handle=db_handle,
                    data=data,
                    ops=ops,
                    sql_statements=sql_statements,
                    expect=expect,
                    float_tol=float_tol,
                    check_column_order=check_column_order,
                    cols_case_sensitive=cols_case_sensitive,
                    check_row_order=check_row_order,
                    test_result_cache=global_test_result_cache,
                    alter_cache=True,
                    test_direct_ops_path=run_direct_ops_path_tests,
                )


def get_test_dbs():
    """
    handles connected to databases for testing.

    """
    db_handles = []
    hdl = data_algebra.SQLite.example_handle()
    assert hdl is not None
    assert hdl.conn is not None
    db_handles.append(hdl)
    if test_PostgreSQL:
        hdl = data_algebra.PostgreSQL.example_handle()
        assert hdl is not None
        assert hdl.conn is not None
        db_handles.append(hdl)
    if test_BigQuery:
        hdl = data_algebra.BigQuery.example_handle()
        assert hdl is not None
        assert hdl.conn is not None
        db_handles.append(hdl)
    if test_MySQL:
        hdl = data_algebra.MySQL.example_handle()
        assert hdl is not None
        assert hdl.conn is not None
        db_handles.append(hdl)
    if test_Spark:
        hdl = data_algebra.SparkSQL.example_handle()
        assert hdl is not None
        assert hdl.conn is not None
        db_handles.append(hdl)
    return db_handles


# noinspection PyShadowingNames
def check_transform(
    ops,
    data,
    expect,
    *,
    float_tol: float = 1e-8,
    check_column_order: bool = False,
    cols_case_sensitive: bool = False,
    check_row_order: bool = False,
    check_parse: bool = True,
    models_to_skip: Optional[Iterable[str]] = None,
    empty_produces_empty: bool = True,
) -> None:
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
    :param models_to_skip: None or set of model names to skip testing
    :param empty_produces_empty: logical, if true assume emtpy inputs should produce empty output
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
    ]

    test_dbs = get_test_dbs()
    db_handles = db_handles + test_dbs

    if models_to_skip is not None:
        models_to_skip = set(models_to_skip)
        db_handles = [h for h in db_handles if str(h.db_model) not in models_to_skip]

    caught = None
    try:
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
            empty_produces_empty=empty_produces_empty,
        )
    except Exception as exc:
        caught = exc

    for handle in db_handles:
        # noinspection PyBroadException
        try:
            handle.close()
        except Exception:
            pass

    if caught is not None:
        raise ValueError("testing caught " + str(caught))
