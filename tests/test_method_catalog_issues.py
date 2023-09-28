


from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.SQLite
import data_algebra.BigQuery
import data_algebra.PostgreSQL
import data_algebra.MySQL
import data_algebra.data_model

import data_algebra.test_util


def mk_example():
    datetime_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y-%m-%d"
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'row_id': [0, 1, 2, 3],
        'a': [False, False, True, True],
        'b': [False, True, False, True],
        'q': [1, 1, 2, 2],
        'x': [.1, .2, .3, .4],
        'y': [2.4, 1.33, 1.2, 1.1],
        'z': [1.6, None, -2.1, 0],
        'g': ['a', 'a', 'b', 'ccc'],
        's2': ['z', 'q', '11', 'b'],
        "str_datetime_col": ["2000-01-01 12:13:21", "2020-04-05 14:03:00", "2000-01-01 12:13:21", "2020-04-05 14:03:00"],
        "str_date_col": ["2000-03-01", "2020-04-05", "2000-03-01", "2020-04-05"],
        "datetime_col_0": data_algebra.data_model.default_data_model().pd.to_datetime(
            data_algebra.data_model.default_data_model().pd.Series(["2010-01-01 12:13:21", "2030-04-05 14:03:00", "2010-01-01 12:13:21", "2030-04-05 14:03:00"]),
            format=datetime_format,
        ),
        "datetime_col_1": data_algebra.data_model.default_data_model().pd.to_datetime(
            data_algebra.data_model.default_data_model().pd.Series(["2010-01-01 12:11:21", "2030-04-06 14:03:00", "2010-01-01 12:11:21", "2030-04-06 14:03:00"]),
            format=datetime_format,
        ),
        "date_col_0": data_algebra.data_model.default_data_model().pd.to_datetime(
            data_algebra.data_model.default_data_model().pd.Series(["2000-01-02", "2035-04-05", "2000-01-02", "2035-04-05"]),
            format=date_format
        ).dt.date,
        "date_col_1": data_algebra.data_model.default_data_model().pd.to_datetime(
            data_algebra.data_model.default_data_model().pd.Series(["2000-01-02", "2035-05-05", "2000-01-02", "2035-05-05"]),
            format=date_format
        ).dt.date,
    })
    return d


def f(expression, *, d):
    return (
        descr(d=d)
            .extend({'new_column': expression})
            .select_columns(['row_id', 'new_column'])
            .order_rows(['row_id'])
    )


def fg(expression, *, d):
    return (
        descr(d=d)
            .extend(
                {'new_column': expression},
                partition_by=['g'])
            .select_columns(['g', 'row_id', 'new_column'])
            .order_rows(['g', 'row_id'])
    )


def fp(expression, *, d):
    return (
        descr(d=d)
            .project(
                {'new_column': expression},
                group_by=['g'])
            .order_rows(['g'])
    )


def fw(expression, *, d):
    return (
        descr(d=d)
            .extend(
                {'new_column': expression},
                partition_by=['g'],
                order_by=['row_id'])
            .select_columns(['g', 'row_id', 'new_column'])
            .order_rows(['g', 'row_id'])
    )


def test_method_catalog_issues_project_min():
    d = mk_example()
    ops = fp('x.min()', d=d)
    d.loc[:, ['g', 'x']]
    res_pandas = ops.transform(d)
    db_handle = data_algebra.SQLite.example_handle()
    db_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = db_handle.read_query(ops)
    db_handle.drop_table('d')
    db_handle.close()
    assert data_algebra.test_util.equivalent_frames(res_pandas, res_db)


def test_method_catalog_issues_project_median():
    d = mk_example()
    ops = fp('x.median()', d=d)
    d.loc[:, ['g', 'x']]
    res_pandas = ops.transform(d)
    db_handle = data_algebra.SQLite.example_handle()
    db_handle.insert_table(d, table_name='d', allow_overwrite=True)
    res_db = db_handle.read_query(ops)
    db_handle.drop_table('d')
    db_handle.close()
    assert data_algebra.test_util.equivalent_frames(res_pandas, res_db)
