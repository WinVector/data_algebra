
import data_algebra
import data_algebra.data_ops


# TODO: share some common fns such as as_int64 and coalesce_0 to all db_handles

# convert datetime to date
def as_int64(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.astype('int64'),  # x is a pandas Series
        args=[col],
        name='as_int64',
        sql_name='CAST',
        sql_suffix=' AS INT64'
    )


# trim string to date portion
def trimstr(col_name, *, start=0, stop):
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(col_name, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.str.slice(start=start, stop=stop),  # x is a pandas Series
        args=[col_name],
        name=f'trimstr_{start+1}_{stop}',
        sql_name='SUBSTR', sql_suffix=f', {start+1}, {stop}')


# replace missing with zeros
def coalesce_0(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.fillna(0),
        args=col,
        name='coalesce_0',
        sql_name='COALESCE',
        sql_suffix=', 0')


# convert datetime to date
def datetime_to_date(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.dt.date.copy(),  # x is a pandas Series
        args=col,
        name='datetime_to_date',
        sql_name='DATE')


# convert str to datetime
# https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions
def parse_datetime(col, *, format="%Y-%m-%d %H:%M:%S"):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        lambda x: data_algebra.default_data_model.pd.to_datetime(x, format=format),  # x is a pandas Series
        args=col,
        name='parse_datetime',
        sql_name='PARSE_DATETIME',  # https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
        sql_prefix='f"{format}", ')


# convert str to date
def parse_date(col, *, format="%Y-%m-%d"):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        lambda x: data_algebra.default_data_model.pd.to_datetime(x, format=format).dt.date.copy(),  # x is a pandas Series
        args=col,
        name='parse_date',
        sql_name='PARSE_DATE',  # https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
        sql_prefix=f'"{format}", ')


# convert date to dayofweek 1 through 7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofweek(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.dayofweek.html#pandas.Series.dt.dayofweek
        # https://stackoverflow.com/a/30222759
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.dayofweek + 1,  # x is a pandas Series
        args=col,
        name='dayofweek',
        sql_name='EXTRACT',
        sql_prefix='DAYOFWEEK FROM ')


# convert date to dayofweek 1 through 7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofyear(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.dayofyear,  # x is a pandas Series
        args=col,
        name='dayofyear',
        sql_name='EXTRACT',
        sql_prefix='DAYOFYEAR FROM ')


# convert date to dayofweek 1 through 7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofmonth(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.day,  # x is a pandas Series
        args=col,
        name='dayofmonth',
        sql_name='EXTRACT',
        sql_prefix='DAYOFMONTH FROM ')


# convert date to month
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def month(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.dayofweek + 1,  # x is a pandas Series
        args=col,
        name='month',
        sql_name='EXTRACT',
        sql_prefix='MONTH FROM ')


# convert date to quarter
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def quarter(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.quarter,  # x is a pandas Series
        args=col,
        name='quarter',
        sql_name='EXTRACT',
        sql_prefix='QUARTER FROM ')


# convert date to year
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def year(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.year,  # x is a pandas Series
        args=col,
        name='year',
        sql_name='EXTRACT',
        sql_prefix='YEAR FROM ')


# compute difference in timestamps in seconds
# https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#timestamp_diff
# TODO: implement and test
def timestamp_diff(col1, col2):
    assert isinstance(col1, str)
    assert isinstance(col2, str)
    return data_algebra.data_ops.user_fn(
        # https://stackoverflow.com/a/41340398
        lambda c1, c2: data_algebra.default_data_model.pd.Timedelta(c2 - c1).total_seconds(),
        args=[col1, col2],
        name='timestamp_diff',
        sql_name='TIMESTAMP_DIFF',
        sql_suffix=', HOUR')

# TODO: format date, format datetime, date diff (days)
# TODO: documentation page

fns = {
    'as_int64': as_int64,
    'trimstr': trimstr,
    'coalesce_0': coalesce_0,
    'datetime_to_date': datetime_to_date,
    'parse_datetime': parse_datetime,
    'parse_date': parse_date,
    'dayofweek': dayofweek,
    'dayofyear': dayofyear,
    'dayofmonth': dayofmonth,
    'month': month,
    'quarter': quarter,
    'year': year,
    'timestamp_diff': timestamp_diff,
}