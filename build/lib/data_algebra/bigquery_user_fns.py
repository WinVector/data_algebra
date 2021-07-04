
import datetime

import data_algebra
import data_algebra.data_ops


# TODO: share some common fns such as as_int64 and coalesce_0 to all db_handles

# TODO: re-eng all userfns to just be SQL constants, without pasting, perhaps pass in db handle


import data_algebra.user_fn


def as_int64(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: x.astype('int64'),  # x is a pandas Series
        sql_fn = lambda subs, db_model: f'CAST({subs[0]} AS INT64)',
        args=[col],
        display_form = f'as_int64({col})',
        name='as_int64',
    )


def as_str(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: x.astype('str'),  # x is a pandas Series
        sql_fn = lambda subs, db_model: f'CAST({subs[0]} AS STRING)',
        args=[col],
        display_form = f'as_int64({col})',
        name='as_str',
    )


# trim string to date portion
def trimstr(col_name, *, start=0, stop):
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(col_name, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: x.str.slice(start=start, stop=stop),  # x is a pandas Series
        sql_fn = lambda subs, db_model: f'SUBSTR({subs[0]}, {start+1}, {stop})',
        args=[col_name],
        display_form = f'trimstr({col_name}, start={start}, stop={stop})',
        name='trimstr',
    )


# replace missing with zeros
def coalesce_0(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: x.fillna(0),   # x is a pandas Series
        sql_fn = lambda subs, db_model: f'COALESCE({subs[0]}, 0)',
        args=[col],
        display_form = f'coalesce_0({col})',
        name='coalesce_0',
    )


# compute difference in dates in days
def coalesce(cols):
    if isinstance(cols, str):
        cols = [cols]
    assert len(cols) > 1
    assert all([isinstance(ci, str) for ci in cols])
    assert len(set(cols)) == len(cols)
    cols = cols.copy()

    def f(*args):
        res = args[0].copy()
        for i in range(1, len(args)):
            res = res.combine_first(args[i])
        return res

    return data_algebra.user_fn.FnTerm(
        pandas_fn = f,
        sql_fn = lambda subs, db_model: f'COALESCE({", ".join(subs)})', # TODO: check SQL
        args=cols,
        display_form = f'coalesce({cols})',
        name='coalesce',
    )


# convert datetime to date
def datetime_to_date(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: x.dt.date.copy(),  # x is a pandas Series
        sql_fn = lambda subs, db_model: f'DATE({subs[0]})',
        args=[col],
        display_form = f'datetime_to_date({col})',
        name='datetime_to_date',
    )


# convert str to datetime
# https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions
def parse_datetime(col, *, format="%Y-%m-%d %H:%M:%S"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return data_algebra.user_fn.FnTerm(
        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        pandas_fn = lambda x: data_algebra.default_data_model.pd.to_datetime(x, format=format),  # x is a pandas Series
        sql_fn = lambda subs, db_model: f'PARSE_DATETIME({db_model.quote_string(format)}, {subs[0]})',
        args=[col],
        display_form = f'parse_datetime({col}, format="{format}")',
        name='parse_datetime',
    )


# convert str to date
def parse_date(col, *, format="%Y-%m-%d"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return data_algebra.user_fn.FnTerm(
        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x, format=format).dt.date.copy(),  # x is a pandas Series
        sql_fn=lambda subs, db_model: f'PARSE_DATE({db_model.quote_string(format)}, {subs[0]})',
        args=[col],
        display_form=f'parse_date({col}, format="{format}")',
        name='parse_date',
    )


# convert datetime to str
# https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions
def format_datetime(col, *, format="%Y-%m-%d %H:%M:%S"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: x.dt.strftime(date_format=format),
        sql_fn=lambda subs, db_model: f'FORMAT_DATETIME({db_model.quote_string(format)}, {subs[0]})',
        args=[col],
        display_form=f'format_datetime({col}, format="{format}")',
        name='format_datetime',
    )


# convert date to str
def format_date(col, *, format="%Y-%m-%d"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.strftime(date_format=format),
        sql_fn=lambda subs, db_model: f'FORMAT_DATE({db_model.quote_string(format)}, {subs[0]})',
        args=[col],
        display_form=f'format_date({col}, format="{format}")',
        name='format_date',
    )


# convert date to dayofweek Sunday=1 through Saturday=7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofweek(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.dayofweek.html#pandas.Series.dt.dayofweek
        # https://stackoverflow.com/a/30222759
        # x is a pandas Series
        pandas_fn=lambda x: 1 + ((data_algebra.default_data_model.pd.to_datetime(x).dt.dayofweek.astype('int64') + 1) % 7),
        sql_fn=lambda subs, db_model: f'EXTRACT(DAYOFWEEK FROM {subs[0]})',
        args=[col],
        display_form=f'dayofweek({col})',
        name='dayofweek',
    )


# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofyear(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.dayofyear.astype('int64'),
        sql_fn=lambda subs, db_model: f'EXTRACT(DAYOFYEAR FROM {subs[0]})',
        args=[col],
        display_form=f'dayofyear({col})',
        name='dayofyear',
    )


# convert date to week of year
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def weekofyear(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.isocalendar().week.astype('int64'),
        sql_fn=lambda subs, db_model: f'EXTRACT(WEEK FROM {subs[0]})',
        args=[col],
        display_form=f'weekofyear({col})',
        name='weekofyear',
    )


# convert date to dayofweek 1 through 7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofmonth(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.day.astype('int64'),
        sql_fn=lambda subs, db_model: f'EXTRACT(DAY FROM {subs[0]})',
        args=[col],
        display_form=f'dayofmonth({col})',
        name='dayofmonth',
    )


# convert date to month
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def month(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.month.astype('int64'),
        sql_fn=lambda subs, db_model: f'EXTRACT(MONTH FROM {subs[0]})',
        args=[col],
        display_form=f'month({col})',
        name='month',
    )


# convert date to quarter
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def quarter(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series
        pandas_fn=lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.quarter.astype('int64'),
        sql_fn=lambda subs, db_model: f'EXTRACT(QUARTER FROM {subs[0]})',
        args=[col],
        display_form=f'quarter({col})',
        name='quarter',
    )


# convert date to year
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def year(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        pandas_fn = lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.year.astype('int64'),
        sql_fn = lambda subs, db_model: f'EXTRACT(YEAR FROM {subs[0]})',
        args=[col],
        display_form = f'year({col})',
        name='year',
    )


# compute difference in timestamps in seconds
# https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#timestamp_diff
def timestamp_diff(col1, col2):
    assert isinstance(col1, str)
    assert isinstance(col2, str)
    return data_algebra.user_fn.FnTerm(
        # https://stackoverflow.com/a/41340398
        # looks like Timedelta is scalar
        # TODO: find vectorized form
        pandas_fn = lambda c1, c2: [
            data_algebra.default_data_model.pd.Timedelta(c1[i] - c2[i]).total_seconds() for i in range(len(c1))],
        sql_fn = lambda subs, db_model: f'TIMESTAMP_DIFF({subs[0]}, {subs[1]}, SECOND)',
        args=[col1, col2],
        display_form = f'timestamp_diff({col1}, {col2})',
        name='timestamp_diff',
    )


# compute difference in dates in days
def date_diff(col1, col2):
    assert isinstance(col1, str)
    assert isinstance(col2, str)
    return data_algebra.user_fn.FnTerm(
        # https://stackoverflow.com/a/41340398
        # looks like Timedelta is scalar
        # TODO: find vectorized form
        pandas_fn=lambda c1, c2: [
            data_algebra.default_data_model.pd.Timedelta(c1[i] - c2[i]).days for i in range(len(c1))],
        sql_fn = lambda subs, db_model: f'TIMESTAMP_DIFF({subs[0]}, {subs[1]}, DAY)',
        args=[col1, col2],
        display_form = f'date_diff({col1}, {col2})',
        name='date_diff',
    )


# find the nearest Sunday at or before this date
def base_Sunday(col):
    assert isinstance(col, str)
    return data_algebra.user_fn.FnTerm(
        # x is a pandas Series of datetime.date
        # TODO: vectorize
        pandas_fn=lambda x: [x[i] - datetime.timedelta(days=(x[i].weekday() + 1) % 7) for i in range(len(x))],
        sql_fn = lambda subs, db_model: f'DATE_SUB({subs[0]}, INTERVAL (EXTRACT(DAYOFWEEK FROM {subs[0]}) - 1) DAY)',
        args=[col],
        display_form = f'base_Sunday({col})',
        name='base_Sunday',
    )


# TODO: documentation page


fns = {
    'as_int64': as_int64,
    'as_str': as_str,
    'trimstr': trimstr,
    'coalesce_0': coalesce_0,
    'coalesce': coalesce,
    'datetime_to_date': datetime_to_date,
    'parse_datetime': parse_datetime,
    'parse_date': parse_date,
    'format_datetime': format_datetime,
    'format_date': format_date,
    'dayofweek': dayofweek,
    'dayofyear': dayofyear,
    'dayofmonth': dayofmonth,
    'weekofyear': weekofyear,
    'month': month,
    'quarter': quarter,
    'year': year,
    'timestamp_diff': timestamp_diff,
    'date_diff': date_diff,
    'base_Sunday': base_Sunday,
}
