
import datetime

import numpy

import data_algebra
import data_algebra.data_ops


# Legacy to be removed


def as_int64(col):
    assert isinstance(col, str)
    return f'{col}.as_int64()'


def as_str(col):
    assert isinstance(col, str)
    return f'{col}.as_str()'


# trim string to date portion
def trimstr(col_name, *, start=0, stop):
    assert isinstance(start, int)
    assert isinstance(stop, int)
    assert isinstance(col_name, str)
    return f'{col_name}.trimstr({start}, {stop})'


# replace missing with zeros
def coalesce_0(col):
    assert isinstance(col, str)
    return f'{col}.coalesce(0)'


def coalesce(cols):
    if isinstance(cols, str):
        cols = [cols]
    assert len(cols) > 1
    return ' %?% '.join(cols)


# convert datetime to date
def datetime_to_date(col):
    assert isinstance(col, str)
    return f'{col}.datetime_to_date()'


# convert str to datetime
# https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions
def parse_datetime(col, *, format="%Y-%m-%d %H:%M:%S"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return f'{col}.parse_datetime("{format}")'


# convert str to date
def parse_date(col, *, format="%Y-%m-%d"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return f'{col}.parse_date("{format}")'


# convert datetime to str
# https://cloud.google.com/bigquery/docs/reference/standard-sql/datetime_functions
def format_datetime(col, *, format="%Y-%m-%d %H:%M:%S"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return f'{col}.format_datetime("{format}")'


# convert date to str
def format_date(col, *, format="%Y-%m-%d"):
    assert isinstance(col, str)
    assert isinstance(format, str)
    return f'{col}.format_date("{format}")'


# convert date to dayofweek Sunday=1 through Saturday=7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofweek(col):
    assert isinstance(col, str)
    return f'{col}.dayofweek()'


# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofyear(col):
    assert isinstance(col, str)
    return f'{col}.dayofyear()'


# convert date to week of year
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def weekofyear(col):
    assert isinstance(col, str)
    return f'{col}.weekofyear()'


# convert date to dayofweek 1 through 7
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def dayofmonth(col):
    assert isinstance(col, str)
    return f'{col}.dayofmonth()'


# convert date to month
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def month(col):
    assert isinstance(col, str)
    return f'{col}.month()'


# convert date to quarter
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def quarter(col):
    assert isinstance(col, str)
    return f'{col}.quarter()'


# convert date to year
# https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
def year(col):
    assert isinstance(col, str)
    return f'{col}.year()'


# compute difference in timestamps in seconds
# https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#timestamp_diff
def timestamp_diff(col1, col2):
    assert isinstance(col1, str)
    assert isinstance(col2, str)
    return f'{col1}.timestamp_diff({col2})'


# compute difference in dates in days
def date_diff(col1, col2):
    assert isinstance(col1, str)
    assert isinstance(col2, str)
    return f'{col1}.date_diff({col2})'


# find the nearest Sunday at or before this date
def base_Sunday(col):
    assert isinstance(col, str)
    return f'{col}.base_Sunday()'


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
