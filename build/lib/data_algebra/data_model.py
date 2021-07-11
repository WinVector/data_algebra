from abc import ABC
import datetime

import numpy

import data_algebra.connected_components


def negate_or_subtract(*args):
    if len(args) == 1:
        return numpy.negative(args[0])
    if len(args) == 2:
        return numpy.subtract(args[0], args[1])
    raise ValueError("unexpected number of arguments in negate_or_subtract")


def _calc_date_diff(x0, x1):
    # x is a pandas Series or list of datetime.date compatible types
    x0 = data_algebra.default_data_model.pd.Series(x0)
    x1 = data_algebra.default_data_model.pd.Series(x1)
    x0_dates = data_algebra.default_data_model.pd.to_datetime(x0).dt.date.copy()
    x1_dates = data_algebra.default_data_model.pd.to_datetime(x1).dt.date.copy()
    deltas = [(x0_dates[i] - x1_dates[i]).days for i in range(len(x0_dates))]
    return deltas


def _calc_base_Sunday(x):
    # x is a pandas Series or list of datetime.date compatible types
    x = data_algebra.default_data_model.pd.Series(x)
    x_dates = data_algebra.default_data_model.pd.to_datetime(x).dt.date.copy()
    res = [xi - datetime.timedelta(days=(xi.weekday() + 1) % 7) for xi in x_dates]
    return res


def _calc_week_of_Year(x):
    # x is a pandas Series or list of datetime.date compatible types
    # TODO: better impl
    # Note was getting inconsistent results on vectorized methods
    x = data_algebra.default_data_model.pd.Series(x)
    x_dates = data_algebra.default_data_model.pd.to_datetime(x).dt.date.copy()
    cur_dates = [datetime.date(dti.year, dti.month, dti.day) for dti in x_dates]
    base_dates = [datetime.date(dti.year, 1, 1) for dti in x_dates]
    base_dates = _calc_base_Sunday(base_dates)
    deltas = [(cur_dates[i] - base_dates[i]).days for i in range(len(cur_dates))]
    res = [di//7 for di in deltas]
    res = numpy.maximum(res, 1)
    return res


def _coalesce(a, b):
    a_is_series = isinstance(a, data_algebra.default_data_model.pd.Series)
    b_is_series = isinstance(b, data_algebra.default_data_model.pd.Series)
    if (not a_is_series) and (not b_is_series):
        raise ValueError("at least one argument must be a Pandas series")
    if not a_is_series:
        a = data_algebra.default_data_model.pd.Series(numpy.array([a] * len(b)))
    if not b_is_series:
        b = data_algebra.default_data_model.pd.Series(numpy.array([b] * len(a)))
    res = a.combine_first(b)
    return res


def populate_impl_map(data_model):
    impl_map = {
        '==': numpy.equal,
        '=': numpy.equal,
        '!=': numpy.not_equal,
        '<>': numpy.not_equal,
        '<': numpy.less,
        '<=': numpy.less_equal,
        '>': numpy.greater,
        '>=': numpy.greater_equal,
        '+': numpy.add,
        '-': negate_or_subtract,
        'neg': numpy.negative,
        '*': numpy.multiply,
        '/': numpy.divide,
        '//': numpy.floor_divide,
        '%': numpy.mod,
        '**': numpy.power,
        'and': numpy.logical_and,
        '&': numpy.logical_and,
        '&&': numpy.logical_and,
        'or': numpy.logical_or,
        '|': numpy.logical_or,
        '||': numpy.logical_or,
        'xor': numpy.logical_xor,
        '^': numpy.logical_xor,
        'not': numpy.logical_not,
        '!': numpy.logical_not,
        'if_else': numpy.where,
        'is_null': data_model.isnull,
        'is_bad': data_model.bad_column_positions,
        'is_in': numpy.isin,
        'concat': lambda a, b: numpy.char.add(numpy.asarray(a, dtype=numpy.str),
                                              numpy.asarray(b, dtype=numpy.str)),
        'coalesce': lambda a, b: _coalesce(a, b),  # assuming Pandas series
        'connected_components': lambda a, b: data_algebra.connected_components.connected_components(a, b),
        'co_equalizer': lambda a, b: data_algebra.connected_components.connected_components(a, b),
        # fns that had been in bigquery_user_fns
        # x is a pandas Series
        'as_int64': lambda x: x.astype('int64').copy(),
        'as_str': lambda x: x.astype('str').copy(),
        'trimstr': lambda x, start, stop: x.str.slice(start=start, stop=stop),
        'datetime_to_date': lambda x: x.dt.date.copy(),
        'parse_datetime': lambda x, format: data_algebra.default_data_model.pd.to_datetime(x, format=format),
        'parse_date': lambda x, format: data_algebra.default_data_model.pd.to_datetime(x, format=format).dt.date.copy(),
        'format_datetime': lambda x, format: x.dt.strftime(date_format=format),
        'format_date': lambda x, format: data_algebra.default_data_model.pd.to_datetime(x).dt.strftime(date_format=format),
        'dayofweek': lambda x: 1 + ((data_algebra.default_data_model.pd.to_datetime(x).dt.dayofweek.astype('int64') + 1) % 7),
        'dayofyear': lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.dayofyear.astype('int64').copy(),
        'weekofyear': _calc_week_of_Year,
        'dayofmonth': lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.day.astype('int64').copy(),
        'month': lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.month.astype('int64').copy(),
        'quarter': lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.quarter.astype('int64').copy(),
        'year': lambda x: data_algebra.default_data_model.pd.to_datetime(x).dt.year.astype('int64').copy(),
        'timestamp_diff': lambda c1, c2: [
            data_algebra.default_data_model.pd.Timedelta(c1[i] - c2[i]).total_seconds() for i in range(len(c1))],
        'date_diff': _calc_date_diff,
        'base_Sunday': _calc_base_Sunday,
    }
    return impl_map


class DataModel(ABC):
    def __init__(self, presentation_model_name):
        self.presentation_model_name = presentation_model_name
        self.impl_map = populate_impl_map(data_model=self)

    # helper functions

    def data_frame(self, arg=None):
        raise NotImplementedError("base method called")

    def is_appropriate_data_instance(self, df):
        raise NotImplementedError("base method called")

    def can_convert_col_to_numeric(self, x):
        """check if non-empty vector can convert to numeric"""
        raise NotImplementedError("base method called")

    def to_numeric(self, x, *, errors="coerce"):
        raise NotImplementedError("base method called")

    def isnull(self, x):
        raise NotImplementedError("base method called")

    def bad_column_positions(self, x):
        """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
        raise NotImplementedError("base method called")

    # operation implementations

    def table_step(self, op, *, data_map, narrow):
        """
        Represents a data input.

        :param op:
        :param data_map:
        :param narrow:
        :return:
        """
        raise NotImplementedError("base method called")

    def extend_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        raise NotImplementedError("base method called")

    def project_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def select_rows_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def select_columns_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def drop_columns_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def order_rows_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def rename_columns_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def natural_join_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def concat_rows_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base method called")

    def convert_records_step(self, op, *, data_map, narrow):
        raise NotImplementedError("base class called")
