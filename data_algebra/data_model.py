from abc import ABC

import numpy

import data_algebra.connected_components


def negate_or_subtract(*args):
    if len(args) == 1:
        return numpy.negative(args[0])
    if len(args) == 2:
        return numpy.subtract(args[0], args[1])
    raise ValueError("unexpected number of arguments in negate_or_subtract")


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
        'coalesce': lambda a, b: a.combine_first(b),  # assuming Pandas series
        'connected_components': lambda a, b: data_algebra.connected_components.connected_components(a, b),
        'co_equalizer': lambda a, b: data_algebra.connected_components.connected_components(a, b),
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
