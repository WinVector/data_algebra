"""
Base class for adapters for Pandas-like APIs
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional
import datetime
import types
import numbers
import warnings

import numpy

import data_algebra.util
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components


# TODO: possibly import dask, Nvidia Rapids, modin, datatable versions


def none_mark_scalar_or_length(v) -> Optional[int]:
    """
    Test if item is a scalar (returning None) if it is, else length of object.

    :param v: value to test
    :return: None if value is a scalar, else length.
    """
    # get some of the obvious types, and str (as str doesn't throw on len)
    if isinstance(v, (type(None), str, int, float)):
        return None  # obvious scalar
    # len() throws on scalars other than str
    try:
        return len(v)
    except TypeError:
        return None  # len() failed, probably a scalar

        
def promote_scalar_to_array(vi, *, target_len: int) -> List:
    """
    Convert a scalar into a vector. Pass a non-trivial array through.

    :param vi: value to promote to scalar
    :target_len: length for vector
    :return: list
    """
    assert isinstance(target_len, int)
    assert target_len >= 0
    if target_len <= 0:
        return []
    len_v = none_mark_scalar_or_length(vi)
    # noinspection PyBroadException
    if len_v is None:
        return [vi] * target_len  # scalar
    assert len_v == target_len
    return vi


def _negate_or_subtract(*args):
    if len(args) == 1:
        return numpy.negative(args[0])
    if len(args) == 2:
        return numpy.subtract(args[0], args[1])
    raise ValueError("unexpected number of arguments in _negate_or_subtract")


def _type_safe_equal(a, b):
    # Could try numpy.logical_and(numpy.greater_equal(a, b), numpy.less_equal(a, b)).
    # However, None >= None throws
    type_a = data_algebra.util.guess_carried_scalar_type(a)
    type_b = data_algebra.util.guess_carried_scalar_type(b)
    if not data_algebra.util.compatible_types([type_a, type_b]):
        raise TypeError(f"can't compare {type_a} to {type_b}")
    return numpy.equal(a, b)


def _type_safe_not_equal(a, b):
    # Could try numpy.logical_or(numpy.greater(a, b), numpy.less(a, b)).
    # However, None > None throws
    type_a = data_algebra.util.guess_carried_scalar_type(a)
    type_b = data_algebra.util.guess_carried_scalar_type(b)
    if not data_algebra.util.compatible_types([type_a, type_b]):
        raise TypeError(f"can't compare {type_a} to {type_b}")
    return numpy.not_equal(a, b)


def _type_safe_is_in(a, b):
    b = list(b)
    if len(b) > 0:
        type_a = data_algebra.util.guess_carried_scalar_type(a)
        type_b = {data_algebra.util.map_type_to_canonical(type(v)) for v in b}
        if not data_algebra.util.compatible_types(type_b):
            raise TypeError(f"multiple types in set: {type_b}")
        type_b = list(type_b)[0]
        if not data_algebra.util.compatible_types([type_a, type_b]):
            raise TypeError(f"can't check for an {type_a} in a set of {type_b}'s")
    return numpy.isin(a, b)


def _k_and(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = numpy.logical_and(res, args[i])
    return res


def _k_or(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = numpy.logical_or(res, args[i])
    return res


def _k_add(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = numpy.add(res, args[i])
    return res


def _k_mul(*args):
    res = args[0]
    for i in range(1, len(args)):
        res = numpy.multiply(res, args[i])
    return res


def _where_expr(*args):
    """
    where(cond, a, b) returns a for positions where cond is True, b otherwise (including cond None).
    """
    assert len(args) == 3
    cond = args[0]
    a = args[1]
    b = args[2]
    return numpy.where(cond, a, b)


# base class for Pandas-like API realization
class PandasModelBase(data_algebra.data_model.DataModel, ABC):
    """
    Base class for implementing the data algebra on pandas-like APIs
    """

    pd: types.ModuleType
    impl_map: Dict[str, Callable]
    transform_op_map: Dict[str, str]
    user_fun_map: Dict[str, Callable]
    _method_dispatch_table: Dict[str, Callable]

    def __init__(self, *, pd: types.ModuleType, presentation_model_name: str):
        assert isinstance(pd, types.ModuleType)
        data_algebra.data_model.DataModel.__init__(
            self, presentation_model_name=presentation_model_name
        )
        self.pd = pd
        self.impl_map = self._populate_impl_map()
        self.transform_op_map = {"any_value": "first"}
        self.user_fun_map = dict()
        self._method_dispatch_table = {
            "ConcatRowsNode": self._concat_rows_step,
            "ConvertRecordsNode": self._convert_records_step,
            "DropColumnsNode": self._drop_columns_step,
            "ExtendNode": self._extend_step,
            "NaturalJoinNode": self._natural_join_step,
            "OrderRowsNode": self._order_rows_step,
            "ProjectNode": self._project_step,
            "MapColumnsNode": self._map_columns_step,
            "RenameColumnsNode": self._rename_columns_step,
            "SelectColumnsNode": self._select_columns_step,
            "SelectRowsNode": self._select_rows_step,
            "SQLNode": self._sql_proxy_step,
            "TableDescription": self._table_step,
        }

    # implementations
    
    def _calc_date_diff(self, x0, x1):
        # x is a pandas Series or list of datetime.date compatible types
        x0 = self.pd.Series(x0)
        x1 = self.pd.Series(x1)
        x0_dates = self.pd.to_datetime(x0).dt.date.copy()
        x1_dates = self.pd.to_datetime(x1).dt.date.copy()
        deltas = [(x0_dates[i] - x1_dates[i]).days for i in range(len(x0_dates))]
        return deltas

    def _calc_base_Sunday(self, x):
        # x is a pandas Series or list of datetime.date compatible types
        x = self.pd.Series(x)
        x_dates = self.pd.to_datetime(x).dt.date.copy()
        res = [xi - datetime.timedelta(days=(xi.weekday() + 1) % 7) for xi in x_dates]
        return res

    def _calc_week_of_Year(self, x):
        # x is a pandas Series or list of datetime.date compatible types
        # TODO: better impl
        # Note was getting inconsistent results on vectorized methods
        x = self.pd.Series(x)
        x_dates = self.pd.to_datetime(x).dt.date.copy()
        cur_dates = [datetime.date(dti.year, dti.month, dti.day) for dti in x_dates]
        base_dates = [datetime.date(dti.year, 1, 1) for dti in x_dates]
        base_dates = self._calc_base_Sunday(base_dates)
        deltas = [(cur_dates[i] - base_dates[i]).days for i in range(len(cur_dates))]
        res = [di // 7 for di in deltas]
        res = numpy.maximum(res, 1)
        return res

    def _coalesce(self, a, b):
        a_is_series = isinstance(a, self.pd.Series)
        b_is_series = isinstance(b, self.pd.Series)
        if (not a_is_series) and (not b_is_series):
            raise ValueError("at least one argument must be a Pandas series")
        if not a_is_series:
            a = self.pd.Series(numpy.array([a] * len(b)))
        if not b_is_series:
            b = self.pd.Series(numpy.array([b] * len(a)))
        res = a.combine_first(b)
        return res

    def _map_v(self, a, value_map, default_value=None):
        """Map values to values."""
        if len(value_map) > 0:
            type_a = data_algebra.util.guess_carried_scalar_type(a)
            type_k = {
                data_algebra.util.map_type_to_canonical(type(v)) for v in value_map.keys()
            }
            if not data_algebra.util.compatible_types(type_k):
                raise TypeError(f"multiple types in dictionary keys: {type_k} in mapv()")
            type_k = list(type_k)[0]
            if not data_algebra.util.compatible_types([type_a, type_k]):
                raise TypeError(f"can't map {type_a} from a dict of {type_k}'s in mapv()")
            type_v = {
                data_algebra.util.map_type_to_canonical(type(v)) for v in value_map.values()
            }
            if not data_algebra.util.compatible_types(type_v):
                raise TypeError(f"multiple types in dictionary values: {type_v} in mapv()")
            type_v = list(type_v)[0]
            if default_value is not None:
                type_d = data_algebra.util.guess_carried_scalar_type(default_value)
                if not data_algebra.util.compatible_types([type_d, type_v]):
                    raise TypeError(f"default is {type_d} for {type_v} values in mapv()'s")
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html
        a = self.pd.Series(a)
        a = a.map(value_map, na_action="ignore")
        a[self.bad_column_positions(a)] = default_value
        return numpy.array(a.values)

    def _if_else_expr(self, *args):
        """
        if_else(cond, a, b) returns a for positions where cond is True, b when cond is False, None otherwise.
        """
        assert len(args) == 3
        cond = args[0]
        a = args[1]
        b = args[2]
        res = numpy.where(cond, a, b)
        bad_posns = self.bad_column_positions(cond)
        if numpy.any(bad_posns):
            res[bad_posns] = None
        return res

    def _populate_impl_map(self) -> Dict[str, Callable]:
        """
        Map symbols to implementations.
        """
        impl_map = {
            "==": _type_safe_equal,
            "=": _type_safe_equal,
            "!=": _type_safe_not_equal,
            "<>": _type_safe_not_equal,
            "<": numpy.less,  # already checks types
            "<=": numpy.less_equal,  # already checks types
            ">": numpy.greater,  # already checks types
            ">=": numpy.greater_equal,  # already checks types
            "+": _k_add,
            "-": _negate_or_subtract,
            "neg": numpy.negative,
            "*": _k_mul,
            "/": numpy.divide,
            "//": numpy.floor_divide,
            "%/%": numpy.divide,
            "%": numpy.mod,
            "**": numpy.power,
            "and": _k_and,
            "&": numpy.bitwise_and,
            "or": _k_or,
            "|": _k_or,
            "xor": numpy.logical_xor,
            "^": numpy.logical_xor,
            "not": numpy.bitwise_or,
            "where": _where_expr,
            "if_else": lambda *args: self._if_else_expr(*args),
            "is_nan": self.isnan,
            "is_inf": self.isinf,
            "is_null": self.isnull,
            "is_bad": self.bad_column_positions,
            "is_in": _type_safe_is_in,
            "concat": lambda a, b: numpy.char.add(
                numpy.asarray(a, dtype=str), numpy.asarray(b, dtype=str)
            ),
            "coalesce": lambda a, b: self._coalesce(a, b),  # assuming Pandas series
            "connected_components": lambda a, b: data_algebra.connected_components.connected_components(
                a, b
            ),
            "co_equalizer": lambda a, b: data_algebra.connected_components.connected_components(
                a, b
            ),
            "mapv": lambda a, value_map, default_value=None: self._map_v(a, value_map, default_value),
            # additonal fns
            # x is a pandas Series
            "as_int64": lambda x: x.astype("int64").copy(),
            "as_str": lambda x: x.astype("str").copy(),
            "trimstr": lambda x, start, stop: x.str.slice(start=start, stop=stop),
            "datetime_to_date": lambda x: x.dt.date.copy(),
            "parse_datetime": lambda x, format: self.pd.to_datetime(
                x, format=format
            ),
            "parse_date": lambda x, format: self.pd.to_datetime(
                x, format=format
            ).dt.date.copy(),
            "format_datetime": lambda x, format: x.dt.strftime(date_format=format),
            "format_date": lambda x, format: self.pd.to_datetime(
                x
            ).dt.strftime(date_format=format),
            "dayofweek": lambda x: 1
            + (
                (
                    self.pd.to_datetime(x).dt.dayofweek.astype(
                        "int64"
                    )
                    + 1
                )
                % 7
            ),
            "dayofyear": lambda x: self.pd.to_datetime(x)
            .dt.dayofyear.astype("int64")
            .copy(),
            "weekofyear": lambda x: self._calc_week_of_Year(x),
            "dayofmonth": lambda x: self.pd.to_datetime(x)
            .dt.day.astype("int64")
            .copy(),
            "month": lambda x: self.pd.to_datetime(x)
            .dt.month.astype("int64")
            .copy(),
            "quarter": lambda x: self.pd.to_datetime(x)
            .dt.quarter.astype("int64")
            .copy(),
            "year": lambda x: self.pd.to_datetime(x)
            .dt.year.astype("int64")
            .copy(),
            "timestamp_diff": lambda c1, c2: [
                self.pd.Timedelta(c1[i] - c2[i]).total_seconds()
                for i in range(len(c1))
            ],
            "date_diff": lambda x0, x1: self._calc_date_diff(x0, x1),
            "base_Sunday": lambda x: self._calc_base_Sunday(x),
        }
        return impl_map

    # utils

    def data_frame(self, arg=None):
        """
        Build a new emtpy data frame.

        :param arg" optional argument passed to constructor.
        :return: data frame
        """
        if arg is None:
            # noinspection PyUnresolvedReferences
            return self.pd.DataFrame()
        # noinspection PyUnresolvedReferences
        return self.pd.DataFrame(arg)

    def is_appropriate_data_instance(self, df):
        """
        Check if df is our type of data frame.
        """
        # noinspection PyUnresolvedReferences
        return isinstance(df, self.pd.DataFrame)

    def can_convert_col_to_numeric(self, x):
        """
        Return True if column or value can be converted to numeric type.
        """
        if isinstance(x, numbers.Number):
            return True
        # noinspection PyUnresolvedReferences
        return self.pd.api.types.is_numeric_dtype(x)

    def to_numeric(self, x, *, errors="coerce"):
        """
        Convert column to numeric.
        """
        # noinspection PyUnresolvedReferences
        return self.pd.to_numeric(x, errors="coerce")

    def isnull(self, x):
        """
        Return vector indicating which entries are null (vectorized).
        """
        return self.pd.isnull(x)

    def isnan(self, x):
        """
        Return vector indicating which entries are nan (vectorized).
        """
        x = numpy.asarray(x + 0.0, dtype=float)
        return numpy.isnan(x)

    def isinf(self, x):
        """
        Return vector indicating which entries are nan (vectorized).
        """
        x = numpy.asarray(x + 0.0, dtype=float)
        return numpy.isinf(x)

    def bad_column_positions(self, x):
        """
        Return vector indicating which entries are bad (null or nan) (vectorized).
        """
        if self.can_convert_col_to_numeric(x):
            x = numpy.asarray(x + 0.0, dtype=float)
            return numpy.logical_or(
                self.pd.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
            )
        return self.pd.isnull(x)

    # bigger stuff

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _table_step(self, op, *, data_map: dict):
        """
        Return a copy of data frame from table description and data_map.
        """
        if op.node_name != "TableDescription":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        if (data_map is not None) and (len(data_map) > 0):
            df = data_map[op.table_name]
            if not self.is_appropriate_data_instance(df):
                raise ValueError(
                    "data_map[" + op.table_name + "] was not the right type"
                )
        else:
            df = op.head
            assert df is not None
            if not self.is_appropriate_data_instance(df):
                raise ValueError(
                    "Unnamed TableDescription stored was not the right type"
                )
        # check all columns we expect are present
        columns_using = op.column_names
        missing = set(columns_using) - set(df.columns)
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        res = df.loc[:, columns_using]
        res = res.reset_index(drop=True, inplace=False)  # copy and clear out indices
        return res

    def _sql_proxy_step(self, op, *, data_map: dict):
        """
        execute SQL
        """
        assert op.node_name == "SQLNode"
        db_handle = data_map[op.view_name]
        res = db_handle.read_query("\n".join(op.sql))
        return res

    def columns_to_frame_(self, cols: Dict[str, Any], *, target_rows: Optional[int] = None):
        """
        Convert a dictionary of column names to series-like objects and scalars into a Pandas data frame.
        Deal with special cases, such as some columns coming in as scalars (often from Panda aggregation).

        :param cols: dictionary mapping column names to columns
        :param target_rows: number of rows we are shooting for
        :return: Pandas data frame.
        """
        # noinspection PyUnresolvedReferences
        assert isinstance(cols, dict)
        assert isinstance(target_rows, (int, type(None)))
        if target_rows is not None:
            assert target_rows >= 0
        if len(cols) < 1:
            # all scalars, so nothing carrying index information
            if target_rows is not None:
                return self.pd.DataFrame({}, index=range(target_rows)).reset_index(drop=True, inplace=False)
            else:
                return self.pd.DataFrame({})
        was_all_scalars = True
        for v in cols.values():
            ln = none_mark_scalar_or_length(v)
            if ln is not None:
                was_all_scalars = False
                if target_rows is None:
                    target_rows = ln
                else:
                    assert target_rows == ln
        if was_all_scalars:
            if target_rows is None:
                target_rows = 1
            # all scalars, so nothing carrying index information
            promoted_cols = {k: promote_scalar_to_array(v, target_len=target_rows) for (k, v) in cols.items()}
            return self.pd.DataFrame(promoted_cols, index=range(target_rows)).reset_index(drop=True, inplace=False)
        assert target_rows is not None
        if target_rows < 1:
            # no rows, so presuming no index information (shouldn't have come from an aggregation)
            return self.pd.DataFrame({k: [] for k in cols.keys()})
        # agg can return scalars, which then can't be made into a self.pd.DataFrame
        promoted_cols = {k: promote_scalar_to_array(v, target_len=target_rows) for (k, v) in cols.items()}
        return self.pd.DataFrame(promoted_cols)

    def add_data_frame_columns_to_data_frame_(self, res, transient_new_frame):
        """
        Add columns from transient_new_frame to res. Res may be altered, and either of res or
        transient_new_frame may be returned.
        """
        if transient_new_frame.shape[1] < 1:
            return res
        if (res.shape[0] == 0) and (transient_new_frame.shape[0] > 0):
            # scalars get interpreted as single row items, instead of zero row items
            # growing the extension frame
            transient_new_frame = transient_new_frame.iloc[range(0), :].reset_index(
                drop=True
            )
        if res.shape[0] == transient_new_frame.shape[0]:
            if res.shape[1] < 1:
                return transient_new_frame
            if transient_new_frame.shape[1] < 1:
                return res
        if (2 * transient_new_frame.shape[1]) > res.shape[1]:
            # lots of columns path
            # https://win-vector.com/2021/08/03/i-think-pandas-may-have-lost-the-plot/
            for c in set(res.columns).intersection(set(transient_new_frame.columns)):
                del res[c]
            return self.pd.concat([res, transient_new_frame], axis=1)
        # normal path
        for c in transient_new_frame.columns:
            res[c] = transient_new_frame[c]
        return res

    def eval(self, op, *, data_map: Dict[str, Any]):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames or data sources
        :return: data frame result
        """
        assert isinstance(data_map, Dict)
        assert isinstance(op, data_algebra.data_ops_types.OperatorPlatform)
        return self._eval_value_source(s=op, data_map=data_map)

    def _eval_value_source(self, s, *, data_map: dict):
        """
        Evaluate an incoming (or value source) node.
        """
        return self._method_dispatch_table[s.node_name](
            op=s, data_map=data_map
        )

    def _extend_step(self, op, *, data_map):
        """
        Execute an extend step, returning a data frame.
        """
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        if res.shape[0] <= 0:
            # special case out no-row frame
            incoming_col_set = set(res.columns)
            v_dict = {k: [] for k in res.columns}
            for k in op.ops.keys():
                if k not in incoming_col_set:
                    v_dict[k] = []
            return self.pd.DataFrame(v_dict)
        window_situation = (
            op.windowed_situation
            or (len(op.partition_by) > 0)
            or (len(op.order_by) > 0)
        )
        if window_situation:
            op.check_extend_window_fns_()
        if not window_situation:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # out of range things like arccosh were warning
                new_cols = {k: opk.act_on(res, data_model=self) for k, opk in op.ops.items()}
            new_frame = self.columns_to_frame_(new_cols, target_rows=res.shape[0])
            res = self.add_data_frame_columns_to_data_frame_(res, new_frame)
        else:
            data_algebra_temp_cols = {}
            standin_name = "_data_algebra_temp_g"  # name of an arbitrary input variable
            # build up a sub-frame to work on
            col_list = [c for c in set(op.partition_by)]
            col_set = set(col_list)
            for c in op.order_by:
                if c not in col_set:
                    col_list.append(c)
                    col_set.add(c)
            order_cols = [c for c in col_list]  # must be partition by followed by order
            for (k, opk) in op.ops.items():
                # assumes all args are column names or values, enforce this earlier
                if len(opk.args) > 0:
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in col_set:
                            col_list.append(value_name)
                            col_set.add(value_name)
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        key = str(opk.args[0].value)
                        if key not in data_algebra_temp_cols.keys():
                            value_name = "data_algebra_extend_temp_col_" + str(
                                len(data_algebra_temp_cols)
                            )
                            data_algebra_temp_cols[key] = value_name
                            col_list.append(value_name)
                            res[value_name] = opk.args[0].value
                    else:
                        raise ValueError("opk must be a ColumnReference or Value")
            ascending = [c not in set(op.reverse) for c in col_list]
            subframe = res[col_list].reset_index(drop=True)
            subframe["_data_algebra_orig_index"] = subframe.index
            if len(order_cols) > 0:
                subframe = subframe.sort_values(
                    by=col_list, ascending=ascending
                ).reset_index(drop=True)
            subframe[standin_name] = 1
            if len(op.partition_by) > 0:
                opframe = subframe.groupby(op.partition_by, observed=True)
                #  Groupby preserves the order of rows within each group.
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
            else:
                opframe = subframe.groupby([standin_name], observed=True)
            # perform calculations
            for (k, opk) in op.ops.items():
                # work on a slice of the data frame
                # Availability roughly documented in:
                # https://github.com/WinVector/data_algebra/blob/main/Examples/Methods/data_algebra_catalog.ipynb
                # (essentially none but _ngroup)
                if len(opk.args) <= 0:
                    # check for and remove initial underbar
                    assert isinstance(opk.op, str)
                    assert len(opk.op) > 1
                    assert opk.op[0] == "_"
                    zero_op = opk.op[1:]
                    if zero_op in {"row_number", "count"}:
                        subframe[k] = opframe.cumcount() + 1
                    elif zero_op in {"ngroup"}:
                        subframe[k] = opframe.ngroup()
                    elif zero_op in {"size"}:
                        transform_op = zero_op
                        try:
                            transform_op = self.transform_op_map[transform_op]
                        except KeyError:
                            pass
                        subframe[k] = opframe[standin_name].transform(
                            transform_op
                        )  # Pandas transform, not data_algebra
                    else:
                        raise KeyError(
                            "not implemented in windowed situation: "
                            + str(k)
                            + ": "
                            + str(opk)
                        )
                else:
                    transform_args = []
                    if len(opk.args) > 1:
                        for i in range(1, len(opk.args)):
                            assert isinstance(opk.args[i], data_algebra.expr_rep.Value)
                        transform_args = [
                            opk.args[i].value for i in range(1, len(opk.args))
                        ]
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in set(col_list):
                            col_list.append(value_name)
                        transform_op = opk.op
                        try:
                            transform_op = self.transform_op_map[transform_op]
                        except KeyError:
                            pass
                        subframe[k] = opframe[value_name].transform(
                            transform_op, *transform_args
                        )  # Pandas transform, not data_algegra
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        value_name = data_algebra_temp_cols[str(opk.args[0].value)]
                        transform_op = opk.op
                        try:
                            transform_op = self.transform_op_map[transform_op]
                        except KeyError:
                            pass
                        subframe[k] = opframe[value_name].transform(
                            transform_op, *transform_args
                        )  # Pandas transform, not data_algegra
                    else:
                        raise ValueError(
                            f"opk must be a ColumnReference or Value ({opk})"
                        )
            # clear some temps
            for value_name in data_algebra_temp_cols.values():
                del res[value_name]
            # copy out results
            subframe = subframe.sort_values(by=["_data_algebra_orig_index"])
            subframe = subframe.loc[:, list(op.ops.keys())]
            subframe = subframe.reset_index(drop=True)
            res = self.add_data_frame_columns_to_data_frame_(res, subframe)
        return res

    def _project_step(self, op, *, data_map):
        """
        Execute a project step, returning a data frame.
        """
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        # check these are forms we are prepared to work with, and build an aggregation dictionary
        # build an agg list: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        # https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with
        # try the following tutorial:
        # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        data_algebra_temp_cols = {}
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        for (k, opk) in op.ops.items():
            if len(opk.args) > 1:
                raise ValueError(
                    "non-trivial aggregation expression: " + str(k) + ": " + str(opk)
                )
            if len(opk.args) > 0:
                if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                    pass
                elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                    key = str(opk.args[0].value)
                    if key not in data_algebra_temp_cols.keys():
                        value_name = "data_algebra_project_temp_col_" + str(
                            len(data_algebra_temp_cols)
                        )
                        data_algebra_temp_cols[key] = value_name
                        res[value_name] = opk.args[0].value
                else:
                    raise ValueError(
                        "windows expression argument must be a column or value: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
        res["_data_table_temp_col"] = 1
        if len(op.group_by) > 0:
            res = res.groupby(op.group_by, observed=True)
        if len(op.ops) > 0:
            cols = {}
            for k, opk in op.ops.items():
                value_name = None
                if len(opk.args) > 0:
                    value_name = str(opk.args[0])
                    if isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        value_name = data_algebra_temp_cols[value_name]
                transform_op = opk.op
                if len(opk.args) > 0:
                    transform_op = opk.op
                    try:
                        transform_op = self.transform_op_map[transform_op]
                    except KeyError:
                        pass
                    vk = res[value_name].agg(transform_op)
                else:
                    # expect and strip off initial underbar
                    assert isinstance(transform_op, str)
                    assert len(transform_op) > 1
                    assert transform_op[0] == "_"
                    transform_op = transform_op[1:]
                    try:
                        transform_op = self.transform_op_map[transform_op]
                    except KeyError:
                        pass
                    vk = res["_data_table_temp_col"].agg(transform_op)
                cols[k] = vk
        else:
            cols = {"_data_table_temp_col": res["_data_table_temp_col"].agg("sum")}
        # agg can return scalars, which then can't be made into a self.pd.DataFrame
        res = self.columns_to_frame_(cols)
        res = res.reset_index(
            drop=(len(op.group_by) < 1) or (res.shape[0] <= 0)
        )  # grouping variables in the index
        missing_group_cols = set(op.group_by) - set(res.columns)
        if res.shape[0] > 0:
            if len(missing_group_cols) != 0:
                raise ValueError("Missing column groups")
        else:
            for g in missing_group_cols:
                res[g] = []
        if "_data_table_temp_col" in res.columns:
            res = res.drop("_data_table_temp_col", axis=1, inplace=False)
        # double check shape is what we expect
        if not data_algebra.util.table_is_keyed_by_columns(res, op.group_by):
            raise ValueError("result wasn't keyed by group_by columns")
        return res

    def _select_rows_step(self, op, *, data_map):
        """
        Execute a select rows step, returning a data frame.
        """
        if op.node_name != "SelectRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        if res.shape[0] < 1:
            return res
        selection = op.expr.act_on(res, data_model=self)
        res = res.loc[selection, :].reset_index(drop=True, inplace=False)
        return res

    def _select_columns_step(self, op, *, data_map):
        """
        Execute a select columns step, returning a data frame.
        """
        if op.node_name != "SelectColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectColumnsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        return res[op.column_selection]

    def _drop_columns_step(self, op, *, data_map):
        """
        Execute a drop columns step, returning a data frame.
        """
        if op.node_name != "DropColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.DropColumnsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        column_selection = [c for c in res.columns if c not in op.column_deletions]
        return res[column_selection]

    def _order_rows_step(self, op, *, data_map):
        """
        Execute an order rows step, returning a data frame.
        """
        if op.node_name != "OrderRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        if res.shape[0] > 1:
            ascending = [
                False if ci in set(op.reverse) else True for ci in op.order_columns
            ]
            res = res.sort_values(by=op.order_columns, ascending=ascending).reset_index(
                drop=True
            )
        if (op.limit is not None) and (res.shape[0] > op.limit):
            res = res.iloc[range(op.limit), :].reset_index(drop=True)
        return res

    def _map_columns_step(self, op, *, data_map):
        """
        Execute a map columns step, returning a data frame.
        """
        if op.node_name != "MapColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.MapColumnsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        res = res.rename(columns=op.column_remapping)
        if (op.column_deletions is not None) and (len(op.column_deletions) > 0):
            column_selection = [c for c in res.columns if c not in op.column_deletions]
            res = res[column_selection]
        return res

    def _rename_columns_step(self, op, *, data_map):
        """
        Execute a rename columns step, returning a data frame.
        """
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        return res.rename(columns=op.reverse_mapping)

    # noinspection PyMethodMayBeStatic
    def standardize_join_code_(self, jointype):
        """
        Map join names to Pandas names. Internal method.
        """
        assert isinstance(jointype, str)
        jointype = jointype.lower()
        mp = {
            "full": "outer",
            "cross": "outer",  # cross new to Pandas 1.2.0 December 2020
        }
        try:
            return mp[jointype]
        except KeyError:
            pass
        return jointype

    def _natural_join_step(self, op, *, data_map):
        """
        Execute a natural join step, returning a data frame.
        """
        if op.node_name != "NaturalJoinNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        left = self._eval_value_source(op.sources[0], data_map=data_map)
        right = self._eval_value_source(op.sources[1], data_map=data_map)
        if (left.shape[0] == 0) and (right.shape[0] == 0):
            # pandas seems to not like this case
            return self.pd.DataFrame({k: [] for k in op.columns_produced()})
        common_cols = set([c for c in left.columns]).intersection(
            [c for c in right.columns]
        )
        type_checks = data_algebra.util.check_columns_appear_compatible(
            left, right, columns=common_cols
        )
        if type_checks is not None:
            raise ValueError(f"join: incompatible column types: {type_checks}")
        on_a = op.on_a
        on_b = op.on_b
        scratch_col = None  # extra column to prevent empty-on issues
        if len(on_a) <= 0:
            scratch_col = "data_algebra_temp_merge_col"
            on_a = [scratch_col]
            on_b = [scratch_col]
            left[scratch_col] = 1
            right[scratch_col] = 1
        # noinspection PyUnresolvedReferences
        res = self.pd.merge(
            left=left,
            right=right,
            how=self.standardize_join_code_(op.jointype),
            left_on=on_a,
            right_on=on_b,
            sort=False,
            suffixes=("", "_tmp_right_col"),
        )
        res = res.reset_index(drop=True)
        if scratch_col is not None:
            del res[scratch_col]
        on_a_set = set(op.on_a)
        for c in common_cols:
            if c not in on_a_set:
                is_null = res[c].isnull()
                res.loc[is_null, c] = res.loc[is_null, c + "_tmp_right_col"]
                res = res.drop(c + "_tmp_right_col", axis=1, inplace=False)
        res = res.reset_index(drop=True)
        return res

    def _concat_rows_step(self, op, *, data_map):
        """
        Execute a concat rows step, returning a data frame.
        """
        if op.node_name != "ConcatRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConcatRowsNode"
            )
        left = self._eval_value_source(op.sources[0], data_map=data_map)
        right = self._eval_value_source(op.sources[1], data_map=data_map)
        if op.id_column is not None:
            if left.shape[0] > 0:
                left[op.id_column] = op.a_name
            else:
                left[op.id_column] = []
            if right.shape[0] > 0:
                right[op.id_column] = op.b_name
            else:
                right[op.id_column] = []
        if left.shape[0] < 1:
            return right
        if right.shape[0] < 1:
            return left
        type_checks = data_algebra.util.check_columns_appear_compatible(left, right)
        if type_checks is not None:
            raise ValueError(f"concat: incompatible column types: {type_checks}")
        # noinspection PyUnresolvedReferences
        res = self.pd.concat([left, right], axis=0, ignore_index=True, sort=False)
        res = res.reset_index(drop=True)
        return res

    def _convert_records_step(self, op, *, data_map):
        """
        Execute record conversion step, returning a data frame.
        """
        if op.node_name != "ConvertRecordsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConvertRecordsNode"
            )
        res = self._eval_value_source(op.sources[0], data_map=data_map)
        return op.record_map.transform(res, local_data_model=self)

    # expression helpers
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        return value
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on
        """
        return arg[value]
    
    def act_on_expression(self, *, arg, values: List, op):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on
        """
        assert isinstance(values, List)
        assert isinstance(op, data_algebra.expr_rep.Expression)
        op_name = op.op
        # check user fns
        # first check chosen mappings
        try:
            method_to_call = self.user_fun_map[op_name]
            return method_to_call(*values)
        except KeyError:
            pass
        # check chosen mappings
        try:
            method_to_call = self.impl_map[op_name]
            return method_to_call(*values)
        except KeyError:
            pass
        # special zero argument functions
        if len(values) == 0:
            if op_name == "_uniform":
                return numpy.random.uniform(size=arg.shape[0])
            else:
                KeyError(f"zero-argument function {op_name} not found")
        # now see if argument (usually Pandas) can do this
        # doubt we hit in this, as most exposed methods are window methods
        try:
            method = getattr(values[0], op_name)
            if callable(method):
                return method(*values[1:])
        except AttributeError:
            pass
        # now see if numpy can do this
        try:
            fn = numpy.__dict__[op_name]
            if callable(fn):
                return fn(*values)
        except KeyError:
            pass
        raise KeyError(f"function {op_name} not found")
