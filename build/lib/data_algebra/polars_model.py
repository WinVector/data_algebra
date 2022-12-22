
"""
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: fully not implemented yet.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import polars as pl

import data_algebra
import data_algebra.data_model
import data_algebra.util
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components
import data_algebra.expression_walker


def _build_lit(v):
    if isinstance(v, int):
        # Polars defaults ints in constructor to Int64,
        # but ints in lit to Int32. Try to prevent type clashes
        return pl.lit(v, pl.Int64)
    return pl.lit(v)


def _reduce_plus(*args):
    assert len(args) > 0
    res = args[0]
    for i in range(1, len(args)):
        res = res + args[i]
    return res


def _reduce_times(*args):
    assert len(args) > 0
    res = args[0]
    for i in range(1, len(args)):
        res = res * args[i]
    return res


def _reduce_and(*args):
    assert len(args) > 0
    res = args[0]
    for i in range(1, len(args)):
        res = res & args[i]
    return res


def _reduce_or(*args):
    assert len(args) > 0
    res = args[0]
    for i in range(1, len(args)):
        res = res | args[i]
    return res


_da_temp_zero_column_name = "_da_temp_zero_column"
_da_temp_one_column_name = "_da_temp_one_column"


class ExpressionRequirements(data_algebra.expression_walker.ExpressionWalker):
    """
    Class to collect what accommodations an expression needs.
    """
    collect_required: bool
    zero_constant_required: bool
    one_constant_required: bool

    def __init__(self) -> None:
        data_algebra.expression_walker.ExpressionWalker.__init__(
            self,
        )
        self.collect_required = False
        self.zero_constant_required = False
        self.one_constant_required = False
        self._needs_zero_constant= {
            "coalesce0",
        }
        self._needs_one_constant= {
            "size", "_size",
            "count", "_count",
            "cumcount", "_cumcount",
        }
        self._collect_required = {
            "uniform", "_uniform",
            "ngroup", "_ngroup",
        }
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        assert not isinstance(value, PolarsTerm)
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: None
        :param value: column name
        :return: arg acted on
        """
        assert arg is None
        assert isinstance(value, str)
    
    def act_on_expression(self, *, arg, values: List, op):
        """
        Action for a column name.

        :param arg: None
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on
        """
        assert arg is None
        assert isinstance(values, List)
        assert isinstance(op, data_algebra.expr_rep.Expression)
        # work on expression requirements
        if (len(values) == 0) or (op.op in self._needs_one_constant):
            self.one_constant_required = True
        if op.op in self._needs_zero_constant:
            self.zero_constant_required = True
        if op.op in self._collect_required:
            self.collect_required = True
    
    def add_in_temp_columns(self, temp_v_columns: List):
        """
        Add required temp columns to temp_v_columns_list
        """
        if self.zero_constant_required:
            temp_v_columns.append(_build_lit(0).alias(_da_temp_zero_column_name))
        if self.one_constant_required:
            temp_v_columns.append(_build_lit(1).alias(_da_temp_one_column_name))


class PolarsTerm:
    """
    Class to carry Polars expression term and annotations about expression tree.
    """

    polars_term: Any
    lit_value: Any
    is_literal: bool
    is_column: bool

    def __init__(
        self, 
        *, 
        polars_term = None, 
        is_literal: bool = False,
        is_column: bool = False,
        lit_value = None,
        ) -> None:
        """
        Carry a Polars expression term (polars_term) plus annotations.

        :param polars_term: Optional Polars expression (None means collect info, not a true term)
        :param is_literal: True if term is a constant
        :param is_column: True if term is a column name
        :param lit_value: original value for a literal
        :param inputs: inputs to expression node
        """
        assert isinstance(is_literal, bool)
        assert isinstance(is_column, bool)
        if lit_value is not None:
            assert is_literal
        self.lit_value = lit_value
        self.polars_term = polars_term
        self.is_literal = is_literal


def _unpack_lits(v):
    if isinstance(v, PolarsTerm):
        if v.is_literal:
            return v.lit_value
        else:
            return v.polars_term
    elif isinstance(v, Iterable):
        return [_unpack_lits(vi) for vi in v]
    else:
        raise ValueError(f"unexpected type to _unpack_lits: {type(v)}")


def _mapv(a, b: Dict, c):
    # TODO: find out if there is another way to do this
    assert isinstance(b, Dict)
    res = _build_lit(c)
    for k, v in b.items():
        res = pl.when(a == _build_lit(k)).then(_build_lit(v)).otherwise(res)
    return res


def _populate_expr_impl_map() -> Dict[int, Dict[str, Callable]]:
    """
    Map symbols to implementations.
    """
    # TODO: fill in more
    impl_map_0 = {
        "count": lambda : pl.col(_da_temp_one_column_name).cumsum(),  # ugly SQL def
        "_count": lambda : pl.col(_da_temp_one_column_name).cumsum(),  # ugly SQL def
        "cumcount": lambda : pl.col(_da_temp_one_column_name).cumsum(),
        "_cumcount": lambda : pl.col(_da_temp_one_column_name).cumsum(),
        "row_number": lambda : pl.col(_da_temp_one_column_name).cumsum(),
        "_row_number": lambda : pl.col(_da_temp_one_column_name).cumsum(),
        "size": lambda : pl.col(_da_temp_one_column_name).sum(),
        "_size": lambda : pl.col(_da_temp_one_column_name).sum(),
    }
    impl_map_1 = {
        "-": lambda x: 0 - x,
        "abs": lambda x: x.abs(),
        "all": lambda x: x.all(),
        "any": lambda x: x.any(),
        "any_value": lambda x: x.min(),
        "arccos": lambda x: x.arccos(),
        "arccosh": lambda x: x.arccosh(),
        "arcsin": lambda x: x.arcsin(),
        "arcsinh": lambda x: x.arcsinh(),
        "arctan": lambda x: x.arctan(),
        "arctan2": lambda x: x.arctan2(),
        "arctanh": lambda x: x.arctanh(),
        "as_int64": lambda x: x.cast(int),
        "as_str": lambda x: x.cast(str),
        "base_Sunday": lambda x: x.base_Sunday(),
        "bfill": lambda x: x.fill_null(strategy='backward'),
        "ceil": lambda x: x.ceil(),
        "coalesce0": lambda x: pl.when(x.is_null()).then(pl.col(_da_temp_zero_column_name)).otherwise(x),
        "cos": lambda x: x.cos(),
        "cosh": lambda x: x.cosh(),
        "count": lambda x: pl.col(_da_temp_one_column_name).cumsum(),
        "cumcount": lambda x: pl.col(_da_temp_one_column_name).cumsum(),
        "cummax": lambda x: x.cummax(),
        "cummin": lambda x: x.cummin(),
        "cumprod": lambda x: x.cumprod(),
        "cumsum": lambda x: x.cumsum(),
        "datetime_to_date": lambda x: x.datetime_to_date(),
        "dayofmonth": lambda x: x.dayofmonth(),
        "dayofweek": lambda x: x.dayofweek(),
        "dayofyear": lambda x: x.dayofyear(),
        "exp": lambda x: x.exp(),
        "expm1": lambda x: x.expm1(),
        "ffill": lambda x: x.fill_null(strategy='forward'),
        "first": lambda x: x.first(),
        "floor": lambda x: x.floor(),
        "format_date": lambda x: x.format_date(),
        "format_datetime": lambda x: x.format_datetime(),
        "is_bad": lambda x: x.is_null() | x.is_infinite() | x.is_nan(),  # recommend only for numeric columns
        "is_inf": lambda x: x.is_infinite(),
        "is_nan": lambda x: x.is_nan(),
        "is_null": lambda x: x.is_null(),
        "last": lambda x: x.last(),
        "log": lambda x: x.log(),
        "log10": lambda x: x.log10(),
        "log1p": lambda x: x.log1p(),
        "max": lambda x: x.max(),
        "mean": lambda x: x.mean(),
        "median": lambda x: x.median(),
        "min": lambda x: x.min(),
        "month": lambda x: x.month(),
        "nunique": lambda x: x.n_unique(),
        "quarter": lambda x: x.quarter(),
        "rank": lambda x: x.rank(),
        "round": lambda x: x.round(decimals=0),
        "shift": lambda x: x.shift(),
        "sign": lambda x: x.sign(),
        "sin": lambda x: x.sin(),
        "sinh": lambda x: x.sinh(),
        "size": lambda x: pl.col(_da_temp_one_column_name).sum(),
        "sqrt": lambda x: x.sqrt(),
        "std": lambda x: x.std(),
        "sum": lambda x: x.sum(),
        "tanh": lambda x: x.tanh(),
        "var": lambda x: x.var(),
        "weekofyear": lambda x: x.weekofyear(),
    }
    impl_map_2 = {
        "-": lambda a, b: a - b,
        "**": lambda a, b: a ** b,
        "/": lambda a, b: a / b,
        "//": lambda a, b: a // b,
        "%": lambda a, b: a % b,
        "%/%": lambda a, b: a / b,
        "around": lambda a, b: a.round(b),
        "coalesce": lambda a, b: pl.when(a.is_null()).then(b).otherwise(a),
        "date_diff": lambda a, b: a.date_diff(b),
        "is_in": lambda a, b: a.is_in(b),
        "mod": lambda a, b: a % b,
        "remainder": lambda a, b: a % b,
        "shift": lambda a, b: a.shift(b),
        "timestamp_diff": lambda a, b: a.timestamp_diff(b),
        "==": lambda a, b: a == b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,  
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b, 
        "!=": lambda a, b: a != b,
        "not": lambda x: x == False,
        "~": lambda x: x == False,
        "!": lambda x: x == False,
        # datetime parsing from https://stackoverflow.com/a/71759536/6901725
        "parse_date": lambda x, format : x.cast(str).str.strptime(pl.Date, fmt=format, strict=False).cast(pl.Date),
        "parse_datetime": lambda x, format : x.cast(str).str.strptime(pl.Datetime, fmt=format, strict=False).cast(pl.Datetime),
    }
    impl_map_3 = {
        "if_else": lambda a, b, c: pl.when(a.is_null()).then(pl.lit(None)).otherwise(pl.when(a).then(b).otherwise(c)),
        "mapv": _mapv,
        "trimstr": lambda a, b, c: a.trimstr(b, c),
        "where": lambda a, b, c: pl.when(a.is_null()).then(c).otherwise(pl.when(a).then(b).otherwise(c)),
    }
    impl_map = {
        0: impl_map_0,
        1: impl_map_1,
        2: impl_map_2,
        3: impl_map_3,
    }
    # could also key the map by grouped, partitioned, regular situation
    return impl_map


class PolarsModel(data_algebra.data_model.DataModel, data_algebra.expression_walker.ExpressionWalker):
    """
    Interface for realizing the data algebra as a sequence of steps over Polars https://www.pola.rs .

    Note: not fully implemented yet.
    """

    use_lazy_eval: bool
    presentation_model_name: str
    _method_dispatch_table: Dict[str, Callable]
    _expr_impl_map: Dict[int, Dict[str, Callable]]
    _impl_map_arbitrary_arity: Dict[str, Callable]
    _collect_required: Set[str]
    _rng: Any

    def __init__(self, *, use_lazy_eval: bool = True):
        data_algebra.data_model.DataModel.__init__(
            self, presentation_model_name="pl", module=pl
        )
        data_algebra.expression_walker.ExpressionWalker.__init__(
            self,
        )
        self._rng = np.random.default_rng()
        assert isinstance(use_lazy_eval, bool)
        self.use_lazy_eval = use_lazy_eval
        self._method_dispatch_table = {
            "ConcatRowsNode": self._concat_rows_step,
            "ConvertRecordsNode": self._convert_records_step,
            "DropColumnsNode": self._drop_columns_step,
            "ExtendNode": self._extend_step,
            "MapColumnsNode": self._map_columns_step,
            "NaturalJoinNode": self._natural_join_step,
            "OrderRowsNode": self._order_rows_step,
            "ProjectNode": self._project_step,
            "RenameColumnsNode": self._rename_columns_step,
            "SelectColumnsNode": self._select_columns_step,
            "SelectRowsNode": self._select_rows_step,
            "SQLNode": self._sql_proxy_step,
            "TableDescription": self._table_step,
        }
        self._expr_impl_map = _populate_expr_impl_map()
        self._impl_map_arbitrary_arity = {
            "concat": lambda *args: pl.concat_str(args),
            "fmax": lambda *args: pl.max(args),
            "fmin": lambda *args: pl.min(args),
            "maximum": lambda *args: pl.max(args),
            "minimum": lambda *args: pl.min(args),
            "+": _reduce_plus,
            "*": _reduce_times,
            "and": _reduce_and,
            "&": _reduce_and,
            "or": _reduce_or,
            "|": _reduce_or,
        }
        self._want_literals_unpacked = {
            "around",
            "is_in",
            "mapv",
            "parse_date", "parse_datetime",
            "shift",
            }

    def data_frame(self, arg=None):
        """
        Build a new data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """
        if arg is None:
            return pl.DataFrame()
        if isinstance(arg, pl.DataFrame):
            return arg
        if isinstance(arg, pl.LazyFrame):
            return arg.collect()
        return pl.DataFrame(arg)

    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
        return isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame)
    
    def clean_copy(self, df):
        """
        Copy of data frame without indices.
        """
        assert self.is_appropriate_data_instance(df)
        # Polars doesn't need explicit copying due to copy on write semantics
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df
    
    def to_pandas(self, df):
        """
        Convert to Pandas
        """
        assert self.is_appropriate_data_instance(df)
        # Polars doesn't need explicit copying due to copy on write semantics
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df.to_pandas()
    
    def drop_indices(self, df) -> None:
        """
        Drop indices in place.
        """
        assert self.is_appropriate_data_instance(df)
        # no operation needed
    
    def bad_column_positions(self, x):
        """
        Return vector indicating which entries are null (vectorized).
        """
        return x.is_null()

    def concat_rows(self, frame_list: List):
        """
        Concatenate rows from frame_list
        """
        frame_list = list(frame_list)
        assert len(frame_list) > 0
        if len(frame_list) == 1:
            return frame_list[0]
        pl.concat(frame_list, how="vertical")

    def concat_columns(self, frame_list):
        """
        Concatenate columns from frame_list
        """
        frame_list = list(frame_list)
        if len(frame_list) <= 0:
            return None
        if len(frame_list) == 1:
            return frame_list[0]
        res = pl.concat(frame_list, how="horizontal")
        return res

    def table_is_keyed_by_columns(self, table, *, column_names: Iterable[str]) -> bool:
        """
        Check if a table is keyed by a given list of column names.

        :param table: DataFrame
        :param column_names: list of column names
        :return: True if rows are uniquely keyed by values in named columns
        """
        # check for ill-condition
        if isinstance(column_names, str):
            column_names = [column_names]
        else:
            column_names = list(column_names)
        missing_columns = set(column_names) - set(table.columns)
        if len(missing_columns) > 0:
            return False
        # get rid of some corner cases
        if table.shape[0] < 2:
            return True
        if len(column_names) < 1:
            return False
        mx = (
            table
                .select(column_names)
                .with_column(pl.lit(1, pl.Int64).alias("_da_count_tmp"))
                .groupby(column_names)
                .sum()["_da_count_tmp"]
                .max()
        )
        return mx <= 1

    # evaluate

    def eval(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]) -> pl.DataFrame:
        """
        Implementation of Polars evaluation of data algebra operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result
        """
        assert isinstance(data_map, Dict)
        assert isinstance(op, data_algebra.data_ops_types.OperatorPlatform)
        res = self._compose_polars_ops(op=op, data_map=data_map)
        if isinstance(res, pl.LazyFrame):
            res = res.collect()
        assert self.is_appropriate_data_instance(res)
        return res

    def _compose_polars_ops(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Convert to polars operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result or polars ops
        """
        assert isinstance(op, data_algebra.data_ops_types.OperatorPlatform)
        assert isinstance(data_map, Dict)
        res = self._method_dispatch_table[op.node_name](op=op, data_map=data_map)
        return res

    # operator step realizations

    def _concat_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a concat rows step, returning a data frame.
        """
        if op.node_name != "ConcatRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConcatRowsNode"
            )
        common_columns = [c for c in op.columns_produced() if c != op.id_column]
        inputs = [self._compose_polars_ops(s, data_map=data_map) for s in op.sources]
        assert len(inputs) == 2
        inputs = [input_i.select(common_columns) for input_i in inputs]  # get columns in same order
        if op.id_column is not None:
            inputs[0] = inputs[0].with_column(_build_lit(op.a_name).alias(op.id_column))
            inputs[1] = inputs[1].with_column(_build_lit(op.b_name).alias(op.id_column))
        res = pl.concat(inputs, how="vertical")
        return res
    
    def _convert_records_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute record conversion step, returning a data frame.
        """
        if op.node_name != "ConvertRecordsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConvertRecordsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        if isinstance(res, pl.LazyFrame):
            res = res.collect()
        res = op.record_map.transform(res, local_data_model=self)
        if self.use_lazy_eval and (not isinstance(res, pl.LazyFrame)):
            res = res.lazy()
        return res
    
    def _extend_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an extend step, returning a data frame.
        """
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        partition_by = op.partition_by
        temp_v_columns = []
        # see if we need to make partition non-empty
        if len(partition_by) <= 0:
            v_name = "_da_extend_temp_partition_column"
            partition_by = [v_name]
            temp_v_columns.append(_build_lit(1).alias(v_name))
        # pre-scan expressions
        er = ExpressionRequirements()
        for opk in op.ops.values():
            opk.act_on(None, expr_walker=er)
        er.add_in_temp_columns(temp_v_columns)
        value_to_send_to_act = None
        if er.collect_required:
            if isinstance(res, pl.LazyFrame):
                res = res.collect()
            value_to_send_to_act = res
        # work on expressions
        produced_columns = []
        for k, opk in op.ops.items():
            if op.windowed_situation:
                if (len(opk.args) == 1) and isinstance(opk.args[0], data_algebra.expr_rep.Value):
                    # TODO: move this to leave of nested expressions
                    # promote value to column for uniformity of API
                    v_name = f"_da_extend_temp_v_column_{len(temp_v_columns)}"
                    v_value = opk.args[0].value
                    temp_v_columns.append(_build_lit(v_value).alias(v_name))
                    opk = data_algebra.expr_rep.Expression(
                        op=opk.op,
                        args=[data_algebra.expr_rep.ColumnReference(column_name=v_name)],
                        params=opk.params,
                        inline=opk.inline,
                        method=opk.method,
                    )
            fld_k_container = opk.act_on(value_to_send_to_act, expr_walker=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            fld_k = fld_k_container.polars_term
            if op.windowed_situation:
                fld_k = fld_k.over(partition_by)
            produced_columns.append(fld_k.alias(k))
        if len(temp_v_columns) > 0:
            res = res.with_columns(temp_v_columns)
        if len(op.order_by) > 0:
            order_cols = list(partition_by)
            partition_set = set(partition_by)
            for c in op.order_by:
                if c not in partition_set:
                    order_cols.append(c)
            reversed_cols = [True if ci in set(op.reverse) else False for ci in op.order_by]
            res = res.sort(by=op.order_by, reverse=reversed_cols)
        res = res.with_columns(produced_columns)
        if len(temp_v_columns) > 0:
            res = res.select(op.columns_produced())
        if self.use_lazy_eval and isinstance(res, pl.DataFrame):
            res = res.lazy()
        return res

    def _project_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a project step, returning a data frame.
        """
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        group_by = op.group_by
        temp_v_columns = []
        # see if we need to make group_by non-empty
        if len(group_by) <= 0:
            v_name = "_da_project_temp_group_by_column"
            group_by = [v_name]
            temp_v_columns.append(_build_lit(1).alias(v_name))
        # pre-scan expressions
        er = ExpressionRequirements()
        for opk in op.ops.values():
            opk.act_on(None, expr_walker=er)
        er.add_in_temp_columns(temp_v_columns)
        value_to_send_to_act = None
        if er.collect_required:
            if isinstance(res, pl.LazyFrame):
                res = res.collect()
            value_to_send_to_act = res
        # work on expressions
        produced_columns = []
        for k, opk in op.ops.items():
            if (len(opk.args) == 1) and isinstance(opk.args[0], data_algebra.expr_rep.Value):
                # TODO: push this into leaves of nested ops
                # promote value to column for uniformity of API
                v_name = f"_da_project_temp_v_column_{len(temp_v_columns)}"
                v_value = opk.args[0].value
                temp_v_columns.append(_build_lit(v_value).alias(v_name))
                opk = data_algebra.expr_rep.Expression(
                    op=opk.op, 
                    args=[data_algebra.expr_rep.ColumnReference(column_name=v_name)], 
                    params=opk.params, 
                    inline=opk.inline, 
                    method=opk.method,
                )
            fld_k_container = opk.act_on(value_to_send_to_act, expr_walker=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            fld_k = fld_k_container.polars_term
            produced_columns.append(fld_k.alias(k))
        if len(temp_v_columns) > 0:
            res = res.with_columns(temp_v_columns)
        res = res.groupby(group_by).agg(produced_columns)
        if len(temp_v_columns) > 0:
            res = res.select(op.columns_produced())
        if self.use_lazy_eval and isinstance(res, pl.DataFrame):
            res = res.lazy()
        return res
    
    def _natural_join_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a natural join step, returning a data frame.
        """
        if op.node_name != "NaturalJoinNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        inputs = [self._compose_polars_ops(s, data_map=data_map) for s in op.sources]
        assert len(inputs) == 2
        how = op.jointype.lower()
        if how == "full":
            how = "outer"
        coalesce_columns = (
            set(op.sources[0].columns_produced()).intersection(op.sources[1].columns_produced()) 
            - set(op.on_a))
        if how != "right":
            res = inputs[0].join(
                inputs[1],
                left_on=op.on_a,
                right_on=op.on_b,
                how=how,
                suffix = "_da_right_tmp",
            )
            if len(coalesce_columns) > 0:
                res = res.with_columns([
                    pl.when(pl.col(c).is_null())
                        .then(pl.col(c + "_da_right_tmp"))
                        .otherwise(pl.col(c))
                        .alias(c)
                    for c in coalesce_columns
                ])
        else:
            # simulate right join with left join
            res = inputs[1].join(
                inputs[0],
                left_on=op.on_b,
                right_on=op.on_a,
                how="left",
                suffix = "_da_left_tmp",
            )
            if len(coalesce_columns) > 0:
                res = res.with_columns([
                    pl.when(pl.col(c + "_da_left_tmp").is_null())
                        .then(pl.col(c))
                        .otherwise(pl.col(c + "_da_left_tmp"))
                        .alias(c)
                    for c in coalesce_columns
                ])
        res = res.select(op.columns_produced())
        return res
    
    def _order_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an order rows step, returning a data frame.
        """
        if op.node_name != "OrderRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        reversed_cols = [True if ci in set(op.reverse) else False for ci in op.order_columns]
        res = res.sort(by=op.order_columns, reverse=reversed_cols)
        if op.limit is not None:
            res = res.head(op.limit)
        return res

    def _rename_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a rename columns step, returning a data frame.
        """
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        if isinstance(res, pl.LazyFrame):
            # work around https://github.com/pola-rs/polars/issues/5882#issue-1507040380
            res = res.collect()
        res = res.rename(op.reverse_mapping)
        res = res.select(op.columns_produced())
        if self.use_lazy_eval and isinstance(res, pl.DataFrame):
            res = res.lazy()
        return res

    def _map_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a map columns step, returning a data frame.
        """
        if op.node_name != "MapColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.MapColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        if isinstance(res, pl.LazyFrame):
            # work around https://github.com/pola-rs/polars/issues/5882#issue-1507040380
            res = res.collect()
        res = res.rename(op.column_remapping)
        res = res.select(op.columns_produced())
        if self.use_lazy_eval and isinstance(res, pl.DataFrame):
            res = res.lazy()
        return res

    def _select_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a select columns step, returning a data frame.
        """
        if op.node_name != "SelectColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        res = res.select(op.columns_produced())
        return res

    def _drop_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a drop columns step, returning a data frame.
        """
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        res = res.select(op.columns_produced())
        return res

    def _select_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a select rows step, returning a data frame.
        """
        if op.node_name != "SelectRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        temp_v_columns = []
        # pre-scan expressions
        er = ExpressionRequirements()
        for opk in op.ops.values():
            opk.act_on(None, expr_walker=er)
        er.add_in_temp_columns(temp_v_columns)
        value_to_send_to_act = None
        if er.collect_required:
            if isinstance(res, pl.LazyFrame):
                res = res.collect()
            value_to_send_to_act = res
        # work on expression
        if len(temp_v_columns) > 0:
            res = res.with_columns(temp_v_columns)
        selection = op.expr.act_on(value_to_send_to_act, expr_walker=self)  # PolarsTerm
        assert isinstance(selection, PolarsTerm)
        res = res.filter(selection.polars_term)
        if len(temp_v_columns) > 0:
            res = res.select(op.columns_produced())
        if self.use_lazy_eval and isinstance(res, pl.DataFrame):
            res = res.lazy()
        return res

    def _sql_proxy_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        execute SQL
        """
        if op.node_name != "SQLNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SQLNode"
            )
        db_handle = data_map[op.view_name]
        # would like (but causes cicrular import) assert isinstance(db_handle, data_algebra.db_model.DBHandle)
        res = db_handle.read_query("\n".join(op.sql))
        res = self.data_frame(res)
        assert self.is_appropriate_data_instance(res)
        if self.use_lazy_eval and (not isinstance(res, pl.LazyFrame)):
            res = res.lazy()
        res = res.select(op.columns_produced())
        return res
    
    def _table_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Return a data frame from table description and data_map.
        """
        if op.node_name != "TableDescription":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        res = data_map[op.table_name]
        if not self.is_appropriate_data_instance(res):
            raise ValueError(
                "data_map[" + op.table_name + "] was not the right type"
            )
        if self.use_lazy_eval and (not isinstance(res, pl.LazyFrame)):
            res = res.lazy()
        res = res.select(op.columns_produced())
        return res
    
    # cdata transforms

    def blocks_to_rowrecs(self, data, *, blocks_in):
        """
        Convert a block record (record spanning multiple rows) into a rowrecord (record in a single row).

        :param data: data frame to be transformed
        :param blocks_in: cdata record specification
        :return: transformed data frame
        """
        assert self.is_appropriate_data_instance(data)
        assert isinstance(data, pl.DataFrame)
        assert blocks_in is not None
        assert blocks_in.control_table.shape[0] > 1
        assert len(blocks_in.control_table_keys) > 0
        data = data.select(blocks_in.block_columns)
        assert set(data.columns) == set(blocks_in.block_columns)
        # table must be keyed by record_keys + control_table_keys
        if data.shape[0] < 1:
            return pl.DataFrame({c: [] for c in blocks_in.row_columns})
        if not self.table_is_keyed_by_columns(
            data, column_names=blocks_in.record_keys + blocks_in.control_table_keys
        ):
            raise ValueError(
                "table is not keyed by blocks_in.record_keys + blocks_in.control_table_keys"
            )
        # split on block keys
        split = data.partition_by(blocks_in.control_table_keys)
        # check same number of ids for each block
        # could also double check id columns are identical
        for i in range(1, len(split)):
            assert split[i].shape[0] == split[0].shape[0]
        sk = None
        if (blocks_in.record_keys is not None) and (len(blocks_in.record_keys) > 0):
            # ensure sorted in record order
            split = [s.sort(blocks_in.record_keys) for s in split]
            # capture the record keys
            sk = split[0][blocks_in.record_keys]
        # limit and rename columns

        def limit_and_rename_cols(s):
            # get keying
            keying = s[0, blocks_in.control_table_keys]
            keys = keying.join(
                self.data_frame(blocks_in.control_table), 
                on=blocks_in.control_table_keys, 
                how="left")
            assert keys.shape[0] == 1
            keys = keys.drop(blocks_in.control_table_keys)
            if (blocks_in.record_keys is not None) and (len(blocks_in.record_keys) > 0):
                s = s.drop(blocks_in.record_keys + blocks_in.control_table_keys)
            else:
                s = s.drop(blocks_in.control_table_keys)
            assert keys.shape[1] == s.shape[1]
            s.columns = [keys[0, i] for i in range(keys.shape[1])]
            return s

        split = [limit_and_rename_cols(s) for s in split]
        if sk is not None:
            res = pl.concat([sk] + split, how="horizontal")
        else:
            res = pl.concat(split, how="horizontal")
        if (blocks_in.record_keys is not None) and (len(blocks_in.record_keys) > 0):
            res = res.sort(blocks_in.record_keys)
        return res
    
    def rowrecs_to_blocks(
        self,
        data,
        *,
        blocks_out,
    ):
        """
        Convert rowrecs (single row records) into block records (multiple row records).

        :param data: data frame to transform.
        :param blocks_out: cdata record specification.
        :return: transformed data frame
        """
        assert self.is_appropriate_data_instance(data)
        assert isinstance(data, pl.DataFrame)
        assert blocks_out is not None
        assert blocks_out.control_table.shape[0] > 1
        assert len(blocks_out.control_table_keys) > 0
        data = data.select(blocks_out.row_columns)
        assert set(data.columns) == set(blocks_out.row_columns)
        if data.shape[0] < 1:
            return pl.DataFrame({c: [] for c in blocks_out.block_columns})
        if not self.table_is_keyed_by_columns(data, column_names=blocks_out.record_keys):
            raise ValueError(
                "table is not keyed by blocks_out.record_keys"
            )
        ct = self.data_frame(blocks_out.control_table)
        new_names = [ct.columns[j] for j in range(ct.shape[1]) if ct.columns[j] not in set(blocks_out.control_table_keys)]

        def extract_rows(i):
            ct_keys = ct[i, blocks_out.control_table_keys]
            col_names = [ct[i, j] for j in range(ct.shape[1]) if ct.columns[j] not in set(blocks_out.control_table_keys)]
            new_dat = data[:, col_names]
            new_dat.columns = new_names
            if (blocks_out.record_keys is not None) and (len(blocks_out.record_keys) > 0):
                row = data[blocks_out.record_keys]
                row = row.with_columns([_build_lit(ct_keys[0, c]).alias(c) for c in ct_keys.columns])
            else:
                row = pl.DataFrame({c: [ct_keys[0, c]] * data.shape[0] for c in ct_keys.columns})
            row = pl.concat([
                row, new_dat],
                how="horizontal"
            )
            return row

        rows = [extract_rows(i) for i in range(ct.shape[0])]
        res = pl.concat(rows, how="vertical")
        if (blocks_out.record_keys is not None) and (len(blocks_out.record_keys) > 0):
            res = res.sort(blocks_out.record_keys + blocks_out.control_table_keys)
        else:
            res = res.sort(blocks_out.control_table_keys)
        return res
    
    # expression helpers
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        assert not isinstance(value, PolarsTerm)
        if isinstance(value, (Dict, List, Set)):
            return PolarsTerm(polars_term=None, is_literal=True, lit_value=value)
        else:
            return PolarsTerm(polars_term=_build_lit(value), is_literal=True, lit_value=value)
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: None
        :param value: column name
        :return: arg acted on
        """
        assert isinstance(arg, (pl.DataFrame, type(None)))
        assert isinstance(value, str)
        return PolarsTerm(polars_term=pl.col(value), is_column=True)
    
    def act_on_expression(self, *, arg, values: List, op):
        """
        Action for a column name.

        :param arg: None
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on
        """
        assert isinstance(arg, (pl.DataFrame, type(None)))
        assert isinstance(values, List)
        assert isinstance(op, data_algebra.expr_rep.Expression)
        # process inputs
        for v in values:
            assert isinstance(v, (List, PolarsTerm))
        want_literals_unpacked = (op.op in self._want_literals_unpacked)
        if want_literals_unpacked:
            args = _unpack_lits(values)
        else:
            args = [v.polars_term for v in values]
        # lookup method
        f = None
        arity = len(values)
        if (f is None) and (arity == 0):
            if op.op in ["_uniform", "uniform"]:
                assert isinstance(arg, pl.DataFrame)
                return PolarsTerm(
                    polars_term=pl.Series(
                        values=self._rng.uniform(0.0, 1.0, arg.shape[0]),
                        dtype=pl.datatypes.Float64,
                        dtype_if_empty=pl.datatypes.Float64),
                )
        if (f is None): 
            if op.op in ["_ngroup", "ngroup"]:
                assert isinstance(arg, pl.DataFrame)
                # n_groups = arg.groupby(["x"]).apply(lambda x: x.head(1)).shape[0]
                raise ValueError(f" {op.op} not implemented for Polars adapter, yet")
        if f is None:
            try:
                f = self._expr_impl_map[len(values)][op.op]
            except KeyError:
                pass
        if (f is None) and (arity > 0):
            try:
                f = self._impl_map_arbitrary_arity[op.op]
            except KeyError:
                pass
        if f is None:
            raise ValueError(f"failed to lookup {op}")
        # apply method
        res = f(*args)
        # wrap result
        return PolarsTerm(
            polars_term=res,
        )


def register_polars_model(key:Optional[str] = None):
    # register data model
    common_key = "default_Polars_model"
    if common_key not in data_algebra.data_model.data_model_type_map.keys():
        pl_model = PolarsModel()
        data_algebra.data_model.data_model_type_map[common_key] = pl_model
        data_algebra.data_model.data_model_type_map["<class 'polars.internals.dataframe.frame.DataFrame'>"] = pl_model
        data_algebra.data_model.data_model_type_map[str(type(pl_model.data_frame()))] = pl_model
        data_algebra.data_model.data_model_type_map["<class 'polars.internals.lazyframe.frame.LazyFrame'>"] = pl_model
        data_algebra.data_model.data_model_type_map[str(type(pl_model.data_frame().lazy()))] = pl_model
        if key is not None:
            assert isinstance(key, str)
            data_algebra.data_model.data_model_type_map[key] = pl_model
