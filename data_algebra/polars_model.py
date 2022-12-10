
"""
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: fully not implemented yet.
"""

from typing import Any, Callable, Dict, List, Optional

import polars as pl

import data_algebra
import data_algebra.data_model
import data_algebra.util
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components


class PolarsTerm:
    """
    Class to carry Polars expression term and annotations.
    """
    def __init__(self, *, polars_term, is_literal: bool) -> None:
        assert isinstance(is_literal, bool)
        assert polars_term is not None
        self.polars_term = polars_term
        self.is_literal = is_literal


def _raise_not_impl(nm: str):   # TODO: get rid of this
    raise ValueError(f" {nm} not implemented for Polars adapter, yet")


def _populate_expr_impl_map() -> Dict[int, Dict[str, Callable]]:
    """
    Map symbols to implementations.
    """
    # TODO: fill in more
    impl_map_0 = {
        "count": lambda : _raise_not_impl("count"),  # TODO: implement
        "_count": lambda : _raise_not_impl("_count"),  # TODO: implement
        "ngroup": lambda : _raise_not_impl("ngroup"),  # TODO: implement
        "_ngroup": lambda : _raise_not_impl("_ngroup"),  # TODO: implement
        "row_number": lambda : _raise_not_impl("row_number"),  # TODO: implement
        "_row_number": lambda : _raise_not_impl("_row_number"),  # TODO: implement
        "size": lambda : _raise_not_impl("size"),  # TODO: implement
        "_size": lambda : _raise_not_impl("_size"),  # TODO: implement
        "uniform": lambda : _raise_not_impl("uniform"),  # TODO: implement
        "_uniform": lambda : _raise_not_impl("_uniform"),  # TODO: implement
    }
    impl_map_1 = {
        "-": lambda x: 0 - x,
        "abs": lambda x: x.abs(),
        "all": lambda x: x.all(),
        "any": lambda x: x.any(),
        "any_value": lambda x: x.any_value(),
        "arccos": lambda x: x.arccos(),
        "arccosh": lambda x: x.arccosh(),
        "arcsin": lambda x: x.arcsin(),
        "arcsinh": lambda x: x.arcsinh(),
        "arctan": lambda x: x.arctan(),
        "arctan2": lambda x: x.arctan2(),
        "arctanh": lambda x: x.arctanh(),
        "as_int64": lambda x: x.as_int64(),
        "as_str": lambda x: x.as_str(),
        "base_Sunday": lambda x: x.base_Sunday(),
        "bfill": lambda x: x.bfill(),
        "ceil": lambda x: x.ceil(),
        "coalesce": lambda x: x.coalesce(0),
        "coalesce0": lambda x: x.coalesce(0),
        "cos": lambda x: x.cos(),
        "cosh": lambda x: x.cosh(),
        "count": lambda x: x.count(),
        "cumcount": lambda x: x.cumcount(),
        "cummax": lambda x: x.cummax(),
        "cummin": lambda x: x.cummin(),
        "cumprod": lambda x: x.cumprod(),
        "cumsum": lambda x: pl.cumsum(x),  # https://stackoverflow.com/a/73950822/6901725 , but not working yet
        "datetime_to_date": lambda x: x.datetime_to_date(),
        "dayofmonth": lambda x: x.dayofmonth(),
        "dayofweek": lambda x: x.dayofweek(),
        "dayofyear": lambda x: x.dayofyear(),
        "exp": lambda x: x.exp(),
        "expm1": lambda x: x.expm1(),
        "ffill": lambda x: x.ffill(),
        "first": lambda x: x.first(),
        "floor": lambda x: x.floor(),
        "format_date": lambda x: x.format_date(),
        "format_datetime": lambda x: x.format_datetime(),
        "is_bad": lambda x: x.is_bad(),
        "is_inf": lambda x: x.is_inf(),
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
        "nunique": lambda x: x.nunique(),
        "parse_date": lambda x: x.parse_date(),
        "parse_datetime": lambda x: x.parse_datetime(),
        "quarter": lambda x: x.quarter(),
        "rank": lambda x: x.rank(),
        "round": lambda x: x.round(),
        "shift": lambda x: x.shift(),
        "sign": lambda x: x.sign(),
        "sin": lambda x: x.sin(),
        "sinh": lambda x: x.sinh(),
        "size": lambda x: x.size(),
        "sqrt": lambda x: x.sqrt(),
        "std": lambda x: x.std(),
        "sum": lambda x: x.sum(),
        "tanh": lambda x: x.tanh(),
        "var": lambda x: x.var(),
        "weekofyear": lambda x: x.weekofyear(),
    }
    impl_map_2 = {
        "*": lambda a, b: a * b,
        "**": lambda a, b: a ** b,
        "/": lambda a, b: a / b,
        "//": lambda a, b: a // b,
        "%": lambda a, b: a % b,
        "%/%": lambda a, b: a / b,
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "&": lambda a, b: a & b,
        "and": lambda a, b: a & b,
        "around": lambda a, b: a.around(b),
        "coalesce": lambda a, b: a.coalesce(b),
        "concat": lambda a, b: a.concat(b),
        "date_diff": lambda a, b: a.date_diff(b),
        "fmax": lambda a, b: a.fmax(b),
        "fmin": lambda a, b: a.fmin(b),
        "is_in": lambda a, b: a.is_in(b),
        "maximum": lambda a, b: a.maximum(b),
        "minimum": lambda a, b: a.minimum(b),
        "mod": lambda a, b: a % b,
        "|": lambda a, b: a | b,
        "or": lambda a, b: a | b,
        "remainder": lambda a, b: a.remainder(b),
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
    }
    impl_map_3 = {
        "if_else": lambda a, b, c: pl.when(a).then(b).otherwise(c),
        "mapv": lambda a, b, c: a.mapv(b, c),
        "trimstr": lambda a, b, c: a.trimstr(b, c),
        "where": lambda a, b, c: pl.when(a).then(b).otherwise(c),
    }
    impl_map = {
        0: impl_map_0,
        1: impl_map_1,
        2: impl_map_2,
        3: impl_map_3,
    }
    return impl_map


class PolarsModel(data_algebra.data_model.DataModel):
    """
    Interface for realizing the data algebra as a sequence of steps over Polars https://www.pola.rs .

    Note: not fully implemented yet.
    """

    use_lazy_eval: bool
    presentation_model_name: str
    _method_dispatch_table: Dict[str, Callable]
    _expr_impl_map: Dict[int, Dict[str, Callable]]

    def __init__(self, *, use_lazy_eval: bool = True):
        data_algebra.data_model.DataModel.__init__(
            self, presentation_model_name="Polars"
        )
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

    def data_frame(self, arg=None):
        """
        Build a new data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """
        if arg is None:
            return pl.DataFrame()
        return pl.DataFrame(arg)

    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
        return isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame)

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
        inputs = [self._compose_polars_ops(s, data_map=data_map) for s in op.sources]
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
        res = res.to_pandas()  # TODO: new Pandas/Polars re-impl of cdata
        res = op.record_map.transform(res, local_data_model=data_algebra.data_model.default_data_model())
        res = pl.DataFrame(res)
        if self.use_lazy_eval and (not isinstance(res, pl.LazyFrame)):
            res = res.lazy()
        return res
    
    def _extend_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an extend step, returning a data frame.
        """
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        assert len(op.order_by) == 0  # TODO: implement non-zero version of this
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        partition_by = op.partition_by
        temp_v_columns = []
        # see if we need to make partition non-empty
        if len(partition_by) <= 0:
            v_name = f"_da_extend_temp_partition_column"
            partition_by = [v_name]
            temp_v_columns.append(pl.lit(1).alias(v_name))
        produced_columns = []
        for k, opk in op.ops.items():
            if op.windowed_situation:
                # enforce is a simple v.f() expression
                assert isinstance(opk, data_algebra.expr_rep.Expression)
                assert len(opk.args) == 1
                assert isinstance(opk.args[0], (data_algebra.expr_rep.Value, data_algebra.expr_rep.ColumnReference))
                if isinstance(opk.args[0], data_algebra.expr_rep.Value):
                    # promote value to column for uniformity of API
                    v_name = f"_da_extend_temp_v_column_{len(temp_v_columns)}"
                    temp_v_columns.append(pl.lit(opk.args[0].value).alias(v_name))
                    opk = data_algebra.expr_rep.Expression(
                        op=opk.op, 
                        args=[data_algebra.expr_rep.ColumnReference(column_name=v_name)], 
                        params=opk.params, 
                        inline=opk.inline, 
                        method=opk.method,
                    )
            fld_k_container = opk.act_on(res, data_model=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            fld_k = fld_k_container.polars_term
            if op.windowed_situation:
                fld_k = fld_k.over(partition_by)
            produced_columns.append(fld_k.alias(k))
        if len(produced_columns) > 0:
            if len(op.order_by) > 0:
                order_cols = list(partition_by)
                partition_set = set(partition_by)
                for c in op.order_by:
                    if c not in partition_set:
                        order_cols.append(c)
                reversed_cols = [True if ci in set(op.reverse) else False for ci in op.order_columns]
                res = res.sort(by=op.order_columns, reverse=reversed_cols)
            if len(temp_v_columns) > 0:
                res = res.with_columns(temp_v_columns)
            res = res.with_columns(produced_columns)  
            if len(temp_v_columns) > 0:
                res = res.select(op.columns_produced())
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
        res = inputs[0].join(
            inputs[1],
            left_on=op.on_a,
            right_on=op.on_b,
            how=op.jointype,
            suffix = "_da_right_tmp",
        )
        coalesce_columns = set(op.sources[0].columns_produced()).intersection(op.sources[1].columns_produced())
        if len(coalesce_columns) > 0:
            res = res.with_columns([
                pl.when(pl.col(c).is_null())
                    .then(pl.col(c + "_da_right_tmp"))
                    .otherwise(pl.col(c))
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
            v_name = f"_da_project_temp_group_by_column"
            group_by = [v_name]
            temp_v_columns.append(pl.lit(1).alias(v_name))
        produced_columns = []
        for k, opk in op.ops.items():
            # enforce is a simple v.f() expression
            assert isinstance(opk, data_algebra.expr_rep.Expression)
            assert len(opk.args) == 1
            assert isinstance(opk.args[0], (data_algebra.expr_rep.Value, data_algebra.expr_rep.ColumnReference))
            if isinstance(opk.args[0], data_algebra.expr_rep.Value):
                # promote value to column for uniformity of API
                v_name = f"_da_project_temp_v_column_{len(temp_v_columns)}"
                temp_v_columns.append(pl.lit(opk.args[0].value).alias(v_name))
                opk = data_algebra.expr_rep.Expression(
                    op=opk.op, 
                    args=[data_algebra.expr_rep.ColumnReference(column_name=v_name)], 
                    params=opk.params, 
                    inline=opk.inline, 
                    method=opk.method,
                )
            fld_k_container = opk.act_on(res, data_model=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            fld_k = fld_k_container.polars_term
            produced_columns.append(fld_k.alias(k))
        if len(temp_v_columns) > 0:
            res = res.with_columns(temp_v_columns)
        res = res.groupby(group_by).agg(produced_columns)
        if len(temp_v_columns) > 0:
            res = res.select(op.columns_produced())
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
        res = res.rename(columns=op.reverse_mapping)
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
        res = res.rename(columns=op.column_remapping)
        if (op.column_deletions is not None) and (len(op.column_deletions) > 0):
            column_selection = [c for c in res.columns if c not in op.column_deletions]
            res = res.select(column_selection)
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
        selection = op.expr.act_on(res, data_model=self)  # PolarsTerm
        assert isinstance(selection, PolarsTerm)
        res = res.filter(selection.polars_term)
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
        res = db_handle.read_query("\n".join(op.sql))
        res = pl.DataFrame(res)
        if self.use_lazy_eval and (not isinstance(res, pl.LazyFrame)):
            res = res.lazy()
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
    
    # expression helpers
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        assert not isinstance(value, PolarsTerm)
        return PolarsTerm(polars_term=pl.lit(value), is_literal=True)
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on
        """
        assert isinstance(value, str)
        return PolarsTerm(polars_term=pl.col(value), is_literal=False)
    
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
        for v in values:
            assert isinstance(v, PolarsTerm)
        f = self._expr_impl_map[len(values)][op.op]
        assert f is not None
        res = f(*[v.polars_term for v in values])
        return PolarsTerm(polars_term=res, is_literal=False)


def register_polars_model(key:Optional[str] = None):
    # register data model
    common_key = "default_Polars_model"
    if common_key not in data_algebra.data_model.data_model_type_map.keys():
        pd_model = PolarsModel()
        data_algebra.data_model.data_model_type_map[common_key] = pd_model
        data_algebra.data_model.data_model_type_map["<class 'polars.internals.dataframe.frame.DataFrame'>"] = pd_model
        data_algebra.data_model.data_model_type_map[str(type(pd_model.data_frame()))] = pd_model
        data_algebra.data_model.data_model_type_map["<class 'polars.internals.lazyframe.frame.LazyFrame'>"] = pd_model
        data_algebra.data_model.data_model_type_map[str(type(pd_model.data_frame().lazy()))] = pd_model
        if key is not None:
            assert isinstance(key, str)
            data_algebra.data_model.data_model_type_map[key] = pd_model
