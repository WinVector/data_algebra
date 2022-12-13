
"""
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: fully not implemented yet.
"""

from typing import Any, Callable, Dict, List, Optional, Set

import polars as pl

import data_algebra
import data_algebra.data_model
import data_algebra.util
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components


class PolarsTerm:
    """
    Class to carry Polars expression term and annotations about expression tree.
    """

    polars_term: Any
    is_literal: bool
    is_column: bool
    collect_required: bool = False  # property of tree, not node
    one_constant_required: bool  # property of tree, not node

    def __init__(
        self, 
        *, 
        polars_term = None, 
        is_literal: bool = False,
        is_column: bool = False,
        collect_required: bool = False,  # property of tree, not node
        one_constant_required: bool = False,  # property of tree, not node
        inputs: Optional[List] = None,
        lit_value = None,
        ) -> None:
        """
        Carry a Polars expression term (polars_term) plus annotations.

        :param polars_term: Optional Polars expression (None means collect info, not a true term)
        :param is_literal: True if term is a constant
        :param is_column: True if term is a column name
        :param collect_required: True if Polars frame collection required by this node or an input node
        :param one_constant_required: True one constant required by this node or an input node
        :param lit_value: original value for a literal
        :param inputs: inputs to expression node
        """
        assert isinstance(is_literal, bool)
        assert isinstance(is_column, bool)
        assert isinstance(collect_required, bool)
        assert isinstance(one_constant_required, bool)
        assert (is_literal + is_column + (inputs is not None) + (polars_term is None)) == 1
        if lit_value is not None:
            assert is_literal
        self.lit_value = lit_value
        self.polars_term = polars_term
        self.is_literal = is_literal
        self.collect_required = collect_required
        self.one_constant_required = one_constant_required
        if inputs is not None:
            assert isinstance(inputs, List)
            for v in inputs:
                self.observe(v)
    
    def observe(self, v) -> None:
        """
        Or conditions of v into our conditions
        """
        assert isinstance(v, PolarsTerm)
        if v.collect_required:
            self.collect_required = True
        if v.one_constant_required:
            self.one_constant_required = True


def _raise_not_impl(nm: str):   # TODO: get rid of this
    raise ValueError(f" {nm} not implemented for Polars adapter, yet")


_da_temp_one_column_name = "_da_temp_one_column"
_da_temp_one_column = pl.lit(1).alias(_da_temp_one_column_name)


def _populate_expr_impl_map() -> Dict[int, Dict[str, Callable]]:
    """
    Map symbols to implementations.
    """
    # TODO: fill in more
    impl_map_0 = {
        "count": lambda : _da_temp_one_column.cumsum(),  # ugly SQL def
        "_count": lambda : _da_temp_one_column.cumsum(),  # ugly SQL def
        "cumcount": lambda : _da_temp_one_column.cumsum(),  # ugly SQL def
        "_cumcount": lambda : _da_temp_one_column.cumsum(),  # ugly SQL def
        "ngroup": lambda : _raise_not_impl("ngroup"),  # TODO: implement
        "_ngroup": lambda : _raise_not_impl("_ngroup"),  # TODO: implement
        "row_number": lambda : _da_temp_one_column.cumsum(),
        "_row_number": lambda : _da_temp_one_column.cumsum(),
        "size": lambda : _da_temp_one_column.sum(),
        "_size": lambda : _da_temp_one_column.sum(),
        "uniform": lambda : _raise_not_impl("uniform"),  # TODO: implement
        "_uniform": lambda : _raise_not_impl("_uniform"),  # TODO: implement
        # how to land new columns: https://github.com/pola-rs/polars/issues/3933#issuecomment-1179241568
        # .with_column(
        #  pl.Series(
        #     name="random_nbr",
        #     values=np.random.default_rng().uniform(0.0, 1.0, df.height),
        # )
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
        "as_int64": lambda x: x.cast(int),
        "as_str": lambda x: x.cast(str),
        "base_Sunday": lambda x: x.base_Sunday(),
        "bfill": lambda x: x.bfill(),
        "ceil": lambda x: x.ceil(),
        "coalesce0": lambda x: x.coalesce(0),
        "cos": lambda x: x.cos(),
        "cosh": lambda x: x.cosh(),
        "count": lambda x: x.count(),
        "cumcount": lambda x: x.cumcount(),
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
        "ffill": lambda x: x.ffill(),
        "first": lambda x: x.first(),
        "floor": lambda x: x.floor(),
        "format_date": lambda x: x.format_date(),
        "format_datetime": lambda x: x.format_datetime(),
        "is_bad": lambda x: x.is_null(),  # TODO: need a different def for numeric v.s. char columns?
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
        # datetime parsing from https://stackoverflow.com/a/71759536/6901725
        # TODO: figure out why format is wrong type
        # TODO: wire up format
        "parse_date": lambda x, format : x.cast(str).str.strptime(pl.Date, fmt=format, strict=False).cast(pl.Date),
        # TODO: wire up format
        "parse_datetime": lambda x, format : x.cast(str).str.strptime(pl.Datetime, fmt=format, strict=False).cast(pl.Datetime),
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
    # could also key the map by grouped, partitioned, regular situation
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
    _collect_required: Set[str]

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
        self._want_literals_unpacked = {"parse_date", "parse_datetime"}
        self._collect_required = set()

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
    
    def clean_copy(self, df):
        """
        Copy of data frame without indices.
        """
        assert self.is_appropriate_data_instance(df)
        # Polars doesn't need explicit copying due to copy on write semantics
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df
    
    def drop_indices(self, df) -> None:
        """
        Drop indices in place.
        """
        assert self.is_appropriate_data_instance(df)
        # no operation needed
    
    def bad_column_positions(self, x):
        """
        Return vector indicating which entries are bad (null or nan) (vectorized).
        """
        _raise_not_impl("bad_column_positions")  # TODO: implement

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
        assert len(inputs) == 2
        if op.id_column is not None:
            inputs[0] = inputs[0].with_column(pl.lit(op.a_name).alias(op.id_column))
            inputs[1] = inputs[1].with_column(pl.lit(op.b_name).alias(op.id_column))
        res = pl.concat(inputs, how="vertical")
        return res
    
    def _convert_records_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute record conversion step, returning a data frame.
        """
        _raise_not_impl("_convert_records_step")  # TODO: implement
    
    def _extend_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an extend step, returning a data frame.
        """
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        partition_by = op.partition_by
        conditions_from_expressions = PolarsTerm()
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
                if len(opk.args) == 0:
                    pass
                elif len(opk.args) == 1:
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
                else:
                    raise ValueError(f"can't take {len(opk.args)} arity argument in windowed extend")
            fld_k_container = opk.act_on(res, data_model=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            conditions_from_expressions.observe(fld_k_container)
            fld_k = fld_k_container.polars_term
            if op.windowed_situation:
                fld_k = fld_k.over(partition_by)
            produced_columns.append(fld_k.alias(k))
        if conditions_from_expressions.one_constant_required:
            temp_v_columns.append(_da_temp_one_column)
        assert not conditions_from_expressions.collect_required  # implement if needed
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
        return res

    def _project_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a project step, returning a data frame.
        """
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        group_by = op.group_by
        conditions_from_expressions = PolarsTerm()
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
            if len(opk.args) == 0:
                pass
            elif len(opk.args) == 1:
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
            else:
                raise ValueError(f"can't take {len(opk.args)} arity argument in project")
            fld_k_container = opk.act_on(res, data_model=self)  # PolarsTerm
            assert isinstance(fld_k_container, PolarsTerm)
            conditions_from_expressions.observe(fld_k_container)
            fld_k = fld_k_container.polars_term
            produced_columns.append(fld_k.alias(k))
        if conditions_from_expressions.one_constant_required:
            temp_v_columns.append(_da_temp_one_column)
        assert not conditions_from_expressions.collect_required  # implement if needed
        if len(temp_v_columns) > 0:
            res = res.with_columns(temp_v_columns)
        res = res.groupby(group_by).agg(produced_columns)
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
            how=op.jointype.lower(),
            suffix = "_da_right_tmp",
        )
        coalesce_columns = set(op.sources[0].columns_produced()).intersection(op.sources[1].columns_produced()) - set(op.on_a)
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

    def _rename_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a rename columns step, returning a data frame.
        """
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        res = res.rename(op.reverse_mapping)
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
        res = res.rename(op.column_remapping)
        if (op.column_deletions is not None) and (len(op.column_deletions) > 0):
            res = res.select(op.columns_produced())
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
    
    # cdata transforms

    def blocks_to_rowrecs(self, data, *, blocks_in):
        """
        Convert a block record (record spanning multiple rows) into a rowrecord (record in a single row).

        :param data: data frame to be transformed
        :param blocks_in: cdata record specification
        :return: transformed data frame
        """
        _raise_not_impl("blocks_to_rowrecs")  # TODO: implement
    
    def rowrecs_to_blocks(
        self,
        data,
        *,
        blocks_out,
        check_blocks_out_keying: bool = False,
    ):
        """
        Convert rowrecs (single row records) into block records (multiple row records).

        :param data: data frame to transform.
        :param blocks_out: cdata record specification.
        :param check_blocks_out_keying: logical, if True confirm keying
        :return: transformed data frame
        """
        _raise_not_impl("blocks_to_rowrecs")  # TODO: implement
    
    # expression helpers
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        assert not isinstance(value, PolarsTerm)
        return PolarsTerm(polars_term=pl.lit(value), is_literal=True, lit_value=value)
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on
        """
        assert isinstance(value, str)
        return PolarsTerm(polars_term=pl.col(value), is_column=True)
    
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
        want_literals_unpacked = op.op in self._want_literals_unpacked
        args = [v.lit_value if (want_literals_unpacked and v.is_literal) else v.polars_term for v in values]
        res = f(*args)
        return PolarsTerm(
            polars_term=res,
            inputs=values,
            one_constant_required=len(values) == 0,
            collect_required=op.op in self._collect_required,
        )


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
