
"""
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: not implemented yet.
"""

from typing import Any, Callable, Dict, List, Optional

import data_algebra
import data_algebra.data_model
import data_algebra.util
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components

import polars as pl


def _populate_expr_impl_map() -> Dict[str, Callable]:
    """
    Map symbols to implementations.
    """
    impl_map = {
        "==": lambda a, b: a == b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,  
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,  
        # TODO: fill in more
    }
    return impl_map


class PolarsModel(data_algebra.data_model.DataModel):
    """
    Interface for realizing the data algebra as a sequence of steps over Polars https://www.pola.rs .

    Note: not implemented yet.
    """

    use_lazy_eval: bool
    presentation_model_name: str
    _method_dispatch_table: Dict[str, Callable]
    _expr_impl_map: Dict[str, Callable]

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
        return isinstance(df, pl.DataFrame)

    # evaluate

    def eval(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any], narrow: bool = True) -> pl.DataFrame:
        """
        Implementation of Polars evaluation of data algebra operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :param narrow: ignored, always narrows to anticipated columns
        :return: data frame result
        """
        assert isinstance(data_map, Dict)
        assert isinstance(narrow, bool)
        assert isinstance(op, data_algebra.data_ops_types.OperatorPlatform)
        res = self._compose_polars_ops(op=op, data_map=data_map)
        if self.use_lazy_eval:
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

    def _concat_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a concat rows step, returning a data frame.
        """
        if op.node_name != "ConcatRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConcatRowsNode"
            )
        raise ValueError("not implemented yet")  # TODO: implement
    
    def _convert_records_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute record conversion step, returning a data frame.
        """
        raise ValueError("not implemented yet")  # TODO: implement

    def _drop_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a drop columns step, returning a data frame.
        """
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        res = res.select(op.columns_produced())
        return res
    
    def _extend_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an extend step, returning a data frame.
        """
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        raise ValueError("not implemented yet")  # TODO: implement
    
    def _map_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a map columns step, returning a data frame.
        """
        if op.node_name != "MapColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.MapColumnsNode"
            )
        raise ValueError("not implemented yet")  # TODO: implement
    
    def _natural_join_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a natural join step, returning a data frame.
        """
        if op.node_name != "NaturalJoinNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        raise ValueError("not implemented yet")  # TODO: implement
    
    def _order_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute an order rows step, returning a data frame.
        """
        if op.node_name != "OrderRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        raise ValueError("not implemented yet")  # TODO: implement

    def _project_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a project step, returning a data frame.
        """
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        raise ValueError("not implemented yet")  # TODO: implement

    def _map_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a map columns step, returning a data frame.
        """
        if op.node_name != "MapColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.MapColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        raise ValueError("not implemented yet")  # TODO: implement

    def _rename_columns_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a rename columns step, returning a data frame.
        """
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        raise ValueError("not implemented yet")  # TODO: implement

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

    def _select_rows_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Execute a select rows step, returning a data frame.
        """
        if op.node_name != "SelectRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = self._compose_polars_ops(op.sources[0], data_map=data_map)
        selection = op.expr.act_on(res, data_model=self)
        res = res.filter(selection)
        return res

    def _sql_proxy_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        execute SQL
        """
        if op.node_name != "SQLNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SQLNode"
            )
        raise ValueError("not implemented yet")  # TODO: implement
    
    def _table_step(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]):
        """
        Return a data frame from table description and data_map.
        """
        if op.node_name != "TableDescription":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        df = data_map[op.table_name]
        if not self.is_appropriate_data_instance(df):
            raise ValueError(
                "data_map[" + op.table_name + "] was not the right type"
            )
        if self.use_lazy_eval:
            df = df.lazy()
        df = df.select(op.columns_produced())
        return df
    
    # expression helpers
    
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
        return pl.lit(value)
    
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on
        """
        assert isinstance(value, str)
        return pl.col(value)
    
    def act_on_expression(self, *, arg, values: List, op):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param values: list of values to work on
        :param op: perator to apply
        :return: arg acted on
        """
        assert isinstance(values, List)
        assert isinstance(op, data_algebra.expr_rep.Expression)
        f = self._expr_impl_map[op.op]
        res = f(*values)
        return res


def register_polars_model():
    # register data model
    default_Polars_model = PolarsModel()
    data_algebra.data_model.data_model_type_map[str(type(default_Polars_model.data_frame()))] = default_Polars_model
