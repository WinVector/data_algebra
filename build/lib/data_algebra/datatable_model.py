# https://github.com/h2oai/datatable
# https://datatable.readthedocs.io/en/latest/?badge=latest
# https://datatable.readthedocs.io/en/latest/using-datatable.html

try:
    # noinspection PyUnresolvedReferences
    import datatable
except ImportError:
    pass


import data_algebra
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops


expr_to_dt_expr_map = {
    "exp": lambda expr: datatable.exp(expr_to_dt_expr(expr.args[0])),
    "sum": lambda expr: datatable.sum(expr_to_dt_expr(expr.args[0])),
    "+": lambda expr: expr_to_dt_expr(expr.args[0]) + expr_to_dt_expr(expr.args[1]),
    "-": lambda expr: expr_to_dt_expr(expr.args[0]) - expr_to_dt_expr(expr.args[1]),
    "*": lambda expr: expr_to_dt_expr(expr.args[0]) * expr_to_dt_expr(expr.args[1]),
    "/": lambda expr: expr_to_dt_expr(expr.args[0]) / expr_to_dt_expr(expr.args[1]),
    "neg": lambda expr: -expr_to_dt_expr(expr.args[0]),
}


def expr_to_dt_expr(expr):
    if isinstance(expr, data_algebra.expr_rep.ColumnReference):
        return datatable.f[expr.column_name]
    if isinstance(expr, data_algebra.expr_rep.Value):
        return expr.value
    f = expr_to_dt_expr_map[expr.op]
    return f(expr)


class DataTableModel(data_algebra.data_model.DataModel):
    def __init__(self):
        data_algebra.data_model.DataModel.__init__(self)

    def assert_is_appropriate_data_instance(self, df, arg_name=""):
        if not isinstance(df, datatable.Frame):
            raise TypeError(arg_name + " was supposed to be a datatable.Frame")

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def table_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.TableDescription):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        if len(op.qualifiers) > 0:
            raise ValueError(
                "table descriptions used with eval_implementation() must not have qualifiers"
            )
        df = data_map[op.table_name]
        self.assert_is_appropriate_data_instance(df, "data_map[" + op.table_name + "]")
        # check all columns we expect are present
        columns_using = df.names
        missing = set(columns_using) - set([c for c in df.names])
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make a copy of the data to isolate side-effects and not deal with indices
        res = df[:, columns_using].copy()
        return res

    def extend_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.ExtendNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        window_situation = (len(op.partition_by) > 0) or (len(op.order_by) > 0)
        if window_situation:
            self.check_extend_window_fns(op)
        if window_situation:
            raise RuntimeError("windowed extend not implemented yet")  # TODO: implement
        # datatable doesn't seem to have per-group transform yet (other than the whole dataframe)
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        if len(op.order_by) > 0:
            ascending = [False if ci in set(op.reverse) else True for ci in op.order_by]
            if not all(ascending):
                raise RuntimeError(
                    "reverse isn't implemented for datatable yet"
                )  # TODO: implement
            syms = [datatable.f[c] for c in op.order_by]
            res = res.sort(*syms)
        if len(op.partition_by) > 0:
            for (col, expr) in op.ops.items():
                dt_expr = expr_to_dt_expr(expr)
                res[col] = res[:, {col: dt_expr}, datatable.by(*op.partition_by)][col]
        else:
            for (col, expr) in op.ops.items():
                dt_expr = expr_to_dt_expr(expr)
                res[col] = res[:, {col: dt_expr}][col]
        return res

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        keys = [k for k in cols.keys()]
        res = datatable.Frame(x=cols[keys[0]])
        res.names = [keys[0]]
        for i in range(1, len(keys)):
            k = keys[i]
            res[k] = cols[k]
        return res

    def project_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.ProjectNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        # check these are forms we are prepared to work with, and build an aggregation dictionary
        # build an agg list: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        # https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with
        for (k, opk) in op.ops.items():
            if len(opk.args) != 1:
                raise ValueError(
                    "non-trivial aggregation expression: " + str(k) + ": " + str(opk)
                )
            if not isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                raise ValueError(
                    "windows expression argument must be a column: "
                    + str(k)
                    + ": "
                    + str(opk)
                )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        cols = []
        if len(op.group_by) > 0:
            for (col, expr) in op.ops.items():
                dt_expr = expr_to_dt_expr(expr)
                cols.append(res[:, {col: dt_expr}, datatable.by(*op.group_by)][col])
        else:
            for (col, expr) in op.ops.items():
                dt_expr = expr_to_dt_expr(expr)
                cols.append(res[:, {col: dt_expr}][col])
        res = self.columns_to_frame(cols)
        return res

    def select_rows_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.SelectRowsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        raise RuntimeError("not implemented yet")  # TODO: implement

    def select_columns_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.SelectColumnsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        return res[:, op.column_names]

    def drop_columns_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.DropColumnsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.DropColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        column_selection = [c for c in res.names if c not in op.column_deletions]
        return res[:, column_selection]

    def order_rows_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.OrderRowsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        ascending = [
            False if ci in set(op.reverse) else True for ci in op.order_columns
        ]
        if not all(ascending):
            raise RuntimeError(
                "reverse isn't implemented for datatable yet"
            )  # TODO: implement
        syms = [datatable.f[c] for c in op.order_columns]
        return res.sort(*syms)

    def rename_columns_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.RenameColumnsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )

        def mp(n):
            try:
                return op.column_remapping[n]
            except KeyError:
                return n

        names = [mp(n) for n in res.names]
        res.names = names
        return res

    def natural_join_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.NaturalJoinNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        left = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        right = op.sources[1].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        common_cols = set([c for c in left.names]).intersection(
            [c for c in right.names]
        )
        raise RuntimeError("not implemented yet")  # TODO: implement

    def convert_records_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("convert_records not implemented for dask yet")
