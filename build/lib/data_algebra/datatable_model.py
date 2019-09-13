
try:
    # noinspection PyUnresolvedReferences
    import datatable
except ImportError:
    pass


import data_algebra
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops


class DataTableModel(data_algebra.data_model.DataModel):
    def __init__(self):
        data_algebra.data_model.DataModel.__init__(self)

    def assert_is_appropriate_data_instance(self, df, arg_name=''):
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
        self.assert_is_appropriate_data_instance(df, 'data_map[' + op.table_name + ']')
        # check all columns we expect are present
        columns_using = df.names
        missing = set(columns_using) - set([c for c in df.columns])
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
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        raise RuntimeError("not implemented yet")  # TODO: implement

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        raise RuntimeError("not implemented yet")  # TODO: implement

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
        raise RuntimeError("not implemented yet")  # TODO: implement

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
        raise RuntimeError("not implemented yet")  # TODO: implement

    def drop_columns_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.DropColumnsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.DropColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        column_selection = [c for c in res.columns if c not in op.column_deletions]
        raise RuntimeError("not implemented yet")  # TODO: implement

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
        raise RuntimeError("not implemented yet")  # TODO: implement

    def rename_columns_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.RenameColumnsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        raise RuntimeError("not implemented yet")  # TODO: implement

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
        common_cols = set([c for c in left.columns]).intersection(
            [c for c in right.columns]
        )
        raise RuntimeError("not implemented yet")  # TODO: implement
