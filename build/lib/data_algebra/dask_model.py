import data_algebra
import data_algebra.data_ops
import data_algebra.pandas_model


try:
    # noinspection PyUnresolvedReferences
    import dask

    # noinspection PyUnresolvedReferences
    import dask.dataframe
except ImportError:
    pass


# Some dask notes:
#  https://examples.dask.org/dataframes/02-groupby.html#
#  https://stackoverflow.com/questions/43207926/groupby-transform-doesnt-work-in-dask-dataframe
#  https://github.com/dask/dask/issues/2536
#


class DaskModel(data_algebra.pandas_model.PandasModel):
    def __init__(self):
        data_algebra.pandas_model.PandasModel.__init__(self)

    def assert_is_appropriate_data_instance(self, df, arg_name=""):
        if not isinstance(df, dask.dataframe.DataFrame):
            raise TypeError(arg_name + " was supposed to be a dask.dataframe.DataFrame")

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
        columns_using = op.column_names
        if op.columns_currently_used is not None and len(op.columns_currently_used) > 0:
            columns_using = [c for c in op.columns_currently_used]
        missing = set(columns_using) - set([c for c in df.columns])
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        res = df.loc[:, columns_using]
        res = res.reset_index(drop=True)
        return res

    def columns_to_frame(self, cols):
        def f(k, v):
            r = v.to_frame()
            r.columns = [k]
            return r

        col_values = [f(k, v) for (k, v) in cols.items()]
        res = dask.dataframe.concat(col_values, axis=1)
        return res

    def extend_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.ExtendNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        window_situation = (len(op.partition_by) > 0) or (len(op.order_by) > 0)
        if not window_situation:
            return super().extend_step(op=op, data_map=data_map, eval_env=eval_env)
        self.check_extend_window_fns(op)
        if len(op.partition_by) > 1:
            raise RuntimeError(
                "ExtendNode doesn't support more than one partition column over dask yet"
            )
        if len(op.order_by) > 1:
            raise RuntimeError(
                "ExtendNode doesn't support more than one order column over dask yet"
            )
        if len(op.reverse) > 0:
            raise RuntimeError("ExtendNode on dask doesn't support reverse sorting yet")
        # get incoming data
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        # move to case where we have exactly one ordering column and one grouping column
        row_id_col = "_data_algebra_temp_row_id"

        res = res.reset_index(drop=True)
        # build unique row IDs
        res[row_id_col] = 1
        res[row_id_col] = res.groupby(row_id_col).cumcount()
        res = res.set_index(res[row_id_col])
        columns_to_remove = [row_id_col]
        if len(op.partition_by) < 1:
            group_col = "_data_algebra_temp_group"
            columns_to_remove.append(group_col)
            res[group_col] = 1
        else:
            group_col = op.partition_by[0]
        if len(op.order_by) < 1:
            order_col = "_data_algebra_temp_order"
            columns_to_remove.append(order_col)
            res[order_col] = res.index
        else:
            order_col = op.order_by[0]
        # see: https://github.com/WinVector/data_algebra/blob/master/Examples/dask/dask_window_fn.ipynb
        for (k, opk) in op.ops.items():
            result_col = k
            if len(opk.args) > 0:
                # aggregate column case
                value_col = opk.args[0].to_pandas()

                dsub = res.loc[:, [group_col, value_col, row_id_col]]
                if opk.op == "sum":
                    dagg = dsub.groupby(dsub[group_col]).sum()
                else:  # TODO: implement more of these
                    raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                dagg = dagg.drop([row_id_col], axis=1)
                dagg.columns = [result_col]
                dsub = dsub.drop([value_col], axis=1)
                dsub = dsub.join(dagg, on=[group_col])
                dsub = dsub.set_index(dsub[row_id_col])
                res[result_col] = dsub[result_col]
            else:
                # free window case such as rownumber or count
                dsub = res.loc[:, [group_col, order_col, row_id_col]]
                dsub = dsub.set_index(dsub[order_col])
                if opk.op == "row_number":
                    dsub[result_col] = dsub.groupby(group_col).cumcount() + 1
                else:  # TODO: implement more of these
                    raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                dsub = dsub.set_index(dsub[row_id_col])
                res[result_col] = dsub[result_col]
        for c in columns_to_remove:
            if c in res.columns:
                res = res.drop(c, axis=1)
        res = res.reset_index(drop=True)
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
        common_cols = set([c for c in left.columns]).intersection(
            [c for c in right.columns]
        )
        res = dask.dataframe.merge(
            left=left,
            right=right,
            how=op.jointype.lower(),
            on=op.by,
            suffixes=("", "_tmp_right_col"),
        )
        res = res.reset_index(drop=True)
        for c in common_cols:
            if c not in op.by:
                is_null = res[c].isnull()
                res.loc[is_null, c] = res.loc[is_null, c + "_tmp_right_col"]
                res = res.drop(c + "_tmp_right_col", axis=1)
        res = res.reset_index(drop=True)
        return res

    def order_rows_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.OrderRowsNode):
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        if len(op.order_columns) > 1:
            raise RuntimeError(
                "sorting doesn't support more than one order column in dask yet"
            )
        if len(op.reverse) > 0:
            raise RuntimeError("sorting doesn't support reverse in dask yet")
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        res.set_index(op.order_columns[0])  # may cause problems in later steps
        return res

    def convert_records_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("convert_records not implemented for dask yet")
