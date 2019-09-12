
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

    def columns_to_frame(self, cols):

        def f(k, v):
            r = v.to_frame()
            r.columns = [k]
            return r

        col_values = [f(k, v) for (k, v) in cols.items()]
        res = dask.dataframe.concat(col_values, axis=1)
        return res

    def project_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.ProjectNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        if len(op.order_by) > 0:
            raise RuntimeError("ProjectNode order_by not implemented for dask yet")  # TODO: implement
        return super().project_step(op=op, data_map=data_map, eval_env=eval_env)


    def extend_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.ExtendNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        window_situation = (len(op.partition_by) > 0) or (len(op.order_by) > 0)
        if not window_situation:
            return super().extend_step(op=op, data_map=data_map, eval_env=eval_env)
        self.check_extend_window_fns(op)
        if len(op.partition_by) > 1:
            raise RuntimeError("ExtendNode doesn't support more than one partition column over dask yet")
        if len(op.order_by) > 1:
            raise RuntimeError("ExtendNode doesn't support more than one order column over dask yet")
        if len(op.reverse) > 0:
            raise RuntimeError("ExtendNode on dask doesn't support reverse sorting yet")
        # get incoming data
        res = op.sources[0].eval_pandas_implementation(data_map=data_map,
                                                       eval_env=eval_env,
                                                       pandas_model=self)
        # move to case where we have exactly one ordering column and one grouping column
        temp_col = '_data_algebra_temp_index'
        res = res.reset_index(drop=True)
        res[temp_col] = res.index
        res = res.set_index(res[temp_col])
        columns_to_remove = [temp_col]
        if len(op.partition_by) < 1:
            group_col = '_data_algebra_temp_group'
            columns_to_remove.append(group_col)
            res[group_col] = 1
        else:
            group_col = op.partition_by[0]
        if len(op.order_by) < 1:
            order_col = '_data_algebra_temp_order'
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
                dsub = res.loc[:, [group_col, value_col]]
                dsub[temp_col] = dsub.index
                if opk.op == 'sum':
                    dagg = dsub.groupby(dsub[group_col]).sum()
                else:  # TODO: implement more of these
                    raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                dagg = dagg.reset_index(drop=False)
                dagg = dagg.set_index(dagg[group_col])
                dsub = dsub.set_index(dsub[group_col])
                dsub[result_col] = dagg[value_col]
                dsub = dsub.set_index(dsub[temp_col])
                res[result_col] = dsub[result_col]
            else:
                # free window case such as rownumber or count
                dsub = res.loc[:, [group_col, order_col]]
                dsub[temp_col] = dsub.index
                dsub = dsub.set_index(dsub[order_col])
                if opk.op == "row_number":
                    dsub[result_col] = dsub.groupby(group_col).cumcount() + 1
                else:  # TODO: implement more of these
                    raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                dsub = dsub.set_index(dsub[temp_col])
                res[result_col] = dsub[result_col]
        for c in columns_to_remove:
            res[c] = None
        res = res.reset_index(drop=True)
        return res

    def natural_join_step(self, op, *, data_map, eval_env):
        if not isinstance(op, data_algebra.data_ops.NaturalJoinNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.NaturalJoinNode")
        left = op.sources[0].eval_pandas_implementation(data_map=data_map,
                                                        eval_env=eval_env,
                                                        pandas_model=self)
        right = op.sources[1].eval_pandas_implementation(data_map=data_map,
                                                         eval_env=eval_env,
                                                         pandas_model=self)
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
                res[c][is_null] = res[c + "_tmp_right_col"]
                res = res.drop(c + "_tmp_right_col", axis=1)
        res = res.reset_index(drop=True)
        return res
