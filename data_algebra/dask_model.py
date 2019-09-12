
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
        res = op.sources[0].eval_pandas_implementation(data_map=data_map,
                                                       eval_env=eval_env,
                                                       pandas_model=self)
        index_col_name = "_data_algebra_orig_index"
        res[index_col_name] = res.index
        res = res.set_index(res[index_col_name])
        # see: https://github.com/WinVector/data_algebra/blob/master/Examples/dask/dask_window_fn.ipynb
        for (k, opk) in op.ops.items():
            # work on a slice of the data frame
            col_list = [c for c in set(op.partition_by)]
            for c in op.order_by:
                if c not in col_list:
                    col_list = col_list + [c]
            if len(opk.args) > 0:
                raise RuntimeError("ExtendNode on dask doesn't support window-function arguments yet: " +
                                   str(k) + ": " + str(opk))
            if len(op.reverse) > 0:
                raise RuntimeError("ExtendNode on dask doesn't support reverse sorting yet")
            subframe = res.loc[:, col_list]
            subframe[index_col_name] = subframe.index
            if len(op.order_by) > 0:
                subframe = subframe.set_index(subframe[op.order_by[0]])
            if len(op.partition_by) > 0:
                opframe = subframe.groupby(op.partition_by)
                #  Groupby preserves the order of rows within each group.
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
            else:
                opframe = subframe
            if opk.op == "row_number":
                subframe[k] = opframe.cumcount() + 1
            else:  # TODO: more of these
                raise KeyError("not implemented: " + str(k) + ": " + str(opk))
            subframe = subframe.set_index(subframe[index_col_name])
            res[k] = subframe[k]
        return res.reset_index(drop=True)

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
