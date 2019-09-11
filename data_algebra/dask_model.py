
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
            raise RuntimeError("ProjectNode order_by not implemented for dask yet")
        return super().project_step(op=op, data_map=data_map, eval_env=eval_env)

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
