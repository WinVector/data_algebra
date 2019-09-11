
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
