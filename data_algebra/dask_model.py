
import data_algebra
import data_algebra.pandas_model


try:
    # noinspection PyUnresolvedReferences
    import dask
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
            r.columns= [k]
            return r

        col_values = [f(k, v) for (k, v) in cols.items()]
        res = dask.dataframe.concat(col_values, axis=1)
        return res
