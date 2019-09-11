
import data_algebra
import data_algebra.pandas_model


class DaskModel(data_algebra.pandas_model.PandasModel):
    def __init__(self):
        data_algebra.pandas_model.PandasModel.__init__(self)
