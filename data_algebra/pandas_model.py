
from data_algebra.pandas_base import PandasModelBase


class PandasModel(PandasModelBase):
    def __init__(self, *, pd, presentation_model_name="pandas"):
        PandasModelBase.__init__(
            self, pd=pd, presentation_model_name=presentation_model_name
        )
