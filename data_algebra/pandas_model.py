"""
Adapter for Pandas API.
"""

from data_algebra.pandas_base import PandasModelBase


class PandasModel(PandasModelBase):
    """
    Realize the data algebra over pandas.
    """

    def __init__(self):
        import pandas as pd
        PandasModelBase.__init__(self, pd=pd, presentation_model_name="pd")


# set up what pandas supplier we are using
default_data_model = PandasModel()
