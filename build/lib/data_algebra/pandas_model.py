"""
Adapter for Pandas API.
"""


import pandas as pd

from data_algebra.pandas_base import PandasModelBase


class PandasModel(PandasModelBase):
    """
    Realize the data algebra over pandas.
    """

    def __init__(self):
        PandasModelBase.__init__(self, pd=pd, presentation_model_name="pd")
