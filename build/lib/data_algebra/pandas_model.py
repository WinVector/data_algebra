"""
Adapter for Pandas API.
"""

import pandas as pd

import data_algebra
import data_algebra.data_model
from data_algebra.pandas_base import PandasModelBase


class PandasModel(PandasModelBase):
    """
    Realize the data algebra over pandas.
    """
    
    def __init__(self):
        PandasModelBase.__init__(self, pd=pd, presentation_model_name="pd")


def register_pandas_model():
    # set up what pandas supplier we are using
    default_data_model = PandasModel()
    data_algebra.data_model.data_model_type_map["default_data_model"] = default_data_model
    data_algebra.data_model.data_model_type_map[str(type(default_data_model.data_frame()))] = default_data_model
