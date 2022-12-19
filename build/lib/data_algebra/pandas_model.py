"""
Adapter for Pandas API.
"""

from typing import Optional
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


def register_pandas_model(key:Optional[str] = None):
    # set up what pandas supplier we are using
    common_key = "default_data_model"
    if common_key not in data_algebra.data_model.data_model_type_map.keys():
        pd_model = PandasModel()
        data_algebra.data_model.data_model_type_map[common_key] = pd_model
        data_algebra.data_model.data_model_type_map["default_Pandas_model"] = pd_model
        data_algebra.data_model.data_model_type_map["<class 'pandas.core.frame.DataFrame'>"] = pd_model
        data_algebra.data_model.data_model_type_map[str(type(pd_model.data_frame()))] = pd_model
        if key is not None:
            assert isinstance(key, str)
            data_algebra.data_model.data_model_type_map[key] = pd_model
